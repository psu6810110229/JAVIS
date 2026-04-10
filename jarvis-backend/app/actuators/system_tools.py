"""actuators/system_tools.py — Built-in Jarvis tools and Phase-4 actuator template.

Every function decorated with @tool is automatically registered in `tool_registry`
when this module is imported. Import this module from orchestrator.py to activate tools.

=======================================================================
HOW TO ADD A NEW ACTUATOR (Phase 4 Guide)
=======================================================================

1.  Define your function below — sync OR async, both work.
2.  Apply the @tool decorator with name, description, and a JSON Schema for parameters.
3.  That's it — the tool is available to the LLM automatically.

Example:
    @tool(
        name="open_application",
        description="Open a desktop application by name.",
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application to open (e.g. 'notepad').",
                }
            },
            "required": ["app_name"],
        },
    )
    async def open_application(app_name: str) -> dict:
        import subprocess
        subprocess.Popen(app_name)
        return {"launched": app_name}

=======================================================================
"""
from __future__ import annotations

import os
import logging
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import difflib
import psutil

from app.actuators.registry import tool
from app.brain.memory_guardian import MemoryGuardian, PAGEFILE_GUARDRAIL_PERCENT
from app.config.settings import (
    DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT,
    DEFAULT_LOW_RAM_FORCE_ECO_BYTES,
    Settings,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".cache",
}

_OPEN_APP_MAP: dict[str, str] = {
    "notepad": "notepad.exe",
    "vscode": "code.cmd",
    "visual studio code": "code.cmd",
    "code": "code.cmd",
    "chrome": "chrome.exe",
    "google": "chrome.exe",
    "google chrome": "chrome.exe",
    "word": "winword.exe",
    "winword": "winword.exe",
    "microsoft word": "winword.exe",
    "excel": "excel.exe",
    "microsoft excel": "excel.exe",
    "powerpnt": "powerpnt.exe",
    "powerpoint": "powerpnt.exe",
    "microsoft powerpoint": "powerpnt.exe",
    "point": "powerpnt.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "cmd": "cmd.exe",
    "taskmgr": "taskmgr.exe",
    "task manager": "taskmgr.exe",
}
_ALLOWED_APPS_STR = "notepad, vscode, chrome, word, excel, powerpoint, explorer, paint, calculator, terminal, taskmgr, cmd"

def _collect_top_processes(limit: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            _ = proc.info.get("name")
            _ = proc.info.get("memory_info")
            proc.cpu_percent(interval=None)
            candidates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    time.sleep(0.08)

    memory_processes: list[dict[str, Any]] = []
    cpu_processes: list[dict[str, Any]] = []
    for proc in candidates:
        try:
            proc_name = proc.name() or f"pid-{proc.pid}"
            mem_bytes = int(proc.memory_info().rss)
            cpu_pct = float(proc.cpu_percent(interval=None))
            memory_processes.append(
                {
                    "pid": int(proc.pid),
                    "name": proc_name,
                    "memory_rss_mb": round(mem_bytes / (1024**2), 2),
                }
            )
            cpu_processes.append(
                {
                    "pid": int(proc.pid),
                    "name": proc_name,
                    "cpu_percent": round(cpu_pct, 2),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    top_memory = sorted(
        memory_processes,
        key=lambda item: float(item["memory_rss_mb"]),
        reverse=True,
    )[:limit]
    top_cpu = sorted(
        cpu_processes,
        key=lambda item: float(item["cpu_percent"]),
        reverse=True,
    )[:limit]
    return top_memory, top_cpu


def _extract_first_percentage(text: str) -> float | None:
    for raw_line in text.splitlines():
        candidate = raw_line.strip()
        if not candidate:
            continue
        cleaned = candidate.replace("%", "")
        if cleaned.isdigit():
            return float(cleaned)
    return None


def _windows_battery_percent_fallback() -> float | None:
    if os.name != "nt":
        return None

    commands = [
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Battery | Select-Object -ExpandProperty EstimatedChargeRemaining -First 1)",
        ],
        ["wmic", "PATH", "Win32_Battery", "Get", "EstimatedChargeRemaining"],
    ]
    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1.2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        percent = _extract_first_percentage(output)
        if percent is not None and 0.0 <= percent <= 100.0:
            return percent

    return None


def _resolve_battery_status() -> dict[str, Any]:
    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            return {
                "battery_percent": float(battery.percent),
                "battery_source": "psutil",
                "battery_state": "charging" if bool(battery.power_plugged) else "discharging",
            }
    except Exception:
        pass

    try:
        fallback_percent = _windows_battery_percent_fallback()
        if fallback_percent is not None:
            return {
                "battery_percent": float(fallback_percent),
                "battery_source": "windows-native",
                "battery_state": "unknown",
            }
    except Exception:
        pass

    return {
        "battery_percent": "AC Power / No Battery Detected",
        "battery_source": "unavailable",
        "battery_state": "unknown",
    }


def _runtime_metrics_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    battery_status = _resolve_battery_status()
    top_memory_processes, top_cpu_processes = _collect_top_processes(limit=3)
    return {
        "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
        "ram_available_bytes": int(vm.available),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "pagefile_usage_percent": float(swap.percent),
        "battery_percent": battery_status["battery_percent"],
        "battery_source": battery_status["battery_source"],
        "battery_state": battery_status["battery_state"],
        "top_memory_processes": top_memory_processes,
        "top_cpu_processes": top_cpu_processes,
    }


def _guardian_for_system_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    settings = Settings()
    guardian = MemoryGuardian(
        low_ram_force_eco_bytes=DEFAULT_LOW_RAM_FORCE_ECO_BYTES,
        high_swap_force_eco_percent=DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT,
        pagefile_guardrail_percent=PAGEFILE_GUARDRAIL_PERCENT,
    )
    decision = guardian.assess(
        tool_name="get_system_status",
        tool_metadata={"estimated_ram_mb": 32, "required_mode": "either"},
        active_mode=settings.get_mode(),
        runtime_metrics={
            "ram_available_bytes": metrics["ram_available_bytes"],
            "pagefile_usage_percent": metrics["pagefile_usage_percent"],
            "host_swap_percent": metrics["pagefile_usage_percent"],
        },
    )
    return {
        "safe_to_execute": decision.allowed,
        "requires_mode_confirmation": decision.requires_mode_confirmation,
        "suggested_mode": decision.suggested_mode,
        "reason": decision.reason,
    }


def _resolve_local_app_path(app_name: str) -> Path | None:
    key = app_name.strip().lower()
    candidates = _OPEN_APP_WHITELIST.get(key)
    if not candidates:
        return None

    known_paths: list[Path] = []
    if key == "notepad":
        known_paths.append(Path("C:/Windows/System32/notepad.exe"))
    elif key == "vscode":
        local_app_data = os.getenv("LOCALAPPDATA", "")
        if local_app_data:
            known_paths.append(Path(local_app_data) / "Programs/Microsoft VS Code/Code.exe")
        known_paths.append(Path("C:/Program Files/Microsoft VS Code/Code.exe"))
        known_paths.append(Path("C:/Program Files (x86)/Microsoft VS Code/Code.exe"))

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if not resolved:
            continue
        path = Path(resolved).resolve()
        if path.is_file():
            return path

    for path in known_paths:
        if path.is_file():
            return path.resolve()

    return None


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

@tool(
    name="get_current_datetime",
    description=(
        "Return the current date and time for a requested IANA timezone. "
        "Use this whenever the user asks about the current time or date."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone_name": {
                "type": "string",
                "description": (
                    "An IANA timezone identifier such as 'UTC', 'Asia/Bangkok', "
                    "or 'America/New_York'. Defaults to 'UTC' if omitted or invalid."
                ),
            }
        },
        "required": [],
    },
)
def get_current_datetime(timezone_name: str = "UTC") -> dict[str, str]:
    """Return the current date/time in the requested timezone."""
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("UTC")
        timezone_name = "UTC"

    now = datetime.now(timezone)
    return {
        "timezone": timezone_name,
        "iso_datetime": now.isoformat(),
        "human_readable": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


@tool(
    name="get_system_status",
    description=(
        "Return current CPU usage, available RAM, pagefile usage, and battery percentage. "
        "Use this for quick local health checks before heavy actions."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    timeout_seconds=1.5,
    estimated_ram_mb=32,
    required_mode="either",
    risk_level="low",
    category="system",
)
def get_system_status() -> dict[str, Any]:
    """Return a compact local system snapshot and guardian assessment."""
    metrics = _runtime_metrics_snapshot()
    guardian = _guardian_for_system_snapshot(metrics)
    return {
        "cpu_percent": metrics["cpu_percent"],
        "ram_available_bytes": metrics["ram_available_bytes"],
        "ram_available_gb": metrics["ram_available_gb"],
        "pagefile_usage_percent": metrics["pagefile_usage_percent"],
        "battery_percent": metrics["battery_percent"],
        "battery_source": metrics["battery_source"],
        "battery_state": metrics["battery_state"],
        "top_memory_processes": metrics["top_memory_processes"],
        "top_cpu_processes": metrics["top_cpu_processes"],
        "guardian": guardian,
    }


@tool(
    name="list_project_files",
    description=(
        "List files from the current project directory with safe limits. "
        "Useful to understand project structure without reading file contents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "max_entries": {
                "type": "integer",
                "description": "Maximum number of paths to return (default 200, max 500).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum folder depth from project root (default 3, max 6).",
            },
        },
        "required": [],
    },
    timeout_seconds=2.0,
    estimated_ram_mb=48,
    required_mode="either",
    risk_level="low",
    category="filesystem",
)
def list_project_files(max_entries: int = 200, max_depth: int = 3) -> dict[str, Any]:
    """Return a bounded, relative file listing for the current project."""
    capped_entries = max(20, min(int(max_entries), 500))
    capped_depth = max(1, min(int(max_depth), 6))

    results: list[str] = []
    truncated = False

    for root, dirs, files in os.walk(_PROJECT_ROOT):
        root_path = Path(root)
        rel_root = root_path.relative_to(_PROJECT_ROOT)
        depth = len(rel_root.parts)
        if depth >= capped_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS and not d.startswith(".")]

        for file_name in files:
            if file_name.startswith("."):
                continue
            rel_path = (rel_root / file_name).as_posix() if rel_root.parts else file_name
            results.append(rel_path)
            if len(results) >= capped_entries:
                truncated = True
                break
        if truncated:
            break

    return {
        "root": _PROJECT_ROOT.as_posix(),
        "count": len(results),
        "truncated": truncated,
        "max_entries": capped_entries,
        "max_depth": capped_depth,
        "files": results,
    }


def _resolve_app_path(app_cmd: str, app_key: str) -> str | None:
    """Attempt to find the full path for common Windows applications. Return None if not found."""
    if os.path.exists(app_cmd):
        return app_cmd
        
    # Check system PATH first
    resolved = shutil.which(app_cmd)
    if resolved:
        return resolved

    # Common Windows installation paths
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    local_app_data = os.environ.get("LOCALAPPDATA", "")

    search_paths = []
    if app_key in ("chrome", "google", "google chrome"):
        search_paths = [
            f"{program_files}\\Google\\Chrome\\Application\\chrome.exe",
            f"{program_files_x86}\\Google\\Chrome\\Application\\chrome.exe",
        ]
    elif app_key in ("vscode", "code", "visual studio code"):
        search_paths = [
            f"{program_files}\\Microsoft VS Code\\bin\\code.cmd",
            f"{local_app_data}\\Programs\\Microsoft VS Code\\bin\\code.cmd",
        ]
    elif app_key in ("word", "winword", "microsoft word"):
        search_paths = [
             f"{program_files}\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
             f"{program_files_x86}\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        ]
    elif app_key in ("excel", "microsoft excel"):
        search_paths = [
             f"{program_files}\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
             f"{program_files_x86}\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
        ]
    elif app_key in ("powerpnt", "powerpoint", "microsoft powerpoint"):
        search_paths = [
             f"{program_files}\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
             f"{program_files_x86}\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
        ]
    elif app_key in ("terminal", "windows terminal"):
         search_paths = [f"{local_app_data}\\Microsoft\\WindowsApps\\wt.exe"]

    for path in search_paths:
        if os.path.exists(path):
            return path
            
    return None

@tool(
    name="open_local_app",
    description=(
        f"Launch an approved local application safely using fuzzy name matching. "
        f"Known apps: {_ALLOWED_APPS_STR}. You can try others, but they might be blocked."
    ),
    parameters={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": f"App identifier or name to launch (e.g., 'word', 'powerpoint', 'chrome'). Can be a fuzzy match.",
            }
        },
        "required": ["app_name"],
    },
    timeout_seconds=2.5,
    estimated_ram_mb=64,
    required_mode="either",
    risk_level="medium",
    category="system",
)
def open_local_app(app_name: str) -> str:
    """Launch a whitelisted local app with fuzzy path resolution."""
    key = app_name.strip().lower()
    
    # Fuzzy match
    matched_key = None
    if key in _OPEN_APP_MAP:
        matched_key = key
    else:
        close_matches = difflib.get_close_matches(key, _OPEN_APP_MAP.keys(), n=1, cutoff=0.5)
        if close_matches:
            matched_key = close_matches[0]

    if not matched_key:
        return f"Execution Failed: '{app_name}' is not recognized or allowed."

    raw_cmd = _OPEN_APP_MAP[matched_key]
    app_cmd = _resolve_app_path(raw_cmd, matched_key)

    if not app_cmd:
        return f"Execution Failed: '{app_name}' not found in system PATH."

    try:
        kwargs = {
            "close_fds": True,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            kwargs["creationflags"] = flags
            
            if app_cmd.lower().endswith((".cmd", ".bat")):
                subprocess.Popen(app_cmd, shell=True, **kwargs)
            else:
                try:
                    os.startfile(app_cmd)
                except Exception:
                    subprocess.Popen(app_cmd, shell=True, **kwargs)
        else:
            kwargs["start_new_session"] = True
            subprocess.Popen(app_cmd, shell=True, **kwargs)
    except Exception as err:
        logger.warning("Failed to launch app '%s' (resolved to '%s'): %s", key, app_cmd, err)
        return f"Execution Failed: {str(err)}"

    return f"Successfully opened {app_name}"


@tool(
    name="get_backend_status",
    description=(
        "Return a concise summary of the Jarvis backend runtime status "
        "including the active model and mode."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def get_backend_status() -> dict[str, str]:
    """Return a lightweight status snapshot without performing any I/O.

    Full system metrics (RAM, GPU, Ollama health) are available via
    GET /v1/system/status — not exposed as a tool to avoid inference-time blocking.
    """
    # Import lazily to avoid circular imports at module load time.
    from app.config.settings import Settings  # noqa: PLC0415

    settings = Settings()
    return {
        "service": "jarvis-backend",
        "mode": settings.get_mode(),
        "active_model": settings.get_active_model(),
        "ollama_base_url": settings.ollama_base_url,
        "intel_gpu_requested": str(settings.intel_gpu_requested).lower(),
        "status": "ready",
    }


@tool(
    name="switch_deployment_mode",
    description="Switch between 'performance' (8B model) and 'eco' (4B model) modes.",
    parameters={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "Target mode: 'performance' or 'eco'."
            }
        },
        "required": ["mode"]
    },
    risk_level="medium",
    category="system"
)
def switch_deployment_mode(mode: str) -> dict[str, Any]:
    """Change the active AI model and hardware profile."""
    from app.config.settings import Settings  # noqa: PLC0415
    settings = Settings()
    
    target = mode.lower().strip()
    if target not in ["performance", "eco"]:
        return {"error": f"Invalid mode '{mode}'. Use 'performance' or 'eco'."}

    old_mode = settings.get_mode()
    new_model = settings.set_mode(target)
    
    return {
        "switched": True,
        "old_mode": old_mode,
        "new_mode": target,
        "active_model": new_model,
        "message": f"Successfully switched to {target} mode."
    }


# ---------------------------------------------------------------------------
# File Explorer Tools
# ---------------------------------------------------------------------------

@tool(
    name="browse_filesystem",
    description="List files and folders in a specific directory on the host computer.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the directory to browse."
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum number of items to return (default 100)."
            }
        },
        "required": ["path"]
    },
    risk_level="low",
    category="filesystem"
)
def browse_filesystem(path: str, max_entries: int = 100) -> dict[str, Any]:
    """List contents of a directory."""
    target_path = Path(path)
    if not target_path.exists():
        return {"error": f"Path '{path}' does not exist."}
    if not target_path.is_dir():
        return {"error": f"Path '{path}' is not a directory."}

    capped_entries = max(10, min(int(max_entries), 500))
    results = []
    
    try:
        for index, item in enumerate(target_path.iterdir()):
            if index >= capped_entries:
                results.append({"truncated": True})
                break
            results.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
    except PermissionError:
        return {"error": f"Permission denied accessing '{path}'."}
    except Exception as e:
         return {"error": f"Error accessing '{path}': {e}"}

    return {
        "path": str(target_path.absolute()),
        "contents": results,
    }

@tool(
    name="open_path",
    description="Open a file or folder in its default Windows application (e.g., File Explorer for folders).",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file or folder to open."
            }
        },
        "required": ["path"]
    },
    risk_level="medium",
    category="system"
)
def open_path(path: str) -> dict[str, Any]:
    """Open a path using os.startfile."""
    target_path = Path(path)
    if not target_path.exists():
        return {"error": f"Cannot open '{path}': Path does not exist."}

    try:
        if os.name == "nt":
            os.startfile(str(target_path.absolute()))
        else:
            return {"error": "open_path is only supported natively on Windows."}
    except Exception as e:
        return {"error": f"Failed to open '{path}': {e}"}

    return {"launched": True, "path": str(target_path.absolute())}

@tool(
    name="search_files",
    description="Search for files recursively in a directory matching a pattern.",
    parameters={
        "type": "object",
        "properties": {
            "root_path": {
                "type": "string",
                "description": "Absolute path to the directory to start searching from."
            },
            "pattern": {
                "type": "string",
                "description": "File name pattern to search for (e.g., '*.txt', 'report*')."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matched paths to return (default 30)."
            }
        },
        "required": ["root_path", "pattern"]
    },
    risk_level="medium",
    category="filesystem"
)
def search_files(root_path: str, pattern: str, max_results: int = 30) -> dict[str, Any]:
    """Search for files in a directory recursively."""
    target_path = Path(root_path)
    if not target_path.exists() or not target_path.is_dir():
        return {"error": f"Root search path '{root_path}' does not exist or is not a directory."}

    capped_results = max(5, min(int(max_results), 100))
    matches = []
    
    try:
        # Using rglog which supports "**/" implicitly
        for match in target_path.rglob(pattern):
            # Basic exclusion check to avoid getting bogged down
            if any(excluded in match.parts for excluded in _EXCLUDED_DIRS):
                continue
            
            matches.append(str(match.absolute()))
            if len(matches) >= capped_results:
                break
    except PermissionError:
        return {"matches": matches, "warning": "Search hit permission denied on some child folders."}
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    return {
        "root": str(target_path.absolute()),
        "pattern": pattern,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# ─── ADD NEW ACTUATORS BELOW THIS LINE ───────────────────────────────────
#
# Phase 4 examples to implement here:
#   - set_volume(level: int)
#   - open_application(app_name: str)
#   - search_web(query: str)
#   - send_notification(title: str, body: str)
#   - read_clipboard() / write_clipboard(text: str)
# ---------------------------------------------------------------------------
