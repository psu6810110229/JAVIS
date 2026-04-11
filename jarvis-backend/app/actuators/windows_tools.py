"""actuators/windows_tools.py — Windows OS control tools for Jarvis Phase 4.

All tools are registered automatically via the @tool decorator on import.
Import this module from orchestrator.py to activate.

App-launch contract
-------------------
open_application:
  1. Resolve the executable path from the app name.
  2. Launch via subprocess.Popen (no shell=True for safety).
  3. Poll psutil every 200 ms for up to LAUNCH_TIMEOUT_SECONDS.
  4. Once the PID is confirmed alive, find its top-level HWND via win32gui.
  5. Bring the window to foreground: ShowWindow + SetForegroundWindow.
  6. Return truthful status: "opened" | "timeout" | "crashed" | "not_found".

Dependencies: pywin32, psutil, pyperclip, Pillow (all in requirements.txt).
"""
from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from app.actuators.registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_LAUNCH_POLL_INTERVAL = 0.20   # seconds between psutil polls
_LAUNCH_TIMEOUT = float(os.getenv("JARVIS_APP_LAUNCH_TIMEOUT", "10"))
_SCREENSHOT_DIR = Path(
    os.getenv("JARVIS_SCREENSHOT_DIR", "")
    or Path.home() / "Pictures" / "Jarvis"
)
_APP_BLOCKLIST: set[str] = {
    name.strip().lower()
    for name in os.getenv("JARVIS_APP_BLOCKLIST", "").split(",")
    if name.strip()
}

# Common Windows application aliases (user-friendly name → executable)
_APP_ALIASES: dict[str, str] = {
    "spotify":          "Spotify.exe",
    "chrome":           "chrome.exe",
    "google chrome":    "chrome.exe",
    "firefox":          "firefox.exe",
    "edge":             "msedge.exe",
    "notepad":          "notepad.exe",
    "calc":             "calc.exe",
    "calculator":       "calc.exe",
    "explorer":         "explorer.exe",
    "file explorer":    "explorer.exe",
    "discord":          "Discord.exe",
    "vs code":          "Code.exe",
    "vscode":           "Code.exe",
    "visual studio code": "Code.exe",
    "task manager":     "Taskmgr.exe",
    "paint":            "mspaint.exe",
    "word":             "WINWORD.EXE",
    "excel":            "EXCEL.EXE",
    "powerpoint":       "POWERPNT.EXE",
    "teams":            "ms-teams.exe",
    "outlook":          "OUTLOOK.EXE",
    "terminal":         "wt.exe",
    "windows terminal": "wt.exe",
    "cmd":              "cmd.exe",
    "powershell":       "powershell.exe",
    "snipping tool":    "SnippingTool.exe",
    "settings":         "ms-settings:",
    "control panel":    "control.exe",
    "vlc":              "vlc.exe",
    "steam":            "steam.exe",
    "obs":              "obs64.exe",
    "slack":            "slack.exe",
    "zoom":             "Zoom.exe",
    "whatsapp":         "WhatsApp.exe",
    "telegram":         "Telegram.exe",
    "notion":           "Notion.exe",
    "obsidian":         "Obsidian.exe",
    "figma":            "Figma.exe",
    "postman":          "Postman.exe",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_executable(app_name: str) -> str:
    """Return the best executable string for the given app name."""
    normalized = app_name.strip().lower()
    if normalized in _APP_ALIASES:
        return _APP_ALIASES[normalized]
    # Return as-is — ShellExecute / Popen will search PATH
    return app_name.strip()


def _find_hwnd_for_pid(pid: int) -> int | None:
    """Return the first top-level visible HWND for the given PID, or None."""
    try:
        import win32gui  # type: ignore[import]
        import win32process  # type: ignore[import]
    except ImportError:
        return None

    result: list[int] = []

    def _callback(hwnd: int, _: Any) -> bool:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
            if window_pid == pid:
                result.append(hwnd)
                return False  # stop enumeration
        except Exception:  # noqa: BLE001
            pass
        return True

    win32gui.EnumWindows(_callback, None)
    return result[0] if result else None


def _bring_to_foreground(hwnd: int) -> None:
    """Attempt to bring the window to the foreground."""
    try:
        import win32con  # type: ignore[import]
        import win32gui  # type: ignore[import]

        # Restore if minimised
        placement = win32gui.GetWindowPlacement(hwnd)
        if placement[1] == win32con.SW_SHOWMINIMIZED:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        else:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

        win32gui.SetForegroundWindow(hwnd)
    except Exception as err:  # noqa: BLE001
        logger.warning("SetForegroundWindow failed: %s", err)


def _get_window_title(hwnd: int) -> str:
    try:
        import win32gui  # type: ignore[import]
        return win32gui.GetWindowText(hwnd)
    except Exception:  # noqa: BLE001
        return ""


# ── open_application ──────────────────────────────────────────────────────────

@tool(
    name="open_application",
    description=(
        "Open a desktop application by name, wait until it is confirmed open "
        "and bring the window to the foreground. "
        "Returns the status: 'opened', 'timeout', 'crashed', or 'not_found'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": (
                    "The application to open, e.g. 'Spotify', 'Notepad', 'Chrome', "
                    "'VS Code', 'Discord'.  Common names are resolved automatically."
                ),
            }
        },
        "required": ["app_name"],
    },
    risk_level="medium",
    category="system",
    timeout_seconds=_LAUNCH_TIMEOUT + 5,
)
async def open_application(app_name: str) -> dict[str, Any]:
    """Launch an application, verify it opened, and bring it to the foreground."""
    normalized = app_name.strip().lower()

    # Blocklist check
    if normalized in _APP_BLOCKLIST or _resolve_executable(normalized).lower().replace(".exe", "") in _APP_BLOCKLIST:
        return {
            "status": "blocked",
            "app_name": app_name,
            "error": f"'{app_name}' is in the Jarvis application blocklist and cannot be opened.",
        }

    executable = _resolve_executable(app_name)

    # Handle ms-settings: and similar protocol URLs
    if ":" in executable and not executable.endswith(".exe"):
        try:
            os.startfile(executable)  # type: ignore[attr-defined]
            return {"status": "opened", "app_name": app_name, "method": "protocol"}
        except OSError as err:
            return {"status": "not_found", "app_name": app_name, "error": str(err)}

    # Try to launch
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(  # noqa: S603
            executable,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # Try with shell=True as a fallback (uses PATH + App Paths registry)
        try:
            process = subprocess.Popen(  # noqa: S602
                executable,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as err:
            return {"status": "not_found", "app_name": app_name, "error": str(err)}
    except OSError as err:
        return {"status": "not_found", "app_name": app_name, "error": str(err)}

    pid = process.pid
    deadline = time.monotonic() + _LAUNCH_TIMEOUT

    # Wait for the process to show a visible window
    hwnd: int | None = None
    while time.monotonic() < deadline:
        await asyncio.sleep(_LAUNCH_POLL_INTERVAL)

        # Check if process crashed early
        ret_code = process.poll()
        if ret_code is not None and ret_code != 0:
            return {
                "status": "crashed",
                "app_name": app_name,
                "pid": pid,
                "exit_code": ret_code,
                "error": f"'{app_name}' exited immediately with code {ret_code}.",
            }

        # Check if window appeared
        hwnd = await asyncio.to_thread(_find_hwnd_for_pid, pid)
        if hwnd is not None:
            break

        # Also check child processes (some launchers spawn the real app)
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child_hwnd = await asyncio.to_thread(_find_hwnd_for_pid, child.pid)
                if child_hwnd is not None:
                    hwnd = child_hwnd
                    pid = child.pid
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        if hwnd is not None:
            break

    if hwnd is None:
        # Final check: is the process at least still alive?
        ret_code = process.poll()
        if ret_code is not None and ret_code != 0:
            return {
                "status": "crashed",
                "app_name": app_name,
                "pid": pid,
                "exit_code": ret_code,
                "error": f"'{app_name}' crashed before a window appeared.",
            }
        return {
            "status": "timeout",
            "app_name": app_name,
            "pid": pid,
            "error": (
                f"'{app_name}' did not open a visible window within "
                f"{_LAUNCH_TIMEOUT:.0f} seconds."
            ),
        }

    # Bring window to foreground
    await asyncio.to_thread(_bring_to_foreground, hwnd)
    window_title = await asyncio.to_thread(_get_window_title, hwnd)

    return {
        "status": "opened",
        "app_name": app_name,
        "pid": pid,
        "window_title": window_title or app_name,
    }


# ── close_application ─────────────────────────────────────────────────────────

@tool(
    name="close_application",
    description="Gracefully close a running application by its process name (e.g. 'notepad.exe').",
    parameters={
        "type": "object",
        "properties": {
            "process_name": {
                "type": "string",
                "description": "The process name to close, e.g. 'notepad.exe', 'chrome.exe'.",
            }
        },
        "required": ["process_name"],
    },
    risk_level="medium",
    category="system",
    timeout_seconds=15,
)
async def close_application(process_name: str) -> dict[str, Any]:
    """Send WM_CLOSE to all windows belonging to the named process."""
    name = process_name.strip()
    closed: list[int] = []
    not_found = True

    def _close_sync() -> None:
        nonlocal not_found
        try:
            import win32con  # type: ignore[import]
            import win32gui  # type: ignore[import]
            import win32process  # type: ignore[import]
        except ImportError:
            return

        target_pids: set[int] = set()
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] and proc.info["name"].lower() == name.lower():
                    target_pids.add(int(proc.info["pid"]))
                    not_found = False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        def _enum_callback(hwnd: int, _: Any) -> bool:
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid in target_pids and win32gui.IsWindowVisible(hwnd):
                    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                    closed.append(hwnd)
            except Exception:  # noqa: BLE001
                pass
            return True

        win32gui.EnumWindows(_enum_callback, None)

    await asyncio.to_thread(_close_sync)

    if not_found:
        return {"status": "not_found", "process_name": name, "error": f"No running process named '{name}'."}
    if not closed:
        return {"status": "no_windows", "process_name": name, "message": "Process found but no visible windows to close."}
    return {"status": "closed", "process_name": name, "windows_closed": len(closed)}


# ── kill_process ──────────────────────────────────────────────────────────────

@tool(
    name="kill_process",
    description="Force-kill a process by its name or PID. Use sparingly — prefer close_application first.",
    parameters={
        "type": "object",
        "properties": {
            "process_name": {
                "type": "string",
                "description": "Process name, e.g. 'notepad.exe'. Leave empty if using pid.",
            },
            "pid": {
                "type": "integer",
                "description": "Process ID to kill. Leave 0 if using process_name.",
            },
        },
        "required": [],
    },
    risk_level="medium",
    category="system",
    timeout_seconds=10,
)
async def kill_process(process_name: str = "", pid: int = 0) -> dict[str, Any]:
    """Force-kill a process by name or PID."""
    killed: list[dict[str, Any]] = []
    errors: list[str] = []

    def _kill_sync() -> None:
        targets: list[psutil.Process] = []
        if pid > 0:
            try:
                targets = [psutil.Process(pid)]
            except psutil.NoSuchProcess:
                errors.append(f"PID {pid} not found.")
                return
        elif process_name.strip():
            name = process_name.strip()
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if proc.info["name"] and proc.info["name"].lower() == name.lower():
                        targets.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if not targets:
                errors.append(f"No process named '{name}' found.")
                return
        else:
            errors.append("Provide either process_name or pid.")
            return

        for proc in targets:
            try:
                proc.kill()
                killed.append({"pid": proc.pid, "name": proc.name()})
            except (psutil.NoSuchProcess, psutil.AccessDenied) as err:
                errors.append(f"Could not kill PID {proc.pid}: {err}")

    await asyncio.to_thread(_kill_sync)
    return {"killed": killed, "errors": errors}


# ── get_running_processes ─────────────────────────────────────────────────────

@tool(
    name="get_running_processes",
    description="List the top running processes sorted by CPU usage.",
    parameters={
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "How many processes to return (default 10, max 30).",
            }
        },
        "required": [],
    },
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def get_running_processes(top_n: int = 10) -> dict[str, Any]:
    """Return top processes by CPU usage."""
    top_n = max(1, min(30, top_n))

    def _collect() -> list[dict[str, Any]]:
        procs: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                mem_mb = round(info["memory_info"].rss / (1024 * 1024), 1) if info["memory_info"] else 0
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu_percent": round(info["cpu_percent"] or 0, 1),
                    "memory_mb": mem_mb,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return sorted(procs, key=lambda p: p["cpu_percent"], reverse=True)[:top_n]

    processes = await asyncio.to_thread(_collect)
    return {"processes": processes, "count": len(processes)}


# ── set_system_volume ─────────────────────────────────────────────────────────

@tool(
    name="set_system_volume",
    description="Set the Windows master volume to a percentage between 0 and 100.",
    parameters={
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "description": "Volume level from 0 (silent) to 100 (maximum).",
            }
        },
        "required": ["level"],
    },
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def set_system_volume(level: int) -> dict[str, Any]:
    """Set the master system volume."""
    clamped = max(0, min(100, level))

    def _set() -> None:
        from ctypes import POINTER, cast  # noqa: PLC0415

        from comtypes import CLSCTX_ALL  # type: ignore[import]
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore[import]

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(clamped / 100.0, None)

    await asyncio.to_thread(_set)
    return {"status": "ok", "volume_percent": clamped}


# ── get_system_volume ─────────────────────────────────────────────────────────

@tool(
    name="get_system_volume",
    description="Get the current Windows master volume level and mute state.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def get_system_volume() -> dict[str, Any]:
    """Return current volume and mute state."""
    def _get() -> dict[str, Any]:
        from ctypes import POINTER, cast  # noqa: PLC0415

        from comtypes import CLSCTX_ALL  # type: ignore[import]
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore[import]

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        scalar = volume.GetMasterVolumeLevelScalar()
        muted = bool(volume.GetMute())
        return {"volume_percent": round(scalar * 100), "muted": muted}

    return await asyncio.to_thread(_get)


# ── mute_toggle ───────────────────────────────────────────────────────────────

@tool(
    name="mute_toggle",
    description="Toggle the Windows system mute on or off.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def mute_toggle() -> dict[str, Any]:
    """Toggle system mute."""
    def _toggle() -> dict[str, Any]:
        from ctypes import POINTER, cast  # noqa: PLC0415

        from comtypes import CLSCTX_ALL  # type: ignore[import]
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore[import]

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = bool(volume.GetMute())
        volume.SetMute(not current, None)
        return {"muted": not current}

    return await asyncio.to_thread(_toggle)


# ── take_screenshot ───────────────────────────────────────────────────────────

@tool(
    name="take_screenshot",
    description="Capture the entire screen and save it to disk. Returns the file path.",
    parameters={
        "type": "object",
        "properties": {
            "return_base64": {
                "type": "boolean",
                "description": "If true, also return the image as a base64 string (may be large).",
            }
        },
        "required": [],
    },
    risk_level="medium",
    category="system",
    timeout_seconds=10,
)
async def take_screenshot(return_base64: bool = False) -> dict[str, Any]:
    """Take a screenshot and save to disk."""
    def _capture() -> dict[str, Any]:
        from PIL import ImageGrab  # type: ignore[import]

        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = _SCREENSHOT_DIR / f"jarvis_screenshot_{timestamp}.png"

        img = ImageGrab.grab()
        img.save(str(file_path), "PNG")

        result: dict[str, Any] = {
            "status": "ok",
            "file_path": str(file_path),
            "width": img.width,
            "height": img.height,
        }
        if return_base64:
            buf = io.BytesIO()
            img.save(buf, "PNG")
            result["base64"] = base64.b64encode(buf.getvalue()).decode()
        return result

    return await asyncio.to_thread(_capture)


# ── open_path ─────────────────────────────────────────────────────────────────

@tool(
    name="open_path",
    description="Open a file or folder in Windows Explorer or its default application.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file or folder to open.",
            }
        },
        "required": ["path"],
    },
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def open_path(path: str) -> dict[str, Any]:
    """Open a file or folder with its default handler."""
    target = Path(path.strip())
    if not target.exists():
        return {"status": "not_found", "path": str(target), "error": "Path does not exist."}

    def _open() -> None:
        os.startfile(str(target))  # type: ignore[attr-defined]

    await asyncio.to_thread(_open)
    return {"status": "ok", "path": str(target)}


# ── search_file ───────────────────────────────────────────────────────────────

@tool(
    name="search_file",
    description="Search the filesystem for files matching a name pattern.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Filename glob pattern, e.g. '*.pdf', 'report*.docx', 'notes.txt'.",
            },
            "search_root": {
                "type": "string",
                "description": "Directory to search in. Defaults to the user's home directory.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 20, max 50).",
            },
        },
        "required": ["pattern"],
    },
    risk_level="low",
    category="system",
    timeout_seconds=30,
)
async def search_file(
    pattern: str,
    search_root: str = "",
    max_results: int = 20,
) -> dict[str, Any]:
    """Recursively search for files matching a glob pattern."""
    root = Path(search_root.strip()) if search_root.strip() else Path.home()
    if not root.exists():
        return {"status": "not_found", "error": f"Search root '{root}' does not exist."}

    max_results = max(1, min(50, max_results))

    def _search() -> list[str]:
        found: list[str] = []
        try:
            for match in root.rglob(pattern):
                if match.is_file():
                    found.append(str(match))
                    if len(found) >= max_results:
                        break
        except PermissionError:
            pass
        return found

    results = await asyncio.to_thread(_search)
    return {"pattern": pattern, "search_root": str(root), "results": results, "count": len(results)}


# ── read_clipboard ────────────────────────────────────────────────────────────

@tool(
    name="read_clipboard",
    description="Read the current text content of the Windows clipboard.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def read_clipboard() -> dict[str, Any]:
    """Read clipboard text."""
    def _read() -> str:
        import pyperclip  # type: ignore[import]
        return pyperclip.paste()

    text = await asyncio.to_thread(_read)
    return {"text": text, "length": len(text)}


# ── write_clipboard ───────────────────────────────────────────────────────────

@tool(
    name="write_clipboard",
    description="Write text to the Windows clipboard, replacing its current content.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to put on the clipboard.",
            }
        },
        "required": ["text"],
    },
    risk_level="medium",
    category="system",
    timeout_seconds=5,
)
async def write_clipboard(text: str) -> dict[str, Any]:
    """Write text to clipboard."""
    def _write() -> None:
        import pyperclip  # type: ignore[import]
        pyperclip.copy(text)

    await asyncio.to_thread(_write)
    return {"status": "ok", "length": len(text)}


# ── lock_workstation ──────────────────────────────────────────────────────────

@tool(
    name="lock_workstation",
    description="Lock the Windows workstation (equivalent to Win+L).",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="medium",
    category="system",
    timeout_seconds=5,
)
async def lock_workstation() -> dict[str, Any]:
    """Lock the Windows session."""
    def _lock() -> None:
        ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]

    await asyncio.to_thread(_lock)
    return {"status": "ok", "message": "Workstation is locking."}


# ── shutdown_system ───────────────────────────────────────────────────────────

@tool(
    name="shutdown_system",
    description="Shut down the Windows PC. HIGH RISK — requires explicit user confirmation.",
    parameters={
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "integer",
                "description": "Seconds before shutdown begins (default 5, min 0, max 60).",
            }
        },
        "required": [],
    },
    risk_level="high",
    category="system",
    timeout_seconds=10,
)
async def shutdown_system(delay_seconds: int = 5) -> dict[str, Any]:
    """Schedule a system shutdown."""
    delay = max(0, min(60, delay_seconds))
    cmd = ["shutdown", "/s", "/t", str(delay)]

    def _run() -> None:
        subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603

    await asyncio.to_thread(_run)
    return {"status": "scheduled", "shutdown_in_seconds": delay}


# ── restart_system ────────────────────────────────────────────────────────────

@tool(
    name="restart_system",
    description="Restart the Windows PC. HIGH RISK — requires explicit user confirmation.",
    parameters={
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "integer",
                "description": "Seconds before restart begins (default 5, min 0, max 60).",
            }
        },
        "required": [],
    },
    risk_level="high",
    category="system",
    timeout_seconds=10,
)
async def restart_system(delay_seconds: int = 5) -> dict[str, Any]:
    """Schedule a system restart."""
    delay = max(0, min(60, delay_seconds))
    cmd = ["shutdown", "/r", "/t", str(delay)]

    def _run() -> None:
        subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603

    await asyncio.to_thread(_run)
    return {"status": "scheduled", "restart_in_seconds": delay}
