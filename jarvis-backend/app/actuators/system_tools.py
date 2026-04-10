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

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.actuators.registry import tool

logger = logging.getLogger(__name__)


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
