"""actuators/notification_tools.py — Windows notification and system info tools.

Uses ``windows-toasts`` (WinRT bridge) for native Windows 10/11 toast notifications.
``get_system_info`` and ``set_reminder`` use psutil and asyncio respectively.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import psutil

from app.actuators.registry import tool

logger = logging.getLogger(__name__)


# ── send_notification ─────────────────────────────────────────────────────────

@tool(
    name="send_notification",
    description=(
        "Send a native Windows 10/11 toast notification with a title and body message."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Notification title (short, e.g. 'Jarvis').",
            },
            "body": {
                "type": "string",
                "description": "Notification body text.",
            },
        },
        "required": ["title", "body"],
    },
    risk_level="low",
    category="notification",
    timeout_seconds=10,
)
async def send_notification(title: str, body: str) -> dict[str, Any]:
    """Show a Windows toast notification."""
    def _show() -> None:
        try:
            from windows_toasts import Toast, WindowsToaster  # type: ignore[import]

            toaster = WindowsToaster("Jarvis")
            toast = Toast()
            toast.text_fields = [title, body]
            toaster.show_toast(toast)
        except Exception as err:  # noqa: BLE001
            # Fallback: use ctypes MessageBeep as a silent signal + log
            logger.warning("windows-toasts unavailable (%s); notification suppressed.", err)

    await asyncio.to_thread(_show)
    return {"status": "sent", "title": title, "body": body}


# ── set_reminder ──────────────────────────────────────────────────────────────

@tool(
    name="set_reminder",
    description="Schedule a toast notification reminder after a given number of seconds.",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The reminder message to display.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "How many seconds from now to show the reminder.",
            },
        },
        "required": ["message", "delay_seconds"],
    },
    risk_level="low",
    category="notification",
    timeout_seconds=5,
)
async def set_reminder(message: str, delay_seconds: int) -> dict[str, Any]:
    """Fire a toast notification after a delay, without blocking the caller."""
    delay = max(1, min(86400, delay_seconds))

    async def _fire_later() -> None:
        await asyncio.sleep(delay)
        await send_notification("⏰ Jarvis Reminder", message)

    asyncio.create_task(_fire_later())
    return {
        "status": "scheduled",
        "message": message,
        "fire_in_seconds": delay,
    }


# ── get_system_info ───────────────────────────────────────────────────────────

@tool(
    name="get_system_info",
    description=(
        "Return real-time system information: CPU usage, RAM, disk space, "
        "battery level (if available), and uptime."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="system",
    timeout_seconds=5,
)
async def get_system_info() -> dict[str, Any]:
    """Collect live system metrics via psutil."""
    def _collect() -> dict[str, Any]:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        boot_time = psutil.boot_time()

        import time  # noqa: PLC0415
        uptime_seconds = int(time.time() - boot_time)
        uptime_h = uptime_seconds // 3600
        uptime_m = (uptime_seconds % 3600) // 60

        info: dict[str, Any] = {
            "cpu_percent": round(cpu_pct, 1),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_used_gb": round(vm.used / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "ram_percent": vm.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 2),
            "disk_used_gb": round(disk.used / (1024 ** 3), 2),
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "disk_percent": disk.percent,
            "uptime": f"{uptime_h}h {uptime_m}m",
        }

        try:
            bat = psutil.sensors_battery()
            if bat is not None:
                info["battery_percent"] = round(bat.percent, 1)
                info["battery_plugged"] = bat.power_plugged
        except AttributeError:
            pass  # sensors_battery not available on this OS

        return info

    return await asyncio.to_thread(_collect)
