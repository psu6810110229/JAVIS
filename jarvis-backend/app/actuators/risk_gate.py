"""actuators/risk_gate.py — Hard-enforcement risk confirmation layer.

Every tool call goes through RiskGate before execution:
    low    → execute immediately, no prompt
    medium → send ``confirmation_required`` WS event, wait up to 30 s
    high   → same, with a stronger warning message

The gate operates over an asyncio.Event per (session_id, tool_name) pair
so it does not block the entire backend — only the specific session
that triggered the risk-flagged tool call.

WebSocket protocol
------------------
Backend → Frontend:
    {
        "type": "confirmation_required",
        "payload": {
            "tool_name": str,
            "human_label": str,
            "risk_level": "medium" | "high",
            "message": str
        }
    }

Frontend → Backend  (handled in app/main.py):
    {"type": "tool.confirm", "payload": {}}
    {"type": "tool.deny",    "payload": {}}
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# How long (seconds) the gate waits for user confirmation before auto-denying.
_CONFIRMATION_TIMEOUT_SECONDS = 30.0

# Human-readable labels for tool names shown in the dialog.
_HUMAN_LABELS: dict[str, str] = {
    "open_application":  "Open application",
    "close_application": "Close application",
    "kill_process":      "Force-kill process",
    "take_screenshot":   "Take screenshot",
    "write_clipboard":   "Write to clipboard",
    "lock_workstation":  "Lock workstation",
    "shutdown_system":   "Shut down PC",
    "restart_system":    "Restart PC",
}

_RISK_MESSAGES: dict[str, str] = {
    "medium": (
        "Jarvis wants to perform the action above. "
        "Confirm to proceed or cancel to abort."
    ),
    "high": (
        "⚠️  This is a HIGH-RISK action that cannot easily be undone. "
        "Only confirm if you are absolutely sure."
    ),
}


class RiskGate:
    """Intercept tool calls and gate medium/high-risk ones on explicit user confirmation.

    One ``RiskGate`` instance should be shared across the entire Orchestrator lifecycle.
    Each pending confirmation is keyed by ``session_id`` so concurrent sessions do not
    interfere with each other.
    """

    def __init__(self) -> None:
        # session_id → asyncio.Event + result dict
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: str,
        session_id: str,
        send_event_fn: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> bool:
        """Return True if the tool should proceed, False if the user denied/timed out.

        ``send_event_fn`` must be an async callable that serialises a dict to the
        WebSocket connection for this session (wrapped by main.py).
        """
        if risk_level == "low":
            return True

        human_label = _HUMAN_LABELS.get(tool_name, tool_name.replace("_", " ").title())
        message = _RISK_MESSAGES.get(risk_level, _RISK_MESSAGES["medium"])

        # Register this session as pending before sending the event so
        # there is no race between the event arriving and the response.
        event = asyncio.Event()
        slot: dict[str, Any] = {"event": event, "approved": False}

        async with self._lock:
            if session_id in self._pending:
                # Already waiting for a confirmation on this session — deny the new one.
                logger.warning(
                    "RiskGate: session '%s' already has a pending confirmation. "
                    "Denying '%s'.",
                    session_id,
                    tool_name,
                )
                return False
            self._pending[session_id] = slot

        try:
            await send_event_fn(
                {
                    "type": "confirmation_required",
                    "payload": {
                        "tool_name": tool_name,
                        "human_label": human_label,
                        "risk_level": risk_level,
                        "arguments": arguments,
                        "message": message,
                        "timeout_seconds": _CONFIRMATION_TIMEOUT_SECONDS,
                    },
                }
            )

            try:
                await asyncio.wait_for(event.wait(), timeout=_CONFIRMATION_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    "RiskGate: confirmation for '%s' timed out after %.0fs — denying.",
                    tool_name,
                    _CONFIRMATION_TIMEOUT_SECONDS,
                )
                return False

            approved: bool = slot.get("approved", False)
            if not approved:
                logger.info("RiskGate: user denied '%s'.", tool_name)
            return approved
        finally:
            async with self._lock:
                self._pending.pop(session_id, None)

    def resolve(self, session_id: str, *, approved: bool) -> bool:
        """Call this when the frontend sends ``tool.confirm`` or ``tool.deny``.

        Returns True if there was a pending slot to resolve, False if there was none
        (e.g. it already timed out).
        """
        slot = self._pending.get(session_id)
        if slot is None:
            logger.warning(
                "RiskGate.resolve: no pending confirmation for session '%s'.", session_id
            )
            return False
        slot["approved"] = approved
        slot["event"].set()
        return True

    def has_pending(self, session_id: str) -> bool:
        """Return True if there is a pending confirmation for this session."""
        return session_id in self._pending
