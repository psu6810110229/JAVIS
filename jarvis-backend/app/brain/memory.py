"""brain/memory.py — Session context management with strict num_ctx enforcement.

The Memory Shield:
- Hard cap of num_ctx tokens per session (default 1024).
- Uses a 4 chars/token heuristic for lightweight estimation (no tokenizer needed).
- When the session exceeds the budget, the oldest user+assistant message pair
  is pruned until the context fits. The system message is always preserved.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 4 chars per token is a safe conservative estimate for English/Thai mixed content.
_CHARS_PER_TOKEN_ESTIMATE: int = 4


def _estimate_tokens(text: str) -> int:
    """Lightweight token count estimate — no ML model required."""
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


class SessionMemory:
    """Thread-safe, per-session conversation memory with automatic context pruning.

    Parameters
    ----------
    max_tokens:
        Hard upper bound on total context tokens across all messages.
        Includes the system message. Default matches the strict Memory Shield (1024).
    """

    def __init__(self, max_tokens: int = 1024) -> None:
        self._max_tokens = max_tokens
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str, system_message: str) -> None:
        """Create a new session, idempotent (no-op if session already exists)."""
        async with self._global_lock:
            if session_id in self._sessions:
                return
            self._locks[session_id] = asyncio.Lock()
            self._sessions[session_id] = [
                {"role": "system", "content": system_message}
            ]
        logger.info("Created session '%s'.", session_id)

    async def close_session(self, session_id: str) -> None:
        """Remove session data and release its lock."""
        async with self._global_lock:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
        logger.info("Closed session '%s'.", session_id)

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session asyncio.Lock, creating one lazily if needed."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def get_messages(self, session_id: str) -> list[dict[str, str]] | None:
        """Return a *reference* to the message list (caller must hold session lock)."""
        return self._sessions.get(session_id)

    def append_user(self, session_id: str, text: str) -> None:
        """Append a user message (call while holding the session lock)."""
        messages = self._sessions.get(session_id)
        if messages is None:
            raise KeyError(f"Session '{session_id}' does not exist.")
        messages.append({"role": "user", "content": text})

    def append_assistant(self, session_id: str, text: str) -> None:
        """Append an assistant reply and immediately prune if over budget."""
        messages = self._sessions.get(session_id)
        if messages is None:
            raise KeyError(f"Session '{session_id}' does not exist.")
        messages.append({"role": "assistant", "content": text})
        self._prune(messages)

    def pop_last(self, session_id: str) -> dict[str, str] | None:
        """Remove and return the last message (used to roll back on error)."""
        messages = self._sessions.get(session_id)
        if messages and len(messages) > 1:  # never remove the system message
            return messages.pop()
        return None

    # ------------------------------------------------------------------
    # Context pruning — the Memory Shield
    # ------------------------------------------------------------------

    def _total_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(_estimate_tokens(m.get("content", "")) for m in messages)

    def _prune(self, messages: list[dict[str, str]]) -> None:
        """Remove the oldest non-system message pairs until under budget."""
        while self._total_tokens(messages) > self._max_tokens:
            # messages[0] is always the system prompt — skip it.
            # Remove messages[1] (oldest user or assistant turn).
            if len(messages) <= 1:
                break  # cannot prune further
            removed = messages.pop(1)
            logger.debug(
                "Context pruned: removed '%s' message (est. %d tokens). Budget: %d.",
                removed.get("role", "?"),
                _estimate_tokens(removed.get("content", "")),
                self._max_tokens,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def estimate_current_tokens(self, session_id: str) -> int:
        messages = self._sessions.get(session_id)
        if not messages:
            return 0
        return self._total_tokens(messages)

    def session_count(self) -> int:
        return len(self._sessions)

    def get_status(self) -> dict[str, Any]:
        return {
            "active_sessions": self.session_count(),
            "max_tokens_per_session": self._max_tokens,
        }
