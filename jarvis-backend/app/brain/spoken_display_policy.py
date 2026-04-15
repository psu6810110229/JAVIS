"""Policy utilities for separating spoken and displayed assistant text."""
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s]+")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_JSON_ONLY_RE = re.compile(r"^\s*[\[{][\s\S]*[\]}]\s*$")
_WHITESPACE_RE = re.compile(r"\s+")


class SpokenDisplayPolicy:
    """Centralized text policy for UI text vs TTS text."""

    def __init__(self, *, max_spoken_chars: int = 320) -> None:
        self._max_spoken_chars = max(80, int(max_spoken_chars))

    @staticmethod
    def to_display_text(text: str) -> str:
        cleaned = (text or "").replace("\x00", "").strip()
        return cleaned or "I couldn't process that request."

    def to_spoken_with_meta(self, text: str) -> tuple[str, list[str]]:
        markers: list[str] = []

        raw = self.to_display_text(text)
        if _JSON_ONLY_RE.match(raw):
            return "I am processing that command.", ["json_only_rewrite"]

        spoken = _CODE_FENCE_RE.sub(" ", raw)
        if spoken != raw:
            markers.append("removed_code_fence")

        spoken_before = spoken
        spoken = _MARKDOWN_LINK_RE.sub(r"\1", spoken)
        if spoken != spoken_before:
            markers.append("stripped_markdown_link_url")

        spoken_before = spoken
        spoken = _INLINE_CODE_RE.sub(r"\1", spoken)
        if spoken != spoken_before:
            markers.append("stripped_inline_code_ticks")

        spoken_before = spoken
        spoken = _URL_RE.sub(" link omitted ", spoken)
        if spoken != spoken_before:
            markers.append("redacted_url")

        spoken_before = spoken
        spoken = _PATH_RE.sub(" file path omitted ", spoken)
        if spoken != spoken_before:
            markers.append("redacted_windows_path")

        # Keep TTS concise and natural by flattening excessive whitespace.
        spoken_before = spoken
        spoken = _WHITESPACE_RE.sub(" ", spoken).strip()
        if spoken != spoken_before:
            markers.append("normalized_whitespace")
        if not spoken:
            return "Done.", markers + ["empty_after_transform_fallback"]

        if len(spoken) <= self._max_spoken_chars:
            return spoken, markers

        clipped = spoken[: self._max_spoken_chars].rstrip(" ,;:")
        last_stop = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if last_stop >= 60:
            clipped = clipped[: last_stop + 1]
        return (clipped + "...").strip(), markers + ["clipped_max_length"]

    def to_spoken_text(self, text: str) -> str:
        spoken, _markers = self.to_spoken_with_meta(text)
        return spoken
