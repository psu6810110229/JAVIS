"""brain/streamer.py — Async LLM-to-TTS pipeline with partial-chunk dispatch.

The key design goal: as soon as the LLM emits enough tokens to form a
sentence boundary, that sentence is dispatched to the TTS worker as a
background asyncio.Task — *without* blocking the main token stream.

This allows audio synthesis to begin while the LLM is still generating
the next sentence, cutting effective voice latency significantly.

Token chunk dispatch
--------------------
On every LLM delta we:
1. Accumulate tokens into a sentence buffer.
2. Detect sentence boundaries with a regex ( . ! ? \\n ).
3. For each completed sentence, spawn a TTS task (bounded by semaphore).
4. In parallel, flush accumulated text to the UI via on_text_chunk at
   100 ms intervals or every 64 chars (whichever comes first).

Audio-synthesis timing
----------------------
Each TTS task logs Audio-Synthesis-Time at INFO level.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Sentence-boundary detection (greedy — ends at punctuation or newline).
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")


class AsyncChunkStreamer:
    """Consumes a raw LLM token stream and dispatches to TTS in parallel.

    Parameters
    ----------
    tts_synthesize:
        Coroutine factory: ``async (text: str) -> Any`` — should return an
        ``AssistantAudioPayload``-like object or raise ``AudioProcessingError``.
        Injected by the Orchestrator to avoid circular imports.
    on_text_chunk:
        Coroutine called with accumulated text for UI streaming.
    on_sentence_audio:
        Coroutine called with ``(sentence_index, sentence_text, audio_payload, error_str|None)``.
    max_tts_concurrency:
        Maximum simultaneous TTS synthesis calls. Default 2 to avoid OOM.
    flush_interval_seconds:
        Maximum time between on_text_chunk flushes.
    flush_min_chars:
        Minimum accumulated chars before an interim flush is emitted.
    """

    def __init__(
        self,
        *,
        tts_synthesize: Callable[[str], Awaitable[Any]] | None,
        on_text_chunk: Callable[[str], Awaitable[None]],
        on_sentence_audio: Callable[[int, str, Any | None, str | None], Awaitable[None]] | None,
        speech_transform: Callable[[str], str] | None = None,
        max_tts_concurrency: int = 2,
        flush_interval_seconds: float = 0.10,
        flush_min_chars: int = 64,
    ) -> None:
        self._tts_synthesize = tts_synthesize
        self._on_text_chunk = on_text_chunk
        self._on_sentence_audio = on_sentence_audio
        self._speech_transform = speech_transform
        self._flush_interval = flush_interval_seconds
        self._flush_min_chars = flush_min_chars

        self._semaphore = asyncio.Semaphore(max_tts_concurrency)
        self._sentence_buffer: str = ""
        self._text_emit_buffer: str = ""
        self._last_flush_monotonic: float = 0.0
        self._sentence_index: int = 0
        self._tts_tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    async def on_delta(self, delta: str) -> None:
        """Called for every LLM token chunk. Thread-safe within asyncio."""
        if not delta:
            return

        # ── Text emit path (UI streaming) ──────────────────────────────
        self._text_emit_buffer += delta
        await self._maybe_flush_text(force=False)

        # ── TTS sentence dispatch path ─────────────────────────────────
        if self._tts_synthesize is None or self._on_sentence_audio is None:
            return

        self._sentence_buffer += delta
        completed_sentences, self._sentence_buffer = _extract_sentences(self._sentence_buffer)
        for sentence_text in completed_sentences:
            self._dispatch_tts(sentence_text)

    async def flush_final(self, trailing_buffer: str = "") -> None:
        """Force-flush any remaining text and dispatch trailing TTS sentence."""
        # Flush remaining text to UI
        if self._text_emit_buffer:
            await self._maybe_flush_text(force=True)

        # Handle trailing partial sentence (e.g., final words without punctuation)
        if trailing_buffer.strip() and self._tts_synthesize is not None and self._on_sentence_audio is not None:
            self._dispatch_tts(trailing_buffer.strip())

        # Wait for all TTS tasks to complete
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)

    async def cancel_all(self) -> None:
        """Cancel any in-flight TTS tasks (called on error path)."""
        for task in self._tts_tasks:
            if not task.done():
                task.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)

    def get_trailing_sentence_buffer(self) -> str:
        """Return accumulated text after last sentence boundary."""
        return self._sentence_buffer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _maybe_flush_text(self, *, force: bool) -> None:
        if not self._text_emit_buffer:
            return
        now = time.monotonic()
        should_flush = (
            force
            or len(self._text_emit_buffer) >= self._flush_min_chars
            or (now - self._last_flush_monotonic) >= self._flush_interval
        )
        if not should_flush:
            return
        chunk = self._text_emit_buffer
        self._text_emit_buffer = ""
        self._last_flush_monotonic = now
        await self._on_text_chunk(chunk)

    def _dispatch_tts(self, sentence_text: str) -> None:
        """Spawn a non-blocking TTS synthesis task for *sentence_text*."""
        if self._speech_transform is not None:
            sentence_text = self._speech_transform(sentence_text)
        if not sentence_text.strip():
            return

        idx = self._sentence_index
        self._sentence_index += 1
        task = asyncio.create_task(
            self._synthesize_sentence(idx, sentence_text),
            name=f"tts-sentence-{idx}",
        )
        self._tts_tasks.append(task)

    async def _synthesize_sentence(self, sentence_index: int, sentence_text: str) -> None:
        """Run TTS synthesis for one sentence; calls on_sentence_audio with result."""
        assert self._tts_synthesize is not None
        assert self._on_sentence_audio is not None

        t_start = time.monotonic()
        try:
            async with self._semaphore:
                audio_payload = await self._tts_synthesize(sentence_text)
            synthesis_time = time.monotonic() - t_start
            logger.info(
                "Audio-Synthesis-Time=%.3fs sentence_index=%d chars=%d",
                synthesis_time,
                sentence_index,
                len(sentence_text),
            )
            await self._on_sentence_audio(sentence_index, sentence_text, audio_payload, None)
        except Exception as err:  # noqa: BLE001 — must not crash the stream
            logger.exception(
                "TTS synthesis failed for sentence %d: %s", sentence_index, err
            )
            await self._on_sentence_audio(sentence_index, sentence_text, None, str(err))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Split *buffer* at sentence boundaries, returning (completed, remainder)."""
    sentences: list[str] = []
    cursor = 0

    while True:
        match = _SENTENCE_BOUNDARY_RE.search(buffer, cursor)
        if match is None:
            break
        boundary_end = match.end()
        sentence = buffer[:boundary_end].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[boundary_end:]
        cursor = 0

    return sentences, buffer
