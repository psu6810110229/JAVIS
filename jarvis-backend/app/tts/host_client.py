"""tts/host_client.py — Async HTTP client for the Windows Kokoro TTS worker.

Replaces the old tts_engine.py:
- Uses httpx.AsyncClient natively (no asyncio.to_thread wrapper)
- Drops the unnecessary local file I/O (KOKORO_OUTPUT_PATH)
- Retains primary/fallback URL strategy and status telemetry
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TTS_URL = "http://kokoro:8880/v1/audio/speech"
DEFAULT_TTS_FALLBACK_URL = "http://kokoro:8880/v1/audio/speech"
DEFAULT_TTS_MODE = "cpu-service"
DEFAULT_TTS_SPEED = 1.2
DEFAULT_TTS_TIMEOUT_SECONDS = 60.0
KOKORO_MODEL = "kokoro"
KOKORO_VOICE = "bm_george"


class TtsEngineError(Exception):
    """Raised when Kokoro TTS synthesis fails (primary and fallback)."""


@dataclass(slots=True)
class TtsSynthesisPayload:
    """Result returned from a successful TTS synthesis call."""

    audio_base64: str
    mime_type: str
    voice: str


class TtsHostClient:
    """Async HTTP client that communicates with the Windows-side Kokoro worker.

    The worker may run as:
        - DML host worker  (host.docker.internal:8870) — iGPU / CPU via ONNX
        - CPU service      (kokoro container:8880)      — CPU-only fallback

    Primary is tried first; on any error the fallback is attempted.
    """

    def __init__(self) -> None:
        self._voice_name: str = KOKORO_VOICE
        self._mode: str = os.getenv("JARVIS_TTS_BACKEND_MODE", DEFAULT_TTS_MODE).strip() or DEFAULT_TTS_MODE
        self._primary_url: str = os.getenv("JARVIS_TTS_URL", DEFAULT_TTS_URL).strip() or DEFAULT_TTS_URL
        self._fallback_url: str = (
            os.getenv("JARVIS_TTS_FALLBACK_URL", DEFAULT_TTS_FALLBACK_URL).strip()
            or DEFAULT_TTS_FALLBACK_URL
        )
        self._active_url: str = self._primary_url
        self._last_status: str = "idle"
        self._last_warning: str | None = None
        self._last_provider: str = "unknown"
        self._last_vram_mb: float | None = None
        self._last_mime_type: str = "audio/mpeg"

        speed_raw = os.getenv("JARVIS_TTS_SPEED", str(DEFAULT_TTS_SPEED)).strip()
        try:
            self._speed: float = max(0.7, min(1.6, float(speed_raw)))
        except ValueError:
            self._speed = DEFAULT_TTS_SPEED

        timeout_raw = os.getenv("JARVIS_TTS_TIMEOUT_SECONDS", str(DEFAULT_TTS_TIMEOUT_SECONDS))
        try:
            self._timeout: float = max(5.0, float(timeout_raw))
        except ValueError:
            self._timeout = DEFAULT_TTS_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def voice_label(self) -> str:
        return self._voice_name

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> TtsSynthesisPayload:
        """Synthesize *text* to audio. Tries primary, then falls back on error.

        Returns
        -------
        TtsSynthesisPayload
            Base64-encoded audio with MIME type and voice label.

        Raises
        ------
        TtsEngineError
            If both primary and fallback endpoints fail.
        """
        if not text.strip():
            raise TtsEngineError("Cannot synthesize empty text.")

        self._last_status = "running"
        self._last_warning = None

        try:
            audio_bytes, mime_type = await self._post_tts(text, self._primary_url)
            self._active_url = self._primary_url
        except (httpx.HTTPError, httpx.TimeoutException, TtsEngineError) as primary_err:
            if self._fallback_url == self._primary_url:
                self._last_status = "error"
                raise TtsEngineError("Jarvis could not generate Kokoro speech output.") from primary_err

            self._last_warning = (
                f"Primary TTS backend unavailable: {primary_err!s}. "
                "Falling back to CPU service."
            )
            logger.warning(self._last_warning)
            try:
                audio_bytes, mime_type = await self._post_tts(text, self._fallback_url)
                self._active_url = self._fallback_url
            except (httpx.HTTPError, httpx.TimeoutException, TtsEngineError) as fallback_err:
                self._last_status = "error"
                raise TtsEngineError("Jarvis could not generate Kokoro speech output.") from fallback_err

        if not audio_bytes:
            self._last_status = "error"
            raise TtsEngineError("Kokoro returned no audio data.")

        self._last_status = "ready"
        self._last_mime_type = mime_type

        return TtsSynthesisPayload(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            mime_type=mime_type,
            voice=self._voice_name,
        )

    async def get_runtime_status(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "primary_url": self._primary_url,
            "fallback_url": self._fallback_url,
            "active_url": self._active_url,
            "status": self._last_status,
            "provider": self._last_provider,
            "vram_mb": self._last_vram_mb,
            "speed": self._speed,
            "warning": self._last_warning,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post_tts(self, text: str, url: str) -> tuple[bytes, str]:
        """POST a TTS request to *url* and return (audio_bytes, mime_type)."""
        request_body = {
            "model": KOKORO_MODEL,
            "input": text,
            "voice": self._voice_name,
            "speed": self._speed,
        }
        timeout = httpx.Timeout(self._timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=request_body)

        if response.status_code >= 400:
            raise TtsEngineError(
                f"TTS endpoint returned HTTP {response.status_code}: {response.text[:200]}"
            )

        # Parse optional telemetry headers from the DML worker
        provider_header = response.headers.get("X-TTS-Provider")
        if provider_header:
            self._last_provider = provider_header
        elif "host.docker.internal" in url:
            self._last_provider = "dml-host"
        else:
            self._last_provider = "cpu-service"

        vram_header = response.headers.get("X-TTS-VRAM-MB")
        if vram_header:
            try:
                self._last_vram_mb = float(vram_header)
            except ValueError:
                self._last_vram_mb = None
        else:
            self._last_vram_mb = None

        mime_type = response.headers.get("Content-Type", "audio/mpeg")
        return response.content, mime_type
