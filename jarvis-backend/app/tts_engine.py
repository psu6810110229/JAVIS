from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_TTS_URL = "http://localhost:8870/v1/audio/speech"
DEFAULT_TTS_FALLBACK_URL = "http://localhost:8880/v1/audio/speech"
DEFAULT_TTS_MODE = "cpu-service"
DEFAULT_TTS_SPEED = 1.2
KOKORO_MODEL = "kokoro"
KOKORO_VOICE = "bm_george"
KOKORO_OUTPUT_PATH = Path("/tmp/jarvis_assistant_audio.mp3")
KOKORO_TIMEOUT_SECONDS = 60


class TtsEngineError(Exception):
    """Raised when Kokoro TTS synthesis fails."""


@dataclass(slots=True)
class TtsSynthesisPayload:
    audio_base64: str
    mime_type: str
    voice: str


class KokoroTtsEngine:
    def __init__(self) -> None:
        self._voice_name = KOKORO_VOICE
        self._mode = os.getenv("JARVIS_TTS_BACKEND_MODE", DEFAULT_TTS_MODE).strip() or DEFAULT_TTS_MODE
        self._primary_url = os.getenv("JARVIS_TTS_URL", DEFAULT_TTS_URL).strip() or DEFAULT_TTS_URL
        self._fallback_url = os.getenv("JARVIS_TTS_FALLBACK_URL", DEFAULT_TTS_FALLBACK_URL).strip() or DEFAULT_TTS_FALLBACK_URL
        self._active_url = self._primary_url
        self._last_status = "idle"
        self._last_warning: str | None = None
        self._last_provider = "unknown"
        self._last_vram_mb: float | None = None
        self._last_mime_type = "audio/mpeg"
        speed_raw = os.getenv("JARVIS_TTS_SPEED", str(DEFAULT_TTS_SPEED)).strip()
        try:
            self._speed = max(0.7, min(1.6, float(speed_raw)))
        except ValueError:
            self._speed = DEFAULT_TTS_SPEED

    @property
    def voice_label(self) -> str:
        return self._voice_name

    @property
    def mode(self) -> str:
        return self._mode

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

    async def synthesize(self, text: str) -> TtsSynthesisPayload:
        if not text.strip():
            raise TtsEngineError("Cannot synthesize empty text.")

        self._last_status = "running"
        self._last_warning = None

        try:
            audio_bytes = await asyncio.to_thread(self._synthesize_sync, text, self._primary_url)
            self._active_url = self._primary_url
        except Exception as error:  # noqa: BLE001
            if self._fallback_url != self._primary_url:
                self._last_warning = f"Primary TTS backend unavailable: {error}. Falling back to CPU service."
                try:
                    audio_bytes = await asyncio.to_thread(self._synthesize_sync, text, self._fallback_url)
                    self._active_url = self._fallback_url
                except Exception as fallback_error:  # noqa: BLE001
                    self._last_status = "error"
                    raise TtsEngineError("Jarvis could not generate Kokoro speech output.") from fallback_error
            else:
                self._last_status = "error"
                raise TtsEngineError("Jarvis could not generate Kokoro speech output.") from error

        if not audio_bytes:
            self._last_status = "error"
            raise TtsEngineError("Kokoro returned no audio data.")

        self._last_status = "ready"

        return TtsSynthesisPayload(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            mime_type=self._last_mime_type,
            voice=self._voice_name,
        )

    def _synthesize_sync(self, text: str, url: str) -> bytes:
        response = requests.post(
            url,
            json={
                "model": KOKORO_MODEL,
                "input": text,
                "voice": self._voice_name,
                "speed": self._speed,
            },
            timeout=KOKORO_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        provider_header = response.headers.get("X-TTS-Provider")
        if provider_header:
            self._last_provider = provider_header
        elif (
            "host.docker.internal:8870" in url
            or "localhost:8870" in url
            or "127.0.0.1:8870" in url
        ):
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

        self._last_mime_type = response.headers.get("Content-Type", "audio/mpeg")

        KOKORO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KOKORO_OUTPUT_PATH.write_bytes(response.content)
        return KOKORO_OUTPUT_PATH.read_bytes()
