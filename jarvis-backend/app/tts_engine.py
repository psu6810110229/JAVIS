from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

import requests

KOKORO_URL = "http://kokoro:8880/v1/audio/speech"
KOKORO_MODEL = "kokoro"
KOKORO_VOICE = "bm_george"
KOKORO_OUTPUT_PATH = Path("/tmp/jarvis_assistant_audio.mp3")
KOKORO_TIMEOUT_SECONDS = 60
KOKORO_LEADING_PAUSE = ", ... "


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

    @property
    def voice_label(self) -> str:
        return self._voice_name

    async def synthesize(self, text: str) -> TtsSynthesisPayload:
        if not text.strip():
            raise TtsEngineError("Cannot synthesize empty text.")

        try:
            prepared_text = self._with_leading_pause(text)
            audio_bytes = await asyncio.to_thread(self._synthesize_sync, prepared_text)
        except Exception as error:  # noqa: BLE001
            raise TtsEngineError("Jarvis could not generate Kokoro speech output.") from error

        if not audio_bytes:
            raise TtsEngineError("Kokoro returned no audio data.")

        return TtsSynthesisPayload(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            mime_type="audio/mpeg",
            voice=self._voice_name,
        )

    def _synthesize_sync(self, text: str) -> bytes:
        response = requests.post(
            KOKORO_URL,
            json={
                "model": KOKORO_MODEL,
                "input": text,
                "voice": self._voice_name,
            },
            timeout=KOKORO_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        KOKORO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KOKORO_OUTPUT_PATH.write_bytes(response.content)
        return KOKORO_OUTPUT_PATH.read_bytes()

    @staticmethod
    def _with_leading_pause(text: str) -> str:
        return f"{KOKORO_LEADING_PAUSE}{text}"
