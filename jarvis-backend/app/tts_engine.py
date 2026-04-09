from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Iterable

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from .config import load_project_env

DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_VOICE = "Adam"
DEFAULT_STABILITY = 0.5
DEFAULT_CLARITY = 0.75

KNOWN_PREMIUM_MALE_VOICES: dict[str, str] = {
    "adam": "pNInz6obpgDQGcFmaJgB",
    "antoni": "ErXwobaYiN019PkySvjV",
}


class TtsEngineError(Exception):
    """Raised when ElevenLabs TTS synthesis fails."""


@dataclass(slots=True)
class TtsSynthesisPayload:
    audio_base64: str
    mime_type: str
    voice: str


class ElevenLabsTtsEngine:
    def __init__(self) -> None:
        load_project_env()
        self._api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        self._model_id = DEFAULT_ELEVENLABS_MODEL
        self._voice_name = os.getenv("ELEVENLABS_VOICE", DEFAULT_ELEVENLABS_VOICE).strip() or DEFAULT_ELEVENLABS_VOICE
        self._voice_id = self._resolve_voice_id(self._voice_name)

    @property
    def voice_label(self) -> str:
        return self._voice_name

    async def synthesize(self, text: str) -> TtsSynthesisPayload:
        if not text.strip():
            raise TtsEngineError("Cannot synthesize empty text.")

        if not self._api_key:
            raise TtsEngineError("ELEVENLABS_API_KEY is not configured.")

        try:
            audio_bytes = await asyncio.to_thread(self._synthesize_sync, text)
        except Exception as error:  # noqa: BLE001
            raise TtsEngineError("Jarvis could not generate ElevenLabs speech output.") from error

        if not audio_bytes:
            raise TtsEngineError("ElevenLabs returned no audio data.")

        return TtsSynthesisPayload(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            mime_type="audio/mpeg",
            voice=self._voice_name,
        )

    def _synthesize_sync(self, text: str) -> bytes:
        client = ElevenLabs(api_key=self._api_key)
        chunks: Iterable[bytes] = client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=DEFAULT_STABILITY,
                similarity_boost=DEFAULT_CLARITY,
            ),
        )

        audio_buffer = bytearray()
        for chunk in chunks:
            if isinstance(chunk, (bytes, bytearray)):
                audio_buffer.extend(chunk)

        return bytes(audio_buffer)

    @staticmethod
    def _resolve_voice_id(voice_name_or_id: str) -> str:
        normalized = voice_name_or_id.strip().lower()
        if normalized in KNOWN_PREMIUM_MALE_VOICES:
            return KNOWN_PREMIUM_MALE_VOICES[normalized]

        return voice_name_or_id
