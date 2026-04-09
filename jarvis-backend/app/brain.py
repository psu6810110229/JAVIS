from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import speech_recognition as sr
from pydantic import BaseModel, Field, ValidationError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .config import load_project_env
from .tts_engine import ElevenLabsTtsEngine, TtsEngineError

logger = logging.getLogger(__name__)
DEFAULT_OLLAMA_MODEL_NAME = "scb10x/typhoon2.5-qwen3-4b"
DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 60.0
DEFAULT_STT_LANGUAGE = "th-TH"


class AudioProcessingError(Exception):
    """Raised when voice input or output processing fails."""


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AssistantAudioPayload(BaseModel):
    audio_base64: str
    mime_type: str
    voice: str


class BrainResponse(BaseModel):
    text: str
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    assistant_audio: AssistantAudioPayload | None = None
    audio_error: str | None = None


class AudioChunkAcknowledgement(BaseModel):
    accepted: bool
    chunk_size: int
    mime_type: str
    detail: str


class AudioChunkPayload(BaseModel):
    mime_type: str = Field(default="audio/webm")
    data: str
    is_final: bool = Field(default=True)


class SpeechTranscriptPayload(BaseModel):
    text: str
    source: str = Field(default="stt")


class VoiceInteractionResult(BaseModel):
    acknowledgement: AudioChunkAcknowledgement
    transcript: SpeechTranscriptPayload
    response: BrainResponse


class JarvisBrain:
    def __init__(self, model_name: str | None = None) -> None:
        load_project_env()
        self._model_name = DEFAULT_OLLAMA_MODEL_NAME
        self._ollama_base_url = DEFAULT_OLLAMA_BASE_URL
        self._ollama_timeout = DEFAULT_OLLAMA_TIMEOUT_SECONDS
        self._tts_engine = ElevenLabsTtsEngine()
        self._tts_voice = self._tts_engine.voice_label
        self._stt_language = DEFAULT_STT_LANGUAGE
        self._system_instruction = (
            "คุณคือ Jarvis (จาร์วิส) เอไอผู้ช่วยส่วนตัวที่ชาญฉลาดและซื่อสัตย์ "
            "คุณต้องเรียกผู้ใช้ว่า 'Sir' (เซอร์) หรือ 'คุณ Fran' เสมอ (ห้ามเรียกว่า สิริ) "
            "การตอบต้องกระชับ มีไหวพริบ และมีความเป็นมืออาชีพ "
            "ต้องลงท้ายประโยคด้วยคำว่า 'ครับ' ทุกครั้งเพื่อให้ดูสุภาพ "
            "สื่อสารด้วยภาษาไทยที่เข้าใจง่ายและเป็นธรรมชาติ"
        )
        self._sessions: dict[str, list[dict[str, str]] | None] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._ollama_available = False
        self._initialization_error: str | None = None
        self._tool_schemas = self._build_tool_schemas()

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return list(self._tool_schemas)

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                self._ollama_available, self._initialization_error = await self._check_ollama_health()
            except (RuntimeError, TypeError, ValueError) as error:
                self._ollama_available = False
                self._initialization_error = str(error)
                logger.exception("JarvisBrain initialization failed: %s", error)

            self._initialized = True
            if self._ollama_available:
                logger.info(
                    "JarvisBrain initialized with Ollama model '%s' at '%s'.",
                    self._model_name,
                    self._ollama_base_url,
                )
            else:
                logger.warning(
                    "JarvisBrain running in degraded mode. Ollama unavailable at '%s': %s",
                    self._ollama_base_url,
                    self._initialization_error or "unknown reason",
                )

    async def create_session(self, session_id: str) -> None:
        await self.initialize()
        if session_id in self._sessions:
            return

        self._session_locks[session_id] = asyncio.Lock()

        self._sessions[session_id] = [
            {
                "role": "system",
                "content": self._system_instruction,
            }
        ]
        logger.info("Created Jarvis session '%s'.", session_id)

    async def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        logger.info("Closed Jarvis session '%s'.", session_id)

    async def handle_text(self, session_id: str, user_text: str) -> BrainResponse:
        if not user_text.strip():
            return BrainResponse(
                text="I did not receive any text to process.",
                tool_schemas=self.tool_schemas,
            )

        await self.create_session(session_id)
        messages = self._sessions.get(session_id)

        if messages is None:
            return BrainResponse(
                text="Jarvis session state is unavailable. Please restart the session.",
                tool_schemas=self.tool_schemas,
            )

        session_lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with session_lock:
            messages.append({"role": "user", "content": user_text})
            try:
                response_text = await self._generate_ollama_response(messages)
                messages.append({"role": "assistant", "content": response_text})
                self._ollama_available = True
                self._initialization_error = None
            except OllamaUnavailableError as error:
                logger.exception("Ollama unavailable for session '%s'.", session_id)
                self._ollama_available = False
                self._initialization_error = str(error)
                messages.pop()
                return BrainResponse(
                    text=f"Jarvis could not reach local Ollama at {self._ollama_base_url}: {error}",
                    tool_schemas=self.tool_schemas,
                )
            except OllamaModelError as error:
                logger.exception("Ollama model error for session '%s'.", session_id)
                messages.pop()
                return BrainResponse(
                    text=(
                        f"Jarvis local model '{self._model_name}' is not ready. "
                        f"Run 'ollama pull {self._model_name}' and retry. "
                        f"Details: {error}"
                    ),
                    tool_schemas=self.tool_schemas,
                )
            except OllamaResponseError as error:
                logger.exception("Ollama response parse failed for session '%s'.", session_id)
                messages.pop()
                return BrainResponse(
                    text=f"Jarvis received an invalid response from Ollama: {error}",
                    tool_schemas=self.tool_schemas,
                )
            except RuntimeError as error:
                logger.exception("Chat session failed for session '%s'.", session_id)
                messages.pop()
                return BrainResponse(
                    text=f"Jarvis encountered a runtime error while processing your message: {error}",
                    tool_schemas=self.tool_schemas,
                )

        response_text = response_text or "Jarvis completed the request without a text response."

        assistant_audio: AssistantAudioPayload | None = None
        audio_error: str | None = None
        try:
            assistant_audio = await self._synthesize_assistant_audio(response_text)
        except AudioProcessingError as error:
            logger.exception("Thai TTS generation failed for session '%s': %s", session_id, error)
            audio_error = str(error)

        return BrainResponse(
            text=response_text,
            tool_schemas=self.tool_schemas,
            assistant_audio=assistant_audio,
            audio_error=audio_error,
        )

    async def _check_ollama_health(self) -> tuple[bool, str | None]:
        try:
            timeout = httpx.Timeout(self._ollama_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                response = await client.get("/api/tags")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            return False, f"unable to connect ({error})"
        except httpx.HTTPError as error:
            return False, f"HTTP error ({error})"

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"

        try:
            payload = response.json()
        except ValueError as error:
            return False, f"invalid health payload ({error})"

        models = payload.get("models")
        if not isinstance(models, list):
            return False, "missing model list in /api/tags response"

        model_name_matches = any(
            isinstance(model, dict)
            and isinstance(model.get("name"), str)
            and model.get("name", "").split(":", 1)[0] == self._model_name
            for model in models
        )
        if not model_name_matches and models:
            logger.warning(
                "Configured model '%s' not currently listed in Ollama tags. It may need to be pulled.",
                self._model_name,
            )

        return True, None

    async def _generate_ollama_response(self, messages: list[dict[str, str]]) -> str:
        request_payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
        }

        try:
            timeout = httpx.Timeout(self._ollama_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                response = await client.post("/api/chat", json=request_payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            raise OllamaUnavailableError(str(error)) from error
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(str(error)) from error

        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            if response.status_code in (400, 404):
                lower_detail = detail.lower()
                if "model" in lower_detail and ("not found" in lower_detail or "missing" in lower_detail):
                    raise OllamaModelError(detail)
            raise OllamaUnavailableError(detail)

        try:
            payload = response.json()
        except ValueError as error:
            raise OllamaResponseError(f"invalid JSON payload ({error})") from error

        payload_error = payload.get("error")
        if isinstance(payload_error, str) and payload_error.strip():
            lower_error = payload_error.lower()
            if "model" in lower_error and ("not found" in lower_error or "missing" in lower_error):
                raise OllamaModelError(payload_error)
            raise OllamaUnavailableError(payload_error)

        message_payload = payload.get("message")
        if not isinstance(message_payload, dict):
            raise OllamaResponseError("missing message object")

        content = message_payload.get("content")
        if not isinstance(content, str):
            raise OllamaResponseError("missing assistant message content")

        cleaned_content = content.strip()
        if not cleaned_content:
            raise OllamaResponseError("assistant message content is empty")

        return cleaned_content

    async def handle_audio_chunk(
        self,
        session_id: str,
        payload: AudioChunkPayload,
    ) -> VoiceInteractionResult:
        decoded_audio = await self._decode_audio_payload(payload)
        wav_audio = await asyncio.to_thread(self._transcode_webm_to_wav, decoded_audio)
        transcript_text = await asyncio.to_thread(self._transcribe_wav_audio, wav_audio)
        response = await self.handle_text(session_id=session_id, user_text=transcript_text)

        acknowledgement = AudioChunkAcknowledgement(
            accepted=True,
            chunk_size=len(decoded_audio),
            mime_type=payload.mime_type,
            detail="Audio chunk accepted and processed for transcription.",
        )
        transcript = SpeechTranscriptPayload(text=transcript_text)

        return VoiceInteractionResult(
            acknowledgement=acknowledgement,
            transcript=transcript,
            response=response,
        )

    async def _decode_audio_payload(self, payload: AudioChunkPayload) -> bytes:
        if not payload.is_final:
            raise AudioProcessingError("Streaming partial audio is not supported in Phase 2.")

        if not payload.mime_type.startswith("audio/webm"):
            raise AudioProcessingError(f"Unsupported audio mime type '{payload.mime_type}'.")

        try:
            return await asyncio.to_thread(base64.b64decode, payload.data, validate=True)
        except (ValueError, binascii.Error) as error:
            logger.warning("Received invalid base64 audio payload: %s", error)
            raise AudioProcessingError("Audio chunk was rejected because the payload is not valid base64.") from error

    @staticmethod
    def _transcode_webm_to_wav(audio_bytes: bytes) -> bytes:
        source_buffer = io.BytesIO(audio_bytes)
        target_buffer = io.BytesIO()

        try:
            audio_segment = AudioSegment.from_file(source_buffer, format="webm")
            audio_segment.export(target_buffer, format="wav")
        except (CouldntDecodeError, OSError, ValueError) as error:
            raise AudioProcessingError("Jarvis could not convert the recorded WebM audio into WAV.") from error

        return target_buffer.getvalue()

    def _transcribe_wav_audio(self, wav_audio: bytes) -> str:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(io.BytesIO(wav_audio)) as source:
                audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language=self._stt_language)
        except sr.UnknownValueError as error:
            raise AudioProcessingError("Jarvis could not understand the recorded Thai speech.") from error
        except sr.RequestError as error:
            raise AudioProcessingError(f"Jarvis STT request failed: {error}") from error
        except (OSError, ValueError) as error:
            raise AudioProcessingError("Jarvis could not read the transcoded WAV audio.") from error

        cleaned_transcript = transcript.strip()
        if not cleaned_transcript:
            raise AudioProcessingError("Jarvis did not detect any spoken text in the recording.")

        return cleaned_transcript

    async def _synthesize_assistant_audio(self, text: str) -> AssistantAudioPayload:
        try:
            synthesized = await self._tts_engine.synthesize(text)
        except TtsEngineError as error:
            raise AudioProcessingError(str(error)) from error

        return AssistantAudioPayload(
            audio_base64=synthesized.audio_base64,
            mime_type=synthesized.mime_type,
            voice=synthesized.voice,
        )

    @staticmethod
    def _build_tool_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                name="get_current_datetime",
                description="Return the current date and time for a requested IANA timezone.",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone_name": {
                            "type": "string",
                            "description": "An IANA timezone such as UTC or Asia/Bangkok.",
                        }
                    },
                    "required": [],
                },
            ),
            ToolSchema(
                name="get_backend_status",
                description="Return a summary of the Jarvis backend runtime status.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    @staticmethod
    def _get_current_datetime(timezone_name: str = "UTC") -> dict[str, str]:
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

    def _get_backend_status(self) -> dict[str, str]:
        return {
            "service": "jarvis-backend",
            "model_name": self._model_name,
            "ollama_base_url": self._ollama_base_url,
            "ollama_configured": str(self._ollama_available).lower(),
            "tts_voice": self._tts_voice,
            "stt_language": self._stt_language,
            "status": "ready" if self._ollama_available else "degraded",
        }

    @staticmethod
    def parse_audio_payload(raw_payload: dict[str, Any]) -> AudioChunkPayload:
        try:
            return AudioChunkPayload.model_validate(raw_payload)
        except ValidationError as error:
            raise ValueError("Invalid audio payload.") from error


class OllamaUnavailableError(Exception):
    """Raised when the Ollama runtime cannot be reached or returns service errors."""


class OllamaModelError(Exception):
    """Raised when the configured model is missing from Ollama."""


class OllamaResponseError(Exception):
    """Raised when Ollama response payload does not contain expected fields."""
