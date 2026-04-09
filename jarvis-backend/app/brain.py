from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import edge_tts
import google.generativeai as genai
import speech_recognition as sr
from aiohttp import ClientError
from google.api_core.exceptions import GoogleAPIError
from pydantic import BaseModel, Field, ValidationError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .config import load_project_env

logger = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL_NAME = "gemini-1.5-flash"
DEFAULT_THAI_TTS_VOICE = "th-TH-PremwadeeNeural"
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
        self._model_name = model_name or os.getenv("JARVIS_MODEL_NAME", DEFAULT_GEMINI_MODEL_NAME)
        self._tts_voice = DEFAULT_THAI_TTS_VOICE
        self._stt_language = DEFAULT_STT_LANGUAGE
        self._system_instruction = (
            "You are Jarvis, a modern voice-first AI assistant. "
            "Respond with concise, helpful, production-minded answers. "
            "Use tools when they are clearly needed, and keep the conversation grounded in the user's request."
        )
        self._model: genai.GenerativeModel | None = None
        self._sessions: dict[str, Any] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._tool_schemas = self._build_tool_schemas()
        self._tools = [self._get_current_datetime, self._get_backend_status]

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return list(self._tool_schemas)

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("Gemini API key is not configured; JarvisBrain will run in degraded mode.")
                self._initialized = True
                return

            try:
                await asyncio.to_thread(genai.configure, api_key=api_key)
                self._model = await asyncio.to_thread(
                    self._create_model,
                    self._model_name,
                    self._system_instruction,
                    self._tools,
                )
            except (GoogleAPIError, RuntimeError, TypeError, ValueError) as error:
                logger.exception("JarvisBrain initialization failed: %s", error)
                self._model = None

            self._initialized = True
            if self._model is not None:
                logger.info("JarvisBrain initialized with model '%s'.", self._model_name)

    async def create_session(self, session_id: str) -> None:
        await self.initialize()
        if session_id in self._sessions:
            return

        self._session_locks[session_id] = asyncio.Lock()

        if self._model is None:
            self._sessions[session_id] = None
            return

        try:
            chat_session = await asyncio.to_thread(
                self._model.start_chat,
                history=[],
                enable_automatic_function_calling=True,
            )
        except (GoogleAPIError, RuntimeError, TypeError, ValueError) as error:
            logger.exception("Could not create Jarvis session '%s': %s", session_id, error)
            self._sessions[session_id] = None
            return

        self._sessions[session_id] = chat_session
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
        chat_session = self._sessions.get(session_id)

        if chat_session is None:
            return BrainResponse(
                text="Gemini is not configured yet. Add GEMINI_API_KEY to the root .env file or container environment, then restart the backend.",
                tool_schemas=self.tool_schemas,
            )

        session_lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with session_lock:
            try:
                response = await asyncio.to_thread(chat_session.send_message, user_text)
            except GoogleAPIError as error:
                logger.exception("Gemini API request failed for session '%s'.", session_id)
                return BrainResponse(
                    text=f"Jarvis could not reach Gemini: {str(error)}",
                    tool_schemas=self.tool_schemas,
                )
            except (TypeError, ValueError) as error:
                logger.exception("Gemini request payload failed for session '%s'.", session_id)
                return BrainResponse(
                    text=f"Jarvis could not process the request payload: {error}",
                    tool_schemas=self.tool_schemas,
                )
            except RuntimeError as error:
                logger.exception("Chat session failed for session '%s'.", session_id)
                return BrainResponse(
                    text=f"Jarvis encountered a runtime error while processing your message: {error}",
                    tool_schemas=self.tool_schemas,
                )

        response_text = getattr(response, "text", "") or "Jarvis completed the request without a text response."

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
        communicate = edge_tts.Communicate(text=text, voice=self._tts_voice)
        audio_bytes = bytearray()

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data = chunk.get("data")
                    if isinstance(audio_data, (bytes, bytearray)):
                        audio_bytes.extend(audio_data)
        except (ClientError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise AudioProcessingError("Jarvis could not generate Thai speech output.") from error

        if not audio_bytes:
            raise AudioProcessingError("Jarvis TTS returned no audio data.")

        return AssistantAudioPayload(
            audio_base64=base64.b64encode(bytes(audio_bytes)).decode("utf-8"),
            mime_type="audio/mpeg",
            voice=self._tts_voice,
        )

    @staticmethod
    def _create_model(
        model_name: str,
        system_instruction: str,
        tools: list[Any],
    ) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            model_name=model_name,
            tools=tools,
            system_instruction=system_instruction,
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
            "gemini_configured": str(self._model is not None).lower(),
            "tts_voice": self._tts_voice,
            "stt_language": self._stt_language,
            "status": "ready" if self._model is not None else "degraded",
        }

    @staticmethod
    def parse_audio_payload(raw_payload: dict[str, Any]) -> AudioChunkPayload:
        try:
            return AudioChunkPayload.model_validate(raw_payload)
        except ValidationError as error:
            raise ValueError("Invalid audio payload.") from error
