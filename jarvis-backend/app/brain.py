from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import re
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import speech_recognition as sr
from pydantic import BaseModel, Field, ValidationError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .config import SystemConfig, load_project_env
from .tts_engine import KokoroTtsEngine, TtsEngineError

logger = logging.getLogger(__name__)
DEFAULT_OLLAMA_MODEL_NAME = "typhoon-v2.5-4b-instruct"
DEFAULT_FALLBACK_OLLAMA_MODEL_NAME = "typhoon-v2.5-4b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 60.0
DEFAULT_OLLAMA_PULL_TIMEOUT_SECONDS = 1800.0
DEFAULT_OLLAMA_TEMPERATURE = 0.7
DEFAULT_OLLAMA_NUM_CTX = 2048
FALLBACK_OLLAMA_NUM_CTX = 1024
DEFAULT_PREWARM_NUM_CTX = 64
DEFAULT_OLLAMA_KEEP_ALIVE = -1
DEFAULT_OLLAMA_NUM_THREAD = 4
DEFAULT_HTTP_CHAT_NUM_THREAD = 4
DEFAULT_SENTENCE_TTS_CONCURRENCY = 2
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
        self._system_config = SystemConfig()
        if model_name is not None:
            self._preferred_model_name = model_name
        else:
            self._preferred_model_name = self._system_config.get_active_model()
        self._fallback_model_name = DEFAULT_FALLBACK_OLLAMA_MODEL_NAME
        self._model_name = self._preferred_model_name
        self._ollama_base_url = DEFAULT_OLLAMA_BASE_URL
        self._ollama_timeout = DEFAULT_OLLAMA_TIMEOUT_SECONDS
        self._ollama_pull_timeout = DEFAULT_OLLAMA_PULL_TIMEOUT_SECONDS
        self._tts_engine = KokoroTtsEngine()
        self._tts_voice = self._tts_engine.voice_label
        self._stt_language = DEFAULT_STT_LANGUAGE
        self._system_instruction = (
            "You are Jarvis. Stay polite, concise, and use a British professional tone. "
            "Acknowledge the current mode (Eco/Performance) only if asked. "
            "For small talk, keep replies brief. For technical requests, respond clearly and directly. "
            "Avoid unnecessary verbosity and keep output practical. "
            "Always answer in English unless the user specifically requests Thai. "
        )
        self._sessions: dict[str, list[dict[str, str]] | None] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._initialization_lock = asyncio.Lock()
        self._model_switch_lock = asyncio.Lock()
        self._initialized = False
        self._ollama_available = False
        self._initialization_error: str | None = None
        self._tool_schemas = self._build_tool_schemas()
        self._sentence_tts_semaphore = asyncio.Semaphore(DEFAULT_SENTENCE_TTS_CONCURRENCY)

    def _active_model_name(self) -> str:
        current_model = self._system_config.get_active_model()
        self._model_name = current_model
        return current_model

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
                self._ollama_available, self._initialization_error = await self._ensure_model_ready()
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

    async def _ensure_model_ready(self) -> tuple[bool, str | None]:
        self._model_name = self._active_model_name()
        healthy, payload_or_error = await self._fetch_ollama_tags()
        if not healthy:
            return False, payload_or_error

        assert isinstance(payload_or_error, dict)
        available_models = self._extract_model_names(payload_or_error)
        if self._model_name in available_models:
            return True, None

        logger.warning("Ollama model '%s' is missing. Attempting pull.", self._model_name)
        pulled, pull_error = await self._pull_model(self._model_name)
        if pulled:
            return True, None

        if self._fallback_model_name != self._model_name:
            logger.warning(
                "Could not pull preferred model '%s' (%s). Falling back to '%s'.",
                self._model_name,
                pull_error or "unknown error",
                self._fallback_model_name,
            )
            self._model_name = self._fallback_model_name
            if self._model_name not in available_models:
                fallback_pulled, fallback_error = await self._pull_model(self._model_name)
                if not fallback_pulled:
                    return (
                        False,
                        (
                            f"Failed to pull preferred model '{self._preferred_model_name}' and "
                            f"fallback model '{self._fallback_model_name}': {fallback_error or pull_error or 'unknown error'}"
                        ),
                    )
            return True, None

        return False, f"Failed to pull model '{self._model_name}': {pull_error or 'unknown error'}"

    async def _fetch_ollama_tags(self) -> tuple[bool, dict[str, Any] | str]:
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

        return True, payload

    async def _pull_model(self, model_name: str) -> tuple[bool, str | None]:
        request_payload = {
            "name": model_name,
            "stream": False,
        }

        try:
            timeout = httpx.Timeout(self._ollama_pull_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                response = await client.post("/api/pull", json=request_payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            return False, str(error)
        except httpx.HTTPError as error:
            return False, str(error)

        if response.status_code >= 400:
            return False, response.text.strip() or f"HTTP {response.status_code}"

        try:
            payload = response.json()
        except ValueError:
            return True, None

        payload_error = payload.get("error")
        if isinstance(payload_error, str) and payload_error.strip():
            return False, payload_error

        return True, None

    @staticmethod
    def _extract_model_names(payload: dict[str, Any]) -> set[str]:
        models = payload.get("models")
        if not isinstance(models, list):
            return set()

        resolved: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            model_name = model.get("name")
            if not isinstance(model_name, str):
                continue
            resolved.add(model_name)
            resolved.add(model_name.split(":", 1)[0])

        return resolved

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
            model_name = self._active_model_name()
            try:
                response_text = await self._generate_ollama_response(messages, model_name)
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
                        f"Jarvis local model '{model_name}' is not ready. "
                        f"Run 'ollama pull {model_name}' and retry. "
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

    async def _check_ollama_health(self, model_name: str | None = None) -> tuple[bool, str | None]:
        model_name = model_name or self._active_model_name()
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
            and model.get("name", "").split(":", 1)[0] == model_name
            for model in models
        )
        if not model_name_matches and models:
            logger.warning(
                "Configured model '%s' not currently listed in Ollama tags. It may need to be pulled.",
                model_name,
            )

        return True, None

    async def _generate_ollama_response(self, messages: list[dict[str, str]], model_name: str) -> str:
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": DEFAULT_OLLAMA_TEMPERATURE,
                "num_ctx": DEFAULT_OLLAMA_NUM_CTX,
                "num_thread": DEFAULT_OLLAMA_NUM_THREAD,
            },
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

            # Retry once with a smaller context when Ollama runner fails to load 8B model reliably.
            if "runner process has terminated" in detail.lower():
                retry_payload = {
                    **request_payload,
                    "options": {
                        "temperature": DEFAULT_OLLAMA_TEMPERATURE,
                        "num_ctx": FALLBACK_OLLAMA_NUM_CTX,
                        "num_thread": DEFAULT_OLLAMA_NUM_THREAD,
                    },
                }
                try:
                    timeout = httpx.Timeout(self._ollama_timeout)
                    async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                        retry_response = await client.post("/api/chat", json=retry_payload)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
                    raise OllamaUnavailableError(str(error)) from error
                except httpx.HTTPError as error:
                    raise OllamaUnavailableError(str(error)) from error

                if retry_response.status_code < 400:
                    response = retry_response
                else:
                    detail = retry_response.text.strip() or f"HTTP {retry_response.status_code}"

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

    async def _generate_ollama_response_streaming(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        on_delta: Callable[[str], Awaitable[None]],
        *,
        num_thread: int = DEFAULT_OLLAMA_NUM_THREAD,
    ) -> str:
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": DEFAULT_OLLAMA_TEMPERATURE,
                "num_ctx": DEFAULT_OLLAMA_NUM_CTX,
                "num_thread": num_thread,
            },
        }

        try:
            timeout = httpx.Timeout(self._ollama_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                async with client.stream("POST", "/api/chat", json=request_payload) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", errors="ignore").strip()
                        detail = detail or f"HTTP {response.status_code}"
                        if response.status_code in (400, 404):
                            lower_detail = detail.lower()
                            if "model" in lower_detail and ("not found" in lower_detail or "missing" in lower_detail):
                                raise OllamaModelError(detail)
                        raise OllamaUnavailableError(detail)

                    accumulated = ""
                    async for line in response.aiter_lines():
                        if not line or not line.strip():
                            continue

                        try:
                            payload = json.loads(line)
                        except ValueError as error:
                            raise OllamaResponseError(f"invalid streaming JSON payload ({error})") from error

                        payload_error = payload.get("error")
                        if isinstance(payload_error, str) and payload_error.strip():
                            lower_error = payload_error.lower()
                            if "model" in lower_error and ("not found" in lower_error or "missing" in lower_error):
                                raise OllamaModelError(payload_error)
                            raise OllamaUnavailableError(payload_error)

                        message_payload = payload.get("message")
                        if isinstance(message_payload, dict):
                            chunk = message_payload.get("content")
                            if isinstance(chunk, str) and chunk:
                                accumulated += chunk
                                await on_delta(chunk)

                    if not accumulated.strip():
                        raise OllamaResponseError("assistant message content is empty")

                    return accumulated.strip()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            raise OllamaUnavailableError(str(error)) from error
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(str(error)) from error

    async def _prewarm_model(self, model_name: str) -> None:
        prewarm_payload = {
            "model": model_name,
            "messages": [],
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_ctx": DEFAULT_PREWARM_NUM_CTX,
                "num_thread": DEFAULT_OLLAMA_NUM_THREAD,
            },
        }
        try:
            timeout = httpx.Timeout(self._ollama_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                response = await client.post("/api/chat", json=prewarm_payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            raise OllamaUnavailableError(str(error)) from error
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(str(error)) from error

        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            lower_detail = detail.lower()
            if response.status_code in (400, 404) and "model" in lower_detail and (
                "not found" in lower_detail or "missing" in lower_detail
            ):
                raise OllamaModelError(detail)
            raise OllamaUnavailableError(detail)

    async def set_active_mode(self, mode: str, prewarm: bool = False) -> dict[str, Any]:
        async with self._model_switch_lock:
            target_model = self._system_config.resolve_model(mode)

            healthy, payload_or_error = await self._fetch_ollama_tags()
            if not healthy:
                raise OllamaUnavailableError(str(payload_or_error))

            assert isinstance(payload_or_error, dict)
            available_models = self._extract_model_names(payload_or_error)
            if target_model not in available_models:
                if mode == "performance":
                    raise OllamaModelError("Performance model not found. Run 'ollama pull' first.")
                raise OllamaModelError(f"Model '{target_model}' not found. Run 'ollama pull {target_model}' first.")

            self._system_config.set_mode(mode)
            self._model_name = target_model
            self._ollama_available = True
            self._initialization_error = None

            prewarm_warning: str | None = None
            if prewarm:
                try:
                    await self._prewarm_model(target_model)
                except (OllamaUnavailableError, OllamaModelError) as error:
                    prewarm_warning = str(error)

            return {
                "active_mode": mode,
                "active_model": target_model,
                "message": f"Switched to {mode} mode.",
                "prewarm_attempted": prewarm,
                "prewarm_warning": prewarm_warning,
            }

    async def prewarm_model_non_blocking(self, model_name: str) -> None:
        try:
            await self._prewarm_model(model_name)
        except (OllamaUnavailableError, OllamaModelError, OllamaResponseError) as error:
            logger.warning("Model prewarm failed for '%s': %s", model_name, error)

    async def handle_text_streaming(
        self,
        session_id: str,
        user_text: str,
        on_stream_start: Callable[[str], Awaitable[None]],
        on_stream_delta: Callable[[str], Awaitable[None]],
        on_stream_end: Callable[[str], Awaitable[None]],
    ) -> BrainResponse:
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
            model_name = self._active_model_name()

            try:
                await on_stream_start(model_name)
                response_text = await self._generate_ollama_response_streaming(
                    messages=messages,
                    model_name=model_name,
                    on_delta=on_stream_delta,
                    num_thread=DEFAULT_OLLAMA_NUM_THREAD,
                )
                await on_stream_end(response_text)
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
                        f"Jarvis local model '{model_name}' is not ready. "
                        f"Run 'ollama pull {model_name}' and retry. "
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

    async def synthesize_manual_text(self, text: str) -> AssistantAudioPayload:
        return await self._synthesize_assistant_audio(text)

    @staticmethod
    def _extract_sentence_chunks(buffer: str) -> tuple[list[str], str]:
        chunks: list[str] = []
        boundary_pattern = re.compile(r"[.!?\n]")
        cursor = 0

        while True:
            match = boundary_pattern.search(buffer, cursor)
            if match is None:
                break

            boundary_index = match.end()
            chunk = buffer[:boundary_index].strip()
            if chunk:
                chunks.append(chunk)
            buffer = buffer[boundary_index:]
            cursor = 0

        return chunks, buffer

    async def _synthesize_sentence_chunk(
        self,
        sentence_index: int,
        sentence_text: str,
        on_sentence_audio: Callable[[int, str, AssistantAudioPayload | None, str | None], Awaitable[None]],
    ) -> None:
        try:
            async with self._sentence_tts_semaphore:
                audio_payload = await self._synthesize_assistant_audio(sentence_text)
            await on_sentence_audio(sentence_index, sentence_text, audio_payload, None)
        except AudioProcessingError as error:
            await on_sentence_audio(sentence_index, sentence_text, None, str(error))

    async def handle_http_chat_streaming(
        self,
        session_id: str,
        user_text: str,
        auto_speak: bool,
        on_text_chunk: Callable[[str], Awaitable[None]],
        on_sentence_audio: Callable[[int, str, AssistantAudioPayload | None, str | None], Awaitable[None]],
    ) -> str:
        if not user_text.strip():
            raise RuntimeError("I did not receive any text to process.")

        await self.create_session(session_id)
        messages = self._sessions.get(session_id)
        if messages is None:
            raise RuntimeError("Jarvis session state is unavailable. Please restart the session.")

        session_lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        sentence_buffer = ""
        sentence_index = 0
        sentence_tasks: list[asyncio.Task[None]] = []

        async def on_delta(delta: str) -> None:
            nonlocal sentence_buffer, sentence_index
            if not delta:
                return

            await on_text_chunk(delta)

            if not auto_speak:
                return

            sentence_buffer += delta
            completed_chunks, sentence_buffer = self._extract_sentence_chunks(sentence_buffer)
            for chunk in completed_chunks:
                current_index = sentence_index
                sentence_index += 1
                sentence_tasks.append(
                    asyncio.create_task(
                        self._synthesize_sentence_chunk(
                            sentence_index=current_index,
                            sentence_text=chunk,
                            on_sentence_audio=on_sentence_audio,
                        )
                    )
                )

        async with session_lock:
            messages.append({"role": "user", "content": user_text})
            model_name = self._active_model_name()

            try:
                final_text = await self._generate_ollama_response_streaming(
                    messages=messages,
                    model_name=model_name,
                    on_delta=on_delta,
                    num_thread=DEFAULT_HTTP_CHAT_NUM_THREAD,
                )

                if auto_speak:
                    trailing_chunk = sentence_buffer.strip()
                    if trailing_chunk:
                        current_index = sentence_index
                        sentence_index += 1
                        sentence_tasks.append(
                            asyncio.create_task(
                                self._synthesize_sentence_chunk(
                                    sentence_index=current_index,
                                    sentence_text=trailing_chunk,
                                    on_sentence_audio=on_sentence_audio,
                                )
                            )
                        )

                if sentence_tasks:
                    await asyncio.gather(*sentence_tasks, return_exceptions=False)

                final_text = final_text.strip()
                messages.append({"role": "assistant", "content": final_text})
                self._ollama_available = True
                self._initialization_error = None
                return final_text
            except (OllamaUnavailableError, OllamaModelError, OllamaResponseError, RuntimeError):
                for task in sentence_tasks:
                    if not task.done():
                        task.cancel()
                if sentence_tasks:
                    await asyncio.gather(*sentence_tasks, return_exceptions=True)
                messages.pop()
                raise

    async def get_system_status(self) -> dict[str, Any]:
        active_mode = self._system_config.get_mode()
        active_model = self._active_model_name()
        models = self._system_config.get_models()
        healthy, health_error = await self._check_ollama_health(active_model)

        self._ollama_available = healthy
        self._initialization_error = None if healthy else health_error

        return {
            "service": "jarvis-backend",
            "active_mode": active_mode,
            "active_model": active_model,
            "models": models,
            "ollama_base_url": self._ollama_base_url,
            "system_load": {
                "ollama_ready": healthy,
            },
            "status": "ready" if healthy else "degraded",
            "error": health_error,
        }

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
        active_mode = self._system_config.get_mode()
        active_model = self._active_model_name()
        return {
            "service": "jarvis-backend",
            "mode": active_mode,
            "model_name": active_model,
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
