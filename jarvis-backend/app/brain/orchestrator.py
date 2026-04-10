"""brain/orchestrator.py — Central coordination layer (replaces JarvisBrain).

The Orchestrator is the single entry point for all higher-level operations:
    - Initialisation (model readiness, GPU preflight, prewarm)
    - Per-session memory management
    - HTTP chat streaming (NDJSON)
    - WebSocket text streaming
    - Non-streaming text inference
    - Voice audio input (STT → LLM → TTS)
    - System status + mode switching
    - Host-optimizer callback
    - Memory pressure guardrails
    - Performance-mode auto-recovery

Hardware affinity note
----------------------
On WSL2/Docker Desktop, taskset and cgroup CPU pinning have limited effect
because Docker Desktop virtualises the Linux kernel. The primary knob is
`num_thread` sent in every Ollama request:
    Performance mode → 8 threads (targets all P-Cores on i5-13420H)
    Eco mode         → 2 threads (leaves headroom for UI / TTS worker)
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import httpx
import psutil
import speech_recognition as sr
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from app.actuators.registry import tool_registry  # activates tool auto-registration
from app.actuators import system_tools as _system_tools_module  # noqa: F401 — side-effects only
from app.brain.exceptions import (
    AudioProcessingError,
    OllamaModelError,
    OllamaResponseError,
    OllamaUnavailableError,
)
from app.brain.memory import SessionMemory
from app.brain.memory_guardian import (
    PAGEFILE_GUARDRAIL_PERCENT,
    GuardianDecision,
    MemoryGuardian,
)
from app.brain.models import (
    AssistantAudioPayload,
    AudioChunkAcknowledgement,
    AudioChunkPayload,
    BrainResponse,
    SpeechTranscriptPayload,
    ToolSchema,
    VoiceInteractionResult,
)
from app.brain.ollama_client import OllamaClient
from app.brain.streamer import AsyncChunkStreamer
from app.config.settings import (
    DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT,
    DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS,
    DEFAULT_LOW_RAM_FORCE_ECO_BYTES,
    DEFAULT_LLM_MEMORY_CAP_BYTES,
    DEFAULT_MODE_SWITCH_COOLDOWN_SECONDS,
    DEFAULT_PERFORMANCE_RETRY_COOLDOWN_SECONDS,
    DEFAULT_SENTENCE_TTS_CONCURRENCY,
    DEFAULT_STREAM_TEXT_FLUSH_MIN_CHARS,
    DEFAULT_STREAM_TEXT_FLUSH_SECONDS,
    DEFAULT_STT_LANGUAGE,
    FALLBACK_PROFILE,
    HTTP_CHAT_NUM_THREAD,
    PERFORMANCE_PROFILE,
    Settings,
    load_project_env,
)
from app.tts.host_client import TtsEngineError, TtsHostClient

logger = logging.getLogger(__name__)

_MAX_TOOL_CALL_ROUNDS = 4
_MAX_TOOL_RESULT_CHARS = 900

# Regex to strip Qwen-family <think>...</think> reasoning blocks from output.
_THINK_TAG_RE = __import__("re").compile(r"<think>.*?</think>", __import__("re").DOTALL)

_SYSTEM_INSTRUCTION = (
    "You are Jarvis. Stay polite, concise, and use a British professional tone. "
    "Acknowledge the current mode (Eco/Performance) only if asked. "
    "For small talk, keep replies brief. For technical requests, respond clearly and directly. "
    "Avoid unnecessary verbosity and keep output practical. "
    "If the user asks about battery, RAM, memory, or CPU status, you MUST call get_system_status first. "
    "Never guess hardware or system metrics from internal knowledge. "
    "When asked what applications you can open, you MUST check the description of the open_local_app tool and list ONLY the exact supported applications. Do not assume or guess applications like Google Docs. "
    "If the user asks to open a file or folder on the local machine, use the open_path tool. "
    "If the user asks what's in a folder or to browse files on the local machine, use the browse_filesystem tool. "
    "If the user asks to find a file on the local machine, use the search_files tool first, then open_path if requested. "
    "Never fabricate file paths. Always use tools to verify and locate local files. "
    "Always answer in English unless the user specifically requests Thai. "
)


class Orchestrator:
    """Central logic coordinator for the Jarvis AI assistant."""

    def __init__(self) -> None:
        load_project_env()
        self._settings = Settings()
        self._memory = SessionMemory(max_tokens=self._settings.performance_profile.num_ctx)
        self._ollama = OllamaClient(self._settings)
        self._tts = TtsHostClient()
        self._tts_voice: str = self._tts.voice_label
        self._stt_language: str = DEFAULT_STT_LANGUAGE

        # Init state
        self._initialized: bool = False
        self._initialization_lock = asyncio.Lock()
        self._ollama_available: bool = False
        self._initialization_error: str | None = None

        # GPU / KV-cache preflight
        self._gpu_preflight_checked: bool = False
        self._gpu_preflight_status: str = "disabled"
        self._gpu_preflight_message: str = "Intel iGPU acceleration not requested."
        self._gpu_soft_fallback: bool = False

        # Pressure guardrails
        self._llm_memory_cap_bytes: int = DEFAULT_LLM_MEMORY_CAP_BYTES
        self._low_ram_force_eco_bytes: int = DEFAULT_LOW_RAM_FORCE_ECO_BYTES
        self._high_swap_force_eco_percent: float = DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT
        self._pagefile_guardrail_percent: float = PAGEFILE_GUARDRAIL_PERCENT
        self._last_pressure_error: str | None = None
        self._last_low_ram_force_eco_message: str | None = None

        # Performance auto-recovery
        self._performance_downgrade_active: bool = False
        self._last_performance_downgrade_at: str | None = None
        self._last_performance_downgrade_reason: str | None = None
        self._performance_retry_after_monotonic: float = 0.0
        self._performance_retry_cooldown: float = DEFAULT_PERFORMANCE_RETRY_COOLDOWN_SECONDS

        # Mode switch cooldown
        self._mode_switch_lock = asyncio.Lock()
        self._last_mode_switch_monotonic: float = 0.0
        self._mode_switch_cooldown: float = DEFAULT_MODE_SWITCH_COOLDOWN_SECONDS

        # Disk I/O telemetry
        self._last_disk_io_bytes: int | None = None
        self._last_disk_io_monotonic: float = 0.0
        self._metrics_lock = asyncio.Lock()
        self._memory_guardian = MemoryGuardian(
            low_ram_force_eco_bytes=self._low_ram_force_eco_bytes,
            high_swap_force_eco_percent=self._high_swap_force_eco_percent,
            pagefile_guardrail_percent=self._pagefile_guardrail_percent,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return tool_registry.get_schemas()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Idempotent startup: ensure model is ready, run GPU preflight, prewarm."""
        if self._initialized:
            return
        async with self._initialization_lock:
            if self._initialized:
                return
            try:
                await self._ollama.open()
                self._ollama_available, self._initialization_error = await self._ensure_model_ready()
                await self._run_gpu_preflight()
                if self._ollama_available:
                    asyncio.create_task(
                        self._prewarm_non_blocking(self._settings.get_active_model())
                    )
            except (RuntimeError, TypeError, ValueError) as err:
                self._ollama_available = False
                self._initialization_error = str(err)
                logger.exception("Orchestrator initialization failed: %s", err)
            self._initialized = True
            if self._ollama_available:
                logger.info(
                    "Orchestrator ready — model='%s' url='%s'",
                    self._settings.get_active_model(),
                    self._settings.ollama_base_url,
                )
            else:
                logger.warning(
                    "Orchestrator in degraded mode — Ollama unavailable: %s",
                    self._initialization_error or "unknown",
                )

    async def shutdown(self) -> None:
        """Close Ollama connection pool gracefully."""
        await self._ollama.close()

    # ------------------------------------------------------------------
    # Session management (delegated to SessionMemory)
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str) -> None:
        await self.initialize()
        await self._memory.create_session(session_id, _SYSTEM_INSTRUCTION)

    async def close_session(self, session_id: str) -> None:
        await self._memory.close_session(session_id)

    # ------------------------------------------------------------------
    # Non-streaming text inference (WebSocket path)
    # ------------------------------------------------------------------

    async def handle_text(self, session_id: str, user_text: str) -> BrainResponse:
        if not user_text.strip():
            return BrainResponse(text="I did not receive any text to process.", tool_schemas=self.tool_schemas)

        await self._maybe_restore_performance_mode()
        await self.create_session(session_id)
        session_lock = self._memory.get_lock(session_id)

        async with session_lock:
            self._memory.append_user(session_id, user_text)
            model_name, profile = self._resolve_model_and_profile()
            try:
                await self._enforce_pressure_guardrail(model_name)
                await self._invoke_host_optimizer(model_name=model_name, mode=profile.mode)
                session_messages = [
                    dict(message) for message in (self._memory.get_messages(session_id) or [])
                ]
                response_text = await self._generate_with_tool_loop(
                    messages=session_messages,
                    model_name=model_name,
                    profile=profile,
                )
                self._memory.append_assistant(session_id, response_text)
                self._ollama_available = True
            except OllamaUnavailableError as err:
                self._ollama_available = False
                self._initialization_error = str(err)
                self._memory.pop_last(session_id)
                return BrainResponse(
                    text=f"Jarvis could not reach Ollama: {err}",
                    tool_schemas=self.tool_schemas,
                )
            except (OllamaModelError, OllamaResponseError, RuntimeError) as err:
                self._memory.pop_last(session_id)
                return BrainResponse(text=str(err), tool_schemas=self.tool_schemas)

        return await self._attach_audio(BrainResponse(text=response_text, tool_schemas=self.tool_schemas))

    @staticmethod
    def _format_mode_label(mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized == "performance":
            return "Performance"
        if normalized == "eco":
            return "Eco"
        return mode or "Unknown"

    def _build_tool_protocol_prompt(self, current_mode: str) -> str:
        lines: list[str] = []
        for schema in self.tool_schemas:
            props = schema.parameters.get("properties", {})
            arg_names = ",".join(str(k) for k in props.keys())
            lines.append(f"- {schema.name}({arg_names})")

        tools_text = "\n".join(lines) if lines else "- (no tools available)"
        mode_label = self._format_mode_label(current_mode)
        return (
            "Tool protocol for this turn:\n"
            f"You are currently in {mode_label} mode.\n"
            "Do not invent or assume reasons for missing data based on your mode.\n"
            "If a tool is required, output only compact JSON with no markdown: "
            '{"t":"tool","n":"tool_name","a":{}}.\n'
            "If no tool is required, answer naturally in plain text.\n"
            "Available tools:\n"
            f"{tools_text}"
        )

    def _inject_tool_protocol(
        self,
        messages: list[dict[str, str]],
        current_mode: str,
    ) -> list[dict[str, str]]:
        protocol_message = {
            "role": "system",
            "content": self._build_tool_protocol_prompt(current_mode=current_mode),
        }
        if messages and messages[0].get("role") == "system":
            return [messages[0], protocol_message, *messages[1:]]
        return [protocol_message, *messages]

    @staticmethod
    def _extract_tool_call(response_text: str) -> tuple[str, dict[str, Any]] | None:
        payload = response_text.strip()
        if not payload:
            return None

        if payload.startswith("```"):
            payload = "\n".join(
                line for line in payload.splitlines() if not line.strip().startswith("```")
            ).strip()

        decoder = json.JSONDecoder()
        candidate: dict[str, Any] | None = None

        for start_index in [0, payload.find("{")]:
            if start_index < 0:
                continue
            try:
                decoded, _ = decoder.raw_decode(payload[start_index:])
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(decoded, dict):
                candidate = decoded
                break

        if not isinstance(candidate, dict):
            return None
        if candidate.get("t") != "tool":
            return None

        tool_name = candidate.get("n")
        tool_args = candidate.get("a", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            return None
        if not isinstance(tool_args, dict):
            return None
        return tool_name.strip(), tool_args

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result)
        if len(text) > _MAX_TOOL_RESULT_CHARS:
            return text[:_MAX_TOOL_RESULT_CHARS] + " ...[truncated]"
        return text

    async def _assess_tool_call(self, tool_name: str, active_mode: str) -> GuardianDecision:
        metadata = tool_registry.get_metadata(tool_name)
        runtime_metrics = await self._collect_runtime_metrics()
        return self._memory_guardian.assess(
            tool_name=tool_name,
            tool_metadata=metadata,
            active_mode=active_mode,
            runtime_metrics=runtime_metrics,
        )

    @staticmethod
    def _guardian_message(decision: GuardianDecision) -> str:
        if decision.requires_mode_confirmation and decision.suggested_mode:
            if decision.tool_name == "get_system_status":
                return (
                    "I can see your battery, RAM, and CPU through get_system_status, "
                    f"but I need your confirmation to switch to {decision.suggested_mode} mode "
                    "first so I can keep memory stable."
                )
            return (
                f"I need your confirmation to switch to {decision.suggested_mode} mode "
                f"before I run '{decision.tool_name}'."
            )
        if decision.tool_name == "get_system_status":
            return (
                "I can see your battery, RAM, and CPU, but I need to save memory right now. "
                f"{decision.reason}"
            )
        return (
            f"I’m pausing '{decision.tool_name}' to keep the system stable. "
            f"{decision.reason}"
        )

    async def _generate_with_tool_loop(
        self,
        *,
        messages: list[dict[str, str]],
        model_name: str,
        profile: Any,
    ) -> str:
        working_messages = [dict(message) for message in messages]
        original_user_request = next(
            (
                str(message.get("content", "")).strip()
                for message in reversed(messages)
                if message.get("role") == "user" and str(message.get("content", "")).strip()
            ),
            "",
        )

        for _ in range(_MAX_TOOL_CALL_ROUNDS + 1):
            protocol_messages = self._inject_tool_protocol(
                working_messages,
                current_mode=profile.mode,
            )
            response_text = await self._ollama.generate(
                protocol_messages,
                model_name,
                profile,
                num_gpu=self._resolve_num_gpu(profile.mode),
            )
            response_text = self._strip_thinking_tags(response_text)
            parsed = self._extract_tool_call(response_text)
            if parsed is None:
                return response_text.strip()

            tool_name, tool_args = parsed
            if not tool_registry.has_tool(tool_name):
                working_messages.append(
                    {"role": "assistant", "content": f"Tool '{tool_name}' is unavailable."}
                )
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That tool is unavailable. Continue without tools and "
                            "answer naturally."
                        ),
                    }
                )
                continue

            decision = await self._assess_tool_call(tool_name=tool_name, active_mode=profile.mode)
            if not decision.allowed:
                return self._guardian_message(decision)

            try:
                tool_result = await tool_registry.execute(tool_name, tool_args)
            except (KeyError, ValueError, RuntimeError) as err:
                working_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Tool '{tool_name}' failed with error: {err}",
                    }
                )
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool execution failed. Continue without that tool and "
                            "answer naturally."
                        ),
                    }
                )
                continue

            result_text = self._format_tool_result(tool_result)
            working_messages.append({"role": "assistant", "content": f"Tool '{tool_name}' executed."})
            working_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result ({tool_name}): {result_text}\n"
                        f"Original user request: {original_user_request or 'Use the user request in context.'}\n"
                        "Now answer the original user request directly in natural language.\n"
                        "If the user asked 'why', include clear causal reasoning grounded in the tool result.\n"
                        "If any metric is unavailable, state that it is unavailable.\n"
                        "If the user contradicts sensor output, acknowledge sensors may be stale or failing.\n"
                        "Do not argue with the user and do not invent theories to defend sensor output.\n"
                        "Never gaslight the user.\n"
                        "Do not guess missing values or blame the current mode.\n"
                        "Do not output JSON, markdown code fences, or raw object dumps."
                    ),
                }
            )

        return "I could not complete the request safely within the local tool-call limit."

    @classmethod
    def _sanitize_visible_text(cls, text: str) -> str:
        cleaned = cls._strip_thinking_tags(text).strip()
        if not cleaned:
            return "I did not generate a response."

        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines() if not line.strip().startswith("```")
            ).strip()

        if cls._extract_tool_call(cleaned) is not None:
            return "Working on that now."

        visible_lines: list[str] = []
        for line in cleaned.splitlines():
            if cls._extract_tool_call(line.strip()) is not None:
                continue
            visible_lines.append(line)

        visible_text = "\n".join(visible_lines).strip()
        return visible_text or "Done."

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove Qwen-family <think>...</think> reasoning blocks from model output."""
        return _THINK_TAG_RE.sub("", text).strip()

    @staticmethod
    def _chunk_text_for_streaming(text: str, chunk_size: int = 56) -> list[str]:
        chunks: list[str] = []
        cursor = 0
        limit = max(16, int(chunk_size))

        while cursor < len(text):
            end = min(len(text), cursor + limit)
            split = end
            if end < len(text):
                window = text[cursor:end]
                pivot = max(window.rfind(" "), window.rfind("\n"))
                if pivot >= 16:
                    split = cursor + pivot + 1
            chunk = text[cursor:split]
            if chunk:
                chunks.append(chunk)
            cursor = split

        return chunks

    async def _emit_stream_deltas(
        self,
        text: str,
        on_stream_delta: Callable[[str], Awaitable[None]],
    ) -> None:
        for chunk in self._chunk_text_for_streaming(text):
            await on_stream_delta(chunk)
            await asyncio.sleep(0)

    @staticmethod
    def _needs_system_status_tool(user_text: str) -> bool:
        normalized = user_text.lower()
        keywords = (
            "battery",
            "cpu",
            "ram",
            "memory",
            "pagefile",
            "swap",
            "system status",
            "check the sys",
            "สถานะเครื่อง",
            "แบต",
            "แรม",
            "ซีพียู",
        )
        return any(keyword in normalized for keyword in keywords)

    async def _build_streaming_messages(
        self,
        *,
        messages: list[dict[str, str]],
        user_text: str,
        active_mode: str,
    ) -> tuple[list[dict[str, str]], str | None]:
        """Build a fast streaming prompt path.

        For system-health queries we execute get_system_status first, then stream a
        grounded synthesis answer. For all other queries we stream directly.
        """
        if not self._needs_system_status_tool(user_text):
            return messages, None

        decision = await self._assess_tool_call(tool_name="get_system_status", active_mode=active_mode)
        if not decision.allowed:
            return [], self._guardian_message(decision)

        try:
            tool_result = await tool_registry.execute("get_system_status", {})
        except (KeyError, ValueError, RuntimeError) as err:
            return [], (
                "I could not run the local system-status tool right now. "
                f"Error: {err}"
            )

        result_text = self._format_tool_result(tool_result)
        grounded_messages = [dict(message) for message in messages]
        grounded_messages.append({"role": "assistant", "content": "Tool 'get_system_status' executed."})
        grounded_messages.append(
            {
                "role": "user",
                "content": (
                    f"Tool result (get_system_status): {result_text}\n"
                    f"Original user request: {user_text.strip()}\n"
                    "Answer directly and naturally.\n"
                    "If the user asked why, provide causal reasoning grounded only in this tool result.\n"
                    "If a value is unavailable, say it is unavailable.\n"
                    "If the user contradicts sensor output, acknowledge sensors may be stale or failing.\n"
                    "Do not argue with the user and do not invent theories to defend sensor output.\n"
                    "Never gaslight the user.\n"
                    "Do not output JSON or raw object dumps."
                ),
            }
        )
        return grounded_messages, None

    # ------------------------------------------------------------------
    # WebSocket streaming text inference
    # ------------------------------------------------------------------

    async def handle_text_streaming(
        self,
        session_id: str,
        user_text: str,
        on_stream_start: Callable[[str], Awaitable[None]],
        on_stream_delta: Callable[[str], Awaitable[None]],
        on_stream_end: Callable[[str], Awaitable[None]],
    ) -> BrainResponse:
        if not user_text.strip():
            return BrainResponse(text="I did not receive any text to process.", tool_schemas=self.tool_schemas)

        await self._maybe_restore_performance_mode()
        await self.create_session(session_id)
        session_lock = self._memory.get_lock(session_id)

        async with session_lock:
            self._memory.append_user(session_id, user_text)
            model_name, profile = self._resolve_model_and_profile()
            try:
                await self._enforce_pressure_guardrail(model_name)
                await self._invoke_host_optimizer(model_name=model_name, mode=profile.mode)
                await on_stream_start(model_name)

                session_messages = [
                    dict(message) for message in (self._memory.get_messages(session_id) or [])
                ]
                streaming_messages, immediate_text = await self._build_streaming_messages(
                    messages=session_messages,
                    user_text=user_text,
                    active_mode=profile.mode,
                )

                if immediate_text is not None:
                    visible_text = self._sanitize_visible_text(immediate_text)
                    await self._emit_stream_deltas(visible_text, on_stream_delta)
                else:
                    visible_text = await self._ollama.generate_streaming(
                        streaming_messages,
                        model_name,
                        profile,
                        on_stream_delta,
                        num_gpu=self._resolve_num_gpu(profile.mode),
                    )
                    visible_text = self._sanitize_visible_text(visible_text)

                await on_stream_end(visible_text)
                self._memory.append_assistant(session_id, visible_text)
                self._ollama_available = True
            except (OllamaUnavailableError, OllamaModelError, OllamaResponseError, RuntimeError) as err:
                self._memory.pop_last(session_id)
                return BrainResponse(text=str(err), tool_schemas=self.tool_schemas)
        return await self._attach_audio(BrainResponse(text=visible_text, tool_schemas=self.tool_schemas))

    # ------------------------------------------------------------------
    # HTTP chat streaming (NDJSON) — primary UI path
    # ------------------------------------------------------------------

    async def handle_http_chat_streaming(
        self,
        session_id: str,
        user_text: str,
        auto_speak: bool,
        num_ctx_override: int | None,
        on_text_chunk: Callable[[str], Awaitable[None]],
        on_sentence_audio: Callable[[int, str, AssistantAudioPayload | None, str | None], Awaitable[None]],
        on_final_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        if not user_text.strip():
            raise RuntimeError("I did not receive any text to process.")

        await self._maybe_restore_performance_mode()
        await self.create_session(session_id)
        session_lock = self._memory.get_lock(session_id)

        tts_synthesize = self._synthesize_audio_raw if auto_speak else None
        tts_callback = on_sentence_audio if auto_speak else None

        streamer = AsyncChunkStreamer(
            tts_synthesize=tts_synthesize,
            on_text_chunk=on_text_chunk,
            on_sentence_audio=tts_callback,
            max_tts_concurrency=DEFAULT_SENTENCE_TTS_CONCURRENCY,
            flush_interval_seconds=DEFAULT_STREAM_TEXT_FLUSH_SECONDS,
            flush_min_chars=DEFAULT_STREAM_TEXT_FLUSH_MIN_CHARS,
        )

        async with session_lock:
            self._memory.append_user(session_id, user_text)
            model_name, profile = self._resolve_model_and_profile()
            http_num_thread = HTTP_CHAT_NUM_THREAD.get(profile.mode, profile.num_thread)
            effective_num_ctx = (
                num_ctx_override
                if isinstance(num_ctx_override, int) and num_ctx_override > 0
                else profile.num_ctx
            )
            effective_profile = PERFORMANCE_PROFILE.__class__(
                mode=profile.mode,
                num_ctx=effective_num_ctx,
                num_thread=http_num_thread,
                num_gpu=profile.num_gpu,
                description=profile.description,
            )

            try:
                await self._enforce_pressure_guardrail(model_name)
                await self._invoke_host_optimizer(model_name=model_name, mode=profile.mode)

                session_messages = [
                    dict(message) for message in (self._memory.get_messages(session_id) or [])
                ]
                streaming_messages, immediate_text = await self._build_streaming_messages(
                    messages=session_messages,
                    user_text=user_text,
                    active_mode=effective_profile.mode,
                )

                if immediate_text is not None:
                    final_text = self._sanitize_visible_text(immediate_text).strip()
                    for chunk in self._chunk_text_for_streaming(final_text):
                        await streamer.on_delta(chunk)
                else:
                    final_text = await self._ollama.generate_streaming(
                        streaming_messages,
                        model_name,
                        effective_profile,
                        streamer.on_delta,
                        num_gpu=self._resolve_num_gpu(effective_profile.mode),
                        num_ctx_override=effective_num_ctx,
                    )
                    final_text = self._sanitize_visible_text(final_text).strip()

                # Flush trailing text and dispatch trailing TTS sentence
                trailing = streamer.get_trailing_sentence_buffer()
                await streamer.flush_final(trailing_buffer=trailing)

                self._memory.append_assistant(session_id, final_text)
                self._ollama_available = True
                self._initialization_error = None

                if on_final_text is not None:
                    try:
                        await on_final_text(final_text)
                    except RuntimeError as err:
                        logger.warning("Final text callback failed: %s", err)

                return final_text

            except (OllamaUnavailableError, OllamaModelError, OllamaResponseError, RuntimeError):
                await streamer.cancel_all()
                self._memory.pop_last(session_id)
                raise

    # ------------------------------------------------------------------
    # Voice (audio chunk → STT → LLM → audio response)
    # ------------------------------------------------------------------

    async def handle_audio_chunk(
        self,
        session_id: str,
        payload: AudioChunkPayload,
    ) -> VoiceInteractionResult:
        decoded_audio = await self._decode_audio_payload(payload)
        wav_audio = await asyncio.to_thread(self._transcode_webm_to_wav, decoded_audio)
        transcript_text = await asyncio.to_thread(self._transcribe_wav_audio, wav_audio)
        response = await self.handle_text(session_id=session_id, user_text=transcript_text)
        return VoiceInteractionResult(
            acknowledgement=AudioChunkAcknowledgement(
                accepted=True,
                chunk_size=len(decoded_audio),
                mime_type=payload.mime_type,
                detail="Audio chunk accepted and processed for transcription.",
            ),
            transcript=SpeechTranscriptPayload(text=transcript_text),
            response=response,
        )

    @staticmethod
    def parse_audio_payload(raw_payload: dict[str, Any]) -> AudioChunkPayload:
        from pydantic import ValidationError as _VE  # avoid top-level import

        try:
            return AudioChunkPayload.model_validate(raw_payload)
        except _VE as err:
            raise ValueError("Invalid audio payload.") from err

    # ------------------------------------------------------------------
    # TTS helpers
    # ------------------------------------------------------------------

    async def synthesize_manual_text(self, text: str) -> AssistantAudioPayload:
        return await self._synthesize_audio(text)

    async def _synthesize_audio(self, text: str) -> AssistantAudioPayload:
        try:
            result = await self._tts.synthesize(text)
        except TtsEngineError as err:
            raise AudioProcessingError(str(err)) from err
        return AssistantAudioPayload(
            audio_base64=result.audio_base64,
            mime_type=result.mime_type,
            voice=result.voice,
        )

    async def _synthesize_audio_raw(self, text: str) -> AssistantAudioPayload:
        """TTS wrapper injected into AsyncChunkStreamer (same signature)."""
        return await self._synthesize_audio(text)

    async def _attach_audio(self, response: BrainResponse) -> BrainResponse:
        try:
            audio = await self._synthesize_audio(response.text)
            return response.model_copy(update={"assistant_audio": audio})
        except AudioProcessingError as err:
            logger.warning("TTS generation failed: %s", err)
            return response.model_copy(update={"audio_error": str(err)})

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    async def get_system_status(self) -> dict[str, Any]:
        active_mode = self._settings.get_mode()
        active_model = self._settings.get_active_model()
        models = self._settings.get_models()
        healthy, health_error = await self._check_ollama_health(active_model)
        runtime_metrics = await self._collect_runtime_metrics()
        tts_runtime = await self._tts.get_runtime_status()

        self._ollama_available = healthy
        self._initialization_error = None if healthy else health_error

        profile = self._settings.get_profile(active_mode)

        return {
            "service": "jarvis-backend",
            "active_mode": active_mode,
            "active_model": active_model,
            "models": models,
            "ollama_base_url": self._settings.ollama_base_url,
            "system_load": {
                "ollama_ready": healthy,
                "GPU_Load": runtime_metrics["gpu_load_percent"],
                "RAM_Available": round(runtime_metrics["ram_available_bytes"] / (1024**3), 2),
                "RAM_Available_Bytes": runtime_metrics["ram_available_bytes"],
                "Disk_IO_MBps": runtime_metrics["disk_io_mbps"],
                "pagefile_usage_percent": runtime_metrics["pagefile_usage_percent"],
                "pagefile_guardrail_percent": runtime_metrics["pagefile_guardrail_percent"],
                "pressure_guardrail_triggered": runtime_metrics["pressure_guardrail_triggered"],
                "telemetry_source": runtime_metrics["telemetry_source"],
            },
            "accelerator": {
                "intel_gpu_requested": self._settings.intel_gpu_requested,
                "gpu_soft_fallback": self._gpu_soft_fallback,
                "preflight_status": self._gpu_preflight_status,
                "preflight_message": self._gpu_preflight_message,
                "kv_cache_supported": self._ollama.kv_cache_supported,
                "kv_cache_warning": self._ollama.kv_cache_warning,
                "performance_downgrade_active": self._performance_downgrade_active,
                "last_performance_downgrade_at": self._last_performance_downgrade_at,
                "last_performance_downgrade_reason": self._last_performance_downgrade_reason,
                "performance_retry_after_ms": max(
                    0, int((self._performance_retry_after_monotonic - time.monotonic()) * 1000)
                ) if self._performance_downgrade_active else 0,
            },
            "guardrail": {
                "last_pressure_error": self._last_pressure_error,
                "low_ram_force_eco_message": self._last_low_ram_force_eco_message,
                "low_ram_force_eco_threshold_bytes": self._low_ram_force_eco_bytes,
                "high_swap_force_eco_percent": self._high_swap_force_eco_percent,
            },
            "profile": {
                "eco": {"num_ctx": self._settings.eco_profile.num_ctx, "num_thread": self._settings.eco_profile.num_thread},
                "performance": {"num_ctx": self._settings.performance_profile.num_ctx, "num_thread": self._settings.performance_profile.num_thread},
                "active_mode_profile": {"num_ctx": profile.num_ctx, "num_thread": profile.num_thread, "num_gpu": profile.num_gpu},
            },
            "memory": self._memory.get_status(),
            "tts": {
                "mode": tts_runtime["mode"],
                "provider": tts_runtime["provider"],
                "status": tts_runtime["status"],
                "primary_url": tts_runtime["primary_url"],
                "active_url": tts_runtime["active_url"],
                "vram_mb": tts_runtime["vram_mb"],
                "speed": tts_runtime.get("speed"),
                "warning": tts_runtime["warning"],
            },
            "tools_registered": len(self.tool_schemas),
            "status": "ready" if healthy else "degraded",
            "error": health_error,
        }

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    async def set_active_mode(self, mode: str, prewarm: bool = False) -> dict[str, Any]:
        async with self._mode_switch_lock:
            now = time.monotonic()
            elapsed = now - self._last_mode_switch_monotonic
            wait_seconds = 0.0
            if self._last_mode_switch_monotonic > 0 and elapsed < self._mode_switch_cooldown:
                wait_seconds = self._mode_switch_cooldown - elapsed
                await asyncio.sleep(wait_seconds)

            target_model = self._settings.resolve_model(mode)
            healthy, payload_or_error = await self._ollama.fetch_tags()
            if not healthy:
                raise OllamaUnavailableError(str(payload_or_error))

            assert isinstance(payload_or_error, dict)
            available = self._ollama.extract_model_names(payload_or_error)
            if target_model not in available:
                raise OllamaModelError(
                    f"Model '{target_model}' not found. Run 'ollama pull {target_model}' first."
                )

            self._settings.set_mode(mode)
            self._ollama_available = True
            self._initialization_error = None
            self._last_mode_switch_monotonic = time.monotonic()

            prewarm_warning: str | None = None
            if prewarm:
                try:
                    await self._ollama.prewarm(
                        target_model,
                        self._settings.get_profile(mode),
                        num_gpu=self._resolve_num_gpu(mode),
                    )
                except (OllamaUnavailableError, OllamaModelError) as err:
                    prewarm_warning = str(err)

            return {
                "active_mode": mode,
                "active_model": target_model,
                "message": f"Switched to {mode} mode.",
                "mode_switch_wait_ms": int(wait_seconds * 1000),
                "prewarm_attempted": prewarm,
                "prewarm_warning": prewarm_warning,
                "prewarm_started": False,
            }

    # ------------------------------------------------------------------
    # Internal — model / pressure resolution
    # ------------------------------------------------------------------

    def _resolve_model_and_profile(self) -> tuple[str, Any]:
        """Return (model_name, profile) after applying pressure checks."""
        mode = self._settings.get_mode()
        model_name = self._settings.get_active_model()
        self._last_low_ram_force_eco_message = None

        if mode == "performance":
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            available_ram = vm.available
            swap_percent = float(swap.percent)
            if (
                available_ram < self._low_ram_force_eco_bytes
                or swap_percent > self._high_swap_force_eco_percent
            ):
                try:
                    eco_model = self._settings.resolve_model("eco")
                except ValueError:
                    eco_model = model_name
                if eco_model != model_name:
                    self._last_low_ram_force_eco_message = (
                        f"Low-memory guard forced eco mode "
                        f"(RAM={available_ram / (1024**3):.2f}GB, swap={swap_percent:.1f}%)."
                    )
                    logger.warning(self._last_low_ram_force_eco_message)
                    model_name = eco_model
                    mode = "eco"

        return model_name, self._settings.get_profile(mode)

    def _resolve_num_gpu(self, mode: str) -> int:
        if self._gpu_soft_fallback:
            return 0
        profile = self._settings.get_profile(mode)
        return profile.num_gpu

    async def _enforce_pressure_guardrail(self, model_name: str) -> None:
        metrics = await self._collect_runtime_metrics()
        if not metrics["pressure_guardrail_triggered"]:
            self._last_pressure_error = None
            return
        error = (
            f"Memory pressure guardrail triggered for '{model_name}' "
            f"(pagefile {metrics['pagefile_usage_percent']:.1f}% > "
            f"{metrics['pagefile_guardrail_percent']:.1f}%). Flush memory and retry."
        )
        self._last_pressure_error = error
        raise OllamaUnavailableError(error)

    # ------------------------------------------------------------------
    # Internal — model readiness / prewarm / GPU preflight
    # ------------------------------------------------------------------

    async def _ensure_model_ready(self) -> tuple[bool, str | None]:
        healthy, payload_or_error = await self._ollama.fetch_tags()
        if not healthy:
            return False, str(payload_or_error)

        assert isinstance(payload_or_error, dict)
        available = self._ollama.extract_model_names(payload_or_error)
        model_name = self._settings.get_active_model()
        if model_name in available:
            return True, None

        logger.warning("Model '%s' missing; attempting pull.", model_name)
        pulled, pull_error = await self._ollama.pull_model(model_name)
        if pulled:
            return True, None

        fallback = self._settings.resolve_model("eco")
        if fallback != model_name:
            logger.warning("Pull failed for '%s'; falling back to '%s'.", model_name, fallback)
            if fallback not in available:
                pulled2, err2 = await self._ollama.pull_model(fallback)
                if not pulled2:
                    return False, f"Could not pull '{model_name}' or fallback '{fallback}': {err2 or pull_error}"
            self._settings.set_mode("eco")
            return True, None

        return False, f"Failed to pull model '{model_name}': {pull_error or 'unknown error'}"

    async def _prewarm_non_blocking(self, model_name: str) -> None:
        try:
            mode = self._settings.get_mode()
            await self._ollama.prewarm(
                model_name,
                self._settings.prewarm_profile,
                num_gpu=self._resolve_num_gpu(mode),
            )
            logger.info("Model '%s' pre-warmed successfully.", model_name)
        except (OllamaUnavailableError, OllamaModelError, OllamaResponseError) as err:
            logger.warning("Prewarm failed for '%s': %s", model_name, err)

    async def _run_gpu_preflight(self) -> None:
        if self._gpu_preflight_checked:
            return
        self._gpu_preflight_checked = True
        if not self._settings.intel_gpu_requested:
            self._gpu_preflight_status = "disabled"
            self._gpu_preflight_message = "Intel iGPU acceleration not requested."
            return

        healthy, ps_payload = await self._ollama.fetch_processes()
        if not healthy:
            self._gpu_soft_fallback = True
            self._gpu_preflight_status = "cpu-fallback"
            self._gpu_preflight_message = f"iGPU preflight: Ollama ps unavailable ({ps_payload}). CPU fallback."
            logger.warning(self._gpu_preflight_message)
            return

        assert isinstance(ps_payload, dict)
        models = ps_payload.get("models") or []
        total_vram = sum(
            m.get("size_vram", 0)
            for m in models
            if isinstance(m, dict) and isinstance(m.get("size_vram"), int)
        )
        total_size = sum(
            m.get("size", 0)
            for m in models
            if isinstance(m, dict) and isinstance(m.get("size"), int)
        )
        if total_size > self._llm_memory_cap_bytes:
            self._gpu_soft_fallback = True
            self._gpu_preflight_status = "cpu-fallback"
            self._gpu_preflight_message = "LLM footprint exceeded 8GB safety cap. CPU fallback."
            logger.warning(self._gpu_preflight_message)
            return
        if total_vram <= 0:
            self._gpu_soft_fallback = True
            self._gpu_preflight_status = "cpu-fallback"
            self._gpu_preflight_message = "iGPU VRAM heuristic failed. CPU fallback."
            logger.warning(self._gpu_preflight_message)
            return

        self._gpu_preflight_status = "active"
        self._gpu_preflight_message = "Intel iGPU acceleration active (Vulkan/oneAPI heuristic passed)."
        logger.info(self._gpu_preflight_message)

    async def _check_ollama_health(self, model_name: str) -> tuple[bool, str | None]:
        healthy, payload_or_error = await self._ollama.fetch_tags()
        if not healthy:
            return False, str(payload_or_error)
        return True, None

    async def _maybe_restore_performance_mode(self) -> None:
        if not self._performance_downgrade_active:
            return
        if time.monotonic() < self._performance_retry_after_monotonic:
            return
        async with self._mode_switch_lock:
            if not self._performance_downgrade_active:
                return
            try:
                perf_model = self._settings.resolve_model("performance")
                eco_model = self._settings.resolve_model("eco")
            except ValueError:
                return
            if self._settings.get_mode() != "eco":
                return
            healthy, payload = await self._ollama.fetch_tags()
            if not healthy:
                self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown
                return
            assert isinstance(payload, dict)
            available = self._ollama.extract_model_names(payload)
            if perf_model not in available:
                self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown
                return
            try:
                await self._ollama.prewarm(perf_model, self._settings.prewarm_profile)
            except (OllamaUnavailableError, OllamaModelError):
                self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown
                return
            self._settings.set_mode("performance")
            self._performance_downgrade_active = False
            logger.info("Performance mode auto-recovered.")

    async def _invoke_host_optimizer(self, model_name: str, mode: str) -> None:
        url = self._settings.host_optimizer_url
        if not url or mode != "performance":
            return
        payload = {
            "action": "pre_inference_flush",
            "mode": mode,
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            timeout = httpx.Timeout(self._settings.host_optimizer_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
            if response.status_code >= 400:
                logger.warning("Host optimizer returned HTTP %s.", response.status_code)
        except (httpx.HTTPError, ValueError) as err:
            logger.warning("Host optimizer callback error: %s", err)

    # ------------------------------------------------------------------
    # Internal — runtime metrics
    # ------------------------------------------------------------------

    async def _collect_runtime_metrics(self) -> dict[str, Any]:
        async with self._metrics_lock:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_io_counters()
            now = time.monotonic()

            disk_io_mbps = 0.0
            if disk is not None:
                total_io = int(disk.read_bytes + disk.write_bytes)
                if self._last_disk_io_bytes is not None and self._last_disk_io_monotonic > 0:
                    elapsed = max(now - self._last_disk_io_monotonic, 1e-6)
                    disk_io_mbps = max(0.0, (total_io - self._last_disk_io_bytes) / elapsed / (1024 * 1024))
                self._last_disk_io_bytes = total_io
                self._last_disk_io_monotonic = now

            cgroup_mem_cur = self._read_int_file("/sys/fs/cgroup/memory.current")
            cgroup_mem_max = self._read_int_file("/sys/fs/cgroup/memory.max")
            cgroup_swap_cur = self._read_int_file("/sys/fs/cgroup/memory.swap.current")
            cgroup_swap_max = self._read_int_file("/sys/fs/cgroup/memory.swap.max")

            container_remaining: int | None = None
            if cgroup_mem_cur is not None and cgroup_mem_max is not None:
                container_remaining = max(0, cgroup_mem_max - cgroup_mem_cur)

            available_ram = int(vm.available)
            if container_remaining is not None:
                available_ram = min(available_ram, container_remaining)

            cgroup_swap_pct: float | None = None
            if cgroup_swap_cur is not None and cgroup_swap_max and cgroup_swap_max > 0:
                cgroup_swap_pct = (cgroup_swap_cur / cgroup_swap_max) * 100.0

            host_swap_pct = float(swap.percent)
            pagefile_pct = cgroup_swap_pct if cgroup_swap_pct is not None else host_swap_pct
            guardrail_triggered = cgroup_swap_pct is not None and cgroup_swap_pct > self._pagefile_guardrail_percent
            proxy_source = "cgroup" if cgroup_swap_pct is not None else "host-informational"

            gpu_load = 0
            healthy_ps, ps_payload = await self._ollama.fetch_processes()
            total_vram = 0
            if healthy_ps and isinstance(ps_payload, dict):
                for m in (ps_payload.get("models") or []):
                    if isinstance(m, dict) and isinstance(m.get("size_vram"), int):
                        total_vram += m["size_vram"]
            if self._settings.intel_gpu_requested and not self._gpu_soft_fallback and total_vram > 0:
                gpu_load = max(1, min(100, int((total_vram / self._llm_memory_cap_bytes) * 100)))

            return {
                "gpu_load_percent": gpu_load,
                "ram_available_bytes": available_ram,
                "disk_io_mbps": round(disk_io_mbps, 2),
                "pagefile_usage_percent": round(pagefile_pct, 2),
                "host_swap_percent": round(host_swap_pct, 2),
                "pagefile_guardrail_percent": self._pagefile_guardrail_percent,
                "pressure_guardrail_triggered": guardrail_triggered,
                "pagefile_proxy_source": proxy_source,
                "telemetry_source": "measured" if healthy_ps else "limited",
            }

    @staticmethod
    def _read_int_file(path: str) -> int | None:
        try:
            with open(path, encoding="utf-8") as f:
                val = f.read().strip()
        except OSError:
            return None
        if not val or val == "max":
            return None
        try:
            return int(val)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Internal — audio/STT helpers
    # ------------------------------------------------------------------

    async def _decode_audio_payload(self, payload: AudioChunkPayload) -> bytes:
        if not payload.is_final:
            raise AudioProcessingError("Streaming partial audio is not supported.")
        if not payload.mime_type.startswith("audio/webm"):
            raise AudioProcessingError(f"Unsupported audio mime type '{payload.mime_type}'.")
        try:
            return await asyncio.to_thread(base64.b64decode, payload.data, validate=True)
        except (ValueError, binascii.Error) as err:
            raise AudioProcessingError("Audio payload is not valid base64.") from err

    @staticmethod
    def _transcode_webm_to_wav(audio_bytes: bytes) -> bytes:
        src = io.BytesIO(audio_bytes)
        dst = io.BytesIO()
        try:
            AudioSegment.from_file(src, format="webm").export(dst, format="wav")
        except (CouldntDecodeError, OSError, ValueError) as err:
            raise AudioProcessingError("Could not convert WebM audio to WAV.") from err
        return dst.getvalue()

    def _transcribe_wav_audio(self, wav_audio: bytes) -> str:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(io.BytesIO(wav_audio)) as source:
                audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language=self._stt_language)
        except sr.UnknownValueError as err:
            raise AudioProcessingError("Could not understand the speech.") from err
        except sr.RequestError as err:
            raise AudioProcessingError(f"STT request failed: {err}") from err
        except (OSError, ValueError) as err:
            raise AudioProcessingError("Could not read the WAV audio.") from err
        cleaned = transcript.strip()
        if not cleaned:
            raise AudioProcessingError("No spoken text detected.")
        return cleaned
