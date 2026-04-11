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
    Performance mode → deep model for complex tasks
    Eco mode         → quick model for lightweight tasks
Both modes run with strong CPU thread allocation.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import re
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
from app.actuators import windows_tools as _windows_tools_module  # noqa: F401
from app.actuators import notification_tools as _notification_tools_module  # noqa: F401
from app.actuators import web_tools as _web_tools_module  # noqa: F401
from app.actuators import spotify_tools as _spotify_tools_module  # noqa: F401
from app.actuators.risk_gate import RiskGate
from app.brain.exceptions import (
    AudioProcessingError,
    OllamaModelError,
    OllamaResponseError,
    OllamaUnavailableError,
)
from app.brain.memory import SessionMemory
from app.brain.memory_guardian import GuardianDecision, MemoryGuardian
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
from app.brain.personal_profile import PersonalProfileStore
from app.brain.streamer import AsyncChunkStreamer
from app.config.settings import (
    DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT,
    DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS,
    DEFAULT_LOW_RAM_FORCE_ECO_BYTES,
    DEFAULT_LLM_MEMORY_CAP_BYTES,
    DEFAULT_MODE_SWITCH_COOLDOWN_SECONDS,
    DEFAULT_PAGEFILE_GUARDRAIL_PERCENT,
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

_MAX_TOOL_CALL_ROUNDS = 2
_MAX_TOOL_RESULT_CHARS = 900

_SYSTEM_INSTRUCTION = (
    "You are Jarvis, a proactive Windows machine agent running natively on the user's PC. "
    "Use a British professional tone — polite, concise, action-oriented. "
    "You have direct access to Windows system controls, Spotify playback, web search, "
    "notifications, clipboard, and the filesystem. "
    "RISK ASSESSMENT: Before calling any tool, evaluate its risk level. "
    "For low-risk tools (get_*, search_*, now_playing, play/pause/volume): proceed directly. "
    "For medium-risk tools (open_application, close_application, write_clipboard, etc.): "
    "briefly warn the user — e.g. 'I'm going to open Notepad now.' — then call the tool. "
    "For high-risk tools (shutdown, restart): ALWAYS state exactly what you are about to do "
    "and why before calling the tool. The system will ask the user for confirmation. "
    "APP LAUNCH PROTOCOL (STRICT): When opening an application: "
    "1. Call the open_application tool. "
    "2. WAIT for the tool result. Do NOT declare success in your text response. "
    "3. ONLY report what the tool result says — if status is 'opened', confirm it. "
    "   If status is 'timeout', 'crashed', or 'not_found', report the failure honestly. "
    "4. NEVER say 'successfully launched' unless the tool explicitly returns status='opened'. "
    "BATTERY & SYSTEM INFO: When asked about battery level, CPU, RAM, or disk usage, "
    "you MUST call the get_system_info tool — never guess or estimate values. "
    "Acknowledge the current mode (Quick/Deep) only if asked. "
    "Always answer in English unless the user specifically requests Thai. "
    "Avoid unnecessary verbosity. Keep output practical and direct."
)


class Orchestrator:
    """Central logic coordinator for the Jarvis AI assistant."""

    def __init__(self) -> None:
        load_project_env()
        self._settings = Settings()
        self._memory = SessionMemory(max_tokens=self._settings.performance_profile.num_ctx)
        self._personal_profile = PersonalProfileStore()
        self._ollama = OllamaClient(self._settings)
        self._tts = TtsHostClient()
        self._tts_voice: str = self._tts.voice_label
        self._stt_language: str = DEFAULT_STT_LANGUAGE
        self._risk_gate = RiskGate()

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
        self._pagefile_guardrail_percent: float = DEFAULT_PAGEFILE_GUARDRAIL_PERCENT
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

    @property
    def risk_gate(self) -> RiskGate:
        """Expose the RiskGate so the WebSocket handler can forward confirmations."""
        return self._risk_gate

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
                await self._log_startup_diagnostics()
                self._ollama_available, self._initialization_error = await self._ensure_model_ready()
                await self._run_gpu_preflight()
                if self._ollama_available:
                    await self._prewarm_non_blocking(self._settings.get_active_model())
                    asyncio.create_task(self._prewarm_tts())
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

    async def _log_startup_diagnostics(self) -> None:
        """Emit high-signal startup diagnostics for runtime endpoints and tools."""
        tool_names = [schema.name for schema in self.tool_schemas]
        logger.info("Startup diagnostics: tools_registered=%d tools=%s", len(tool_names), tool_names)
        logger.info("Startup diagnostics: ollama_base_url=%s", self._settings.ollama_base_url)

        try:
            tts_runtime = await self._tts.get_runtime_status()
            logger.info(
                "Startup diagnostics: tts_mode=%s primary_url=%s fallback_url=%s active_url=%s",
                tts_runtime.get("mode"),
                tts_runtime.get("primary_url"),
                tts_runtime.get("fallback_url"),
                tts_runtime.get("active_url"),
            )
        except RuntimeError as err:
            logger.warning("Startup diagnostics: failed to read TTS runtime status: %s", err)

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
            self._personal_profile.ingest_user_text(user_text)
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

    _TOOL_KEYWORDS = frozenset({
        "open", "launch", "start", "close", "kill", "volume", "mute",
        "screenshot", "clipboard", "lock", "shutdown", "restart",
        "battery", "cpu", "ram", "disk", "system", "spotify", "play",
        "pause", "next", "previous", "search", "find", "notify",
        "reminder", "process", "app", "application", "notepad", "chrome",
        "calculator", "explorer", "discord", "code", "terminal",
        "run", "execute", "show", "tell", "get", "time", "date",
    })

    _IMPERATIVE_VERBS = (
        "open", "launch", "start", "close", "kill", "play", "pause", "resume",
        "stop", "search", "find", "show", "tell", "get", "set", "switch",
        "turn", "increase", "decrease", "mute", "unmute", "lock", "shutdown",
        "restart", "copy", "paste", "run", "execute",
    )

    def _allowed_tool_names(self) -> list[str]:
        return [schema.name for schema in self.tool_schemas]

    def _unknown_tool_correction_prompt(self, attempted_tool_name: str) -> str:
        suggested = self._suggest_tool_names(attempted_tool_name, limit=6)
        suggested_text = ", ".join(suggested) if suggested else "(none)"
        return (
            f"The tool '{attempted_tool_name}' is not registered. "
            "If you still need a tool, output compact JSON only and choose tool_name EXACTLY from likely matches: "
            f"{suggested_text}. "
            "If no tool is needed, answer naturally in plain text."
        )

    def _suggest_tool_names(self, attempted_tool_name: str, limit: int = 6) -> list[str]:
        names = self._allowed_tool_names()
        if not names:
            return []

        attempted = attempted_tool_name.lower().strip()
        attempted_tokens = set(re.findall(r"[a-z0-9_]+", attempted))
        scored: list[tuple[int, str]] = []

        for name in names:
            lowered = name.lower()
            name_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
            overlap = len(attempted_tokens & name_tokens)
            substring_boost = 2 if attempted and (attempted in lowered or lowered in attempted) else 0
            score = overlap + substring_boost
            if score > 0:
                scored.append((score, name))

        if not scored:
            return names[:limit]

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[:limit]]

    @classmethod
    def _message_needs_tools(cls, user_text: str) -> bool:
        """Quick heuristic: does this message likely need a tool call?"""
        words = set(re.findall(r"[a-z0-9_]+", user_text.lower()))
        return bool(words & cls._TOOL_KEYWORDS)

    @classmethod
    def _is_direct_command(cls, user_text: str) -> bool:
        normalized = user_text.strip().lower()
        if not normalized:
            return False
        normalized = re.sub(r"^(please|jarvis|hey jarvis|can you|could you)\s+", "", normalized)
        if normalized.endswith("?"):
            return False
        return any(normalized.startswith(f"{verb} ") for verb in cls._IMPERATIVE_VERBS)

    @staticmethod
    def _latest_user_text(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def _infer_timezone_from_text(user_text: str) -> str:
        normalized = user_text.lower()
        if "sydney" in normalized or "australia" in normalized:
            return "Australia/Sydney"
        if "thailand" in normalized or "bangkok" in normalized or "thai" in normalized:
            return "Asia/Bangkok"
        if "tokyo" in normalized or "japan" in normalized:
            return "Asia/Tokyo"
        if "london" in normalized or "uk" in normalized or "britain" in normalized:
            return "Europe/London"
        if "new york" in normalized or "usa" in normalized or "us" in normalized:
            return "America/New_York"
        return "UTC"

    def _deterministic_tool_fallback(self, user_text: str) -> tuple[str, dict[str, Any]] | None:
        normalized = user_text.lower().strip()

        if re.search(r"\b(open|launch|start)\b.*\bspotify\b", normalized):
            return "open_application", {"app_name": "spotify"}

        if re.search(r"\b(open|launch|start)\b", normalized):
            app_aliases = {
                "notepad": "notepad",
                "calculator": "calculator",
                "chrome": "chrome",
                "discord": "discord",
                "explorer": "explorer",
                "terminal": "terminal",
                "code": "code",
            }
            for alias, app_name in app_aliases.items():
                if alias in normalized:
                    return "open_application", {"app_name": app_name}

        if re.search(r"\b(time|date|datetime)\b", normalized):
            timezone_name = self._infer_timezone_from_text(normalized)
            return "get_current_datetime", {"timezone_name": timezone_name}

        if "battery" in normalized:
            return "get_system_info", {"type": "battery"}

        if re.search(r"\b(cpu|processor)\b", normalized):
            return "get_system_info", {"type": "cpu"}

        if re.search(r"\b(ram|memory)\b", normalized):
            return "get_system_info", {"type": "ram"}

        if re.search(r"\b(disk|storage|space)\b", normalized):
            return "get_system_info", {"type": "disk"}

        if re.search(r"\b(system status|system info|status)\b", normalized):
            return "get_system_info", {}

        if re.search(r"\b(play|resume)\b.*\bspotify\b", normalized):
            return "play_spotify", {}

        if re.search(r"\bpause\b.*\bspotify\b", normalized):
            return "pause_spotify", {}

        if re.search(r"\b(next|skip)\b.*\bspotify\b", normalized):
            return "next_track", {}

        if re.search(r"\b(previous|back)\b.*\bspotify\b", normalized):
            return "previous_track", {}

        return None

    @staticmethod
    def _inject_system_context(messages: list[dict[str, str]], context_text: str) -> list[dict[str, str]]:
        context_message = {"role": "system", "content": context_text}
        if messages and messages[0].get("role") == "system":
            return [messages[0], context_message, *messages[1:]]
        return [context_message, *messages]

    def _build_personalization_prompt(self, user_text: str, *, strict_command_mode: bool) -> str:
        profile = self._personal_profile.snapshot()
        name = str(profile.get("name") or "").strip()
        goals = "; ".join(str(x) for x in (profile.get("goals") or [])[:4]) or "none yet"
        likes = "; ".join(str(x) for x in (profile.get("likes") or [])[:4]) or "none yet"
        notes = "; ".join(str(x) for x in (profile.get("notes") or [])[:3]) or "none"

        style_pref = str(profile.get("response_style") or "adaptive")
        normalized = user_text.lower()
        if style_pref == "short":
            turn_style = "short: 1-2 direct sentences"
        elif style_pref == "detailed":
            turn_style = "detailed: concise summary then practical steps"
        elif any(token in normalized for token in ("why", "how", "plan", "compare", "strategy", "explain")):
            turn_style = "detailed: concise summary then practical steps"
        elif strict_command_mode:
            turn_style = "short: execute first, then confirm outcome"
        else:
            turn_style = "medium: practical and human, 2-5 sentences"

        command_priority = bool(profile.get("command_priority", True))
        command_line = (
            "For explicit commands, prioritize execution and provide outcome-backed confirmation."
            if command_priority
            else "Balance command execution with clarification when intent is ambiguous."
        )

        name_line = f"User preferred name: {name}." if name else "User preferred name: unknown."
        return (
            "Personalization context for this user:\n"
            f"- {name_line}\n"
            f"- Known goals: {goals}.\n"
            f"- Known likes: {likes}.\n"
            f"- User notes: {notes}.\n"
            f"- Response style for this turn: {turn_style}.\n"
            f"- {command_line}\n"
            "Do not fabricate personal facts. If unknown, stay honest and ask briefly when needed."
        )

    @staticmethod
    def _should_auto_switch_low_risk(decision: GuardianDecision, risk_level: str) -> bool:
        _ = decision
        _ = risk_level
        return False

    def _build_tool_protocol_prompt(self) -> str:
        lines: list[str] = []
        for schema in self.tool_schemas:
            props = schema.parameters.get("properties", {})
            arg_names = ",".join(str(k) for k in props.keys())
            lines.append(f"- {schema.name}({arg_names})")

        tools_text = "\n".join(lines) if lines else "- (no tools available)"
        allowed_names = ", ".join(self._allowed_tool_names()) if self.tool_schemas else "(none)"
        return (
            "Tool protocol for this turn:\n"
            "If a tool is required, output only compact JSON with no markdown: "
            '{"t":"tool","n":"tool_name","a":{}}.\n'
            "tool_name must match EXACTLY one registered tool name (no aliases).\n"
            f"Allowed tool names: {allowed_names}\n"
            "If no tool is required, answer naturally in plain text.\n"
            "Available tools:\n"
            f"{tools_text}"
        )

    def _inject_tool_protocol(self, messages: list[dict[str, str]], *, force: bool = False) -> list[dict[str, str]]:
        # Only inject tool protocol if the latest user message likely needs a tool
        # This avoids adding ~200+ tokens of overhead to every simple chat message
        if not force:
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if not self._message_needs_tools(last_user_msg):
                return messages

        protocol_message = {"role": "system", "content": self._build_tool_protocol_prompt()}
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

        # Canonical compact format: {"t":"tool","n":"...","a":{...}}
        # Also accept common model variants for resilience.
        declared_type = candidate.get("t") or candidate.get("type")
        if declared_type is not None and declared_type != "tool":
            return None

        tool_name = (
            candidate.get("n")
            or candidate.get("tool_name")
            or candidate.get("name")
            or candidate.get("function")
            or candidate.get("tool")
        )
        tool_args = (
            candidate.get("a")
            or candidate.get("args")
            or candidate.get("arguments")
            or candidate.get("parameters")
            or {}
        )

        if isinstance(tool_name, dict):
            # OpenAI-style: {"function": {"name": "...", "arguments": {...}}}
            nested_name = tool_name.get("name")
            nested_args = tool_name.get("arguments") or {}
            tool_name = nested_name
            if isinstance(nested_args, dict):
                tool_args = nested_args

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

    @staticmethod
    def _tool_outcome_prompt(tool_name: str, tool_result: dict[str, Any]) -> str:
        status = str(tool_result.get("status", "unknown"))
        verified = tool_result.get("verified")
        evidence = str(tool_result.get("evidence") or "No evidence provided.")
        result_text = Orchestrator._format_tool_result(tool_result)
        return (
            f"Tool outcome for {tool_name}: status={status}, verified={verified}, evidence={evidence}. "
            f"Full result: {result_text}. "
            "When answering: never claim success unless status indicates success and verified is true. "
            "If verified is false or status is partial/unverified/failed/timeout/not_found/crashed, use cautious wording and explain what failed. "
            "Answer the original request naturally. Do not output JSON."
        )

    @staticmethod
    def _tool_only_fallback_text(tool_name: str, tool_result: dict[str, Any]) -> str:
        """Build a deterministic response when Ollama is unavailable after a tool call."""
        status = str(tool_result.get("status") or "unknown").lower()
        verified = tool_result.get("verified")
        evidence = str(tool_result.get("evidence") or "No verification evidence.")

        if tool_name == "get_current_datetime":
            timezone_name = str(tool_result.get("timezone") or "UTC")
            human_readable = str(tool_result.get("human_readable") or "").strip()
            iso_value = str(tool_result.get("iso_datetime") or "").strip()
            if human_readable:
                return f"The current date and time in {timezone_name} is {human_readable}."
            if iso_value:
                return f"The current date and time in {timezone_name} is {iso_value}."

        if tool_name == "open_application":
            app_name = str(tool_result.get("app_name") or "the application")
            if status == "opened":
                return f"I opened {app_name}."
            if status == "opened_no_window":
                return (
                    f"I started {app_name}, but I could not confirm a visible window. "
                    f"Evidence: {evidence}"
                )

        failed_statuses = {"failed", "error", "timeout", "not_found", "crashed", "blocked"}
        uncertain_statuses = {"partial", "unverified", "opened_no_window", "no_windows"}

        if status in failed_statuses:
            return f"I could not complete {tool_name}. Status: {status}. Evidence: {evidence}"

        if status in uncertain_statuses or verified is False:
            return (
                f"I attempted {tool_name}, but I cannot confirm full success yet. "
                f"Status: {status}. Evidence: {evidence}"
            )

        return f"I completed {tool_name}. Status: {status}."

    @staticmethod
    def _enforce_claim_integrity(tool_name: str, tool_result: dict[str, Any], candidate_text: str) -> str:
        status = str(tool_result.get("status", "")).lower()
        verified = tool_result.get("verified")
        evidence = str(tool_result.get("evidence") or "No verification evidence.")

        positive_claim = bool(re.search(r"\b(success|successfully|done|completed|opened|launched|playing|paused|transferred)\b", candidate_text, re.IGNORECASE))
        failed_statuses = {"failed", "error", "timeout", "not_found", "crashed", "blocked"}
        uncertain_statuses = {"partial", "unverified", "opened_no_window", "no_windows"}

        if status in failed_statuses:
            return (
                f"I could not complete {tool_name}. Status: {status}. "
                f"Evidence: {evidence}"
            )

        if status in uncertain_statuses or verified is False:
            if positive_claim:
                return (
                    f"I attempted {tool_name}, but I cannot confirm full success yet. "
                    f"Status: {status or 'unknown'}. Evidence: {evidence}"
                )

        return candidate_text

    @staticmethod
    def _summarize_tool_outcome(tool_name: str | None, tool_result: dict[str, Any] | None) -> dict[str, Any] | None:
        if tool_name is None or tool_result is None:
            return None
        return {
            "tool_name": tool_name,
            "status": tool_result.get("status"),
            "verified": tool_result.get("verified"),
            "evidence": tool_result.get("evidence"),
        }

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
            mode_label = "Deep" if decision.suggested_mode == "performance" else "Quick"
            return (
                f"I need your confirmation to switch to {mode_label} mode "
                f"before I run '{decision.tool_name}'."
            )
        return decision.reason + " Please free memory and retry."

    async def _generate_with_tool_loop(
        self,
        *,
        messages: list[dict[str, str]],
        model_name: str,
        profile: Any,
    ) -> str:
        working_messages = [dict(message) for message in messages]
        request_text = self._latest_user_text(working_messages)
        strict_command_mode = self._is_direct_command(request_text)
        personalization_prompt = self._build_personalization_prompt(
            request_text,
            strict_command_mode=strict_command_mode,
        )
        unknown_tool_retry_used = False
        deterministic_fallback_used = False
        last_tool_name: str | None = None
        last_tool_result: dict[str, Any] | None = None
        current_model_name = model_name
        current_profile = profile
        pending_tool_call = self._deterministic_tool_fallback(request_text)
        if pending_tool_call is not None:
            logger.info("[Tool fallback] Command-first dispatch: %s(%s)", pending_tool_call[0], pending_tool_call[1])
            deterministic_fallback_used = True

        for _ in range(_MAX_TOOL_CALL_ROUNDS + 1):
            response_text = ""
            if pending_tool_call is not None:
                parsed = pending_tool_call
                pending_tool_call = None
            else:
                force_protocol = strict_command_mode or unknown_tool_retry_used or (last_tool_result is not None)
                protocol_messages = self._inject_tool_protocol(working_messages, force=force_protocol)
                protocol_messages = self._inject_system_context(protocol_messages, personalization_prompt)
                try:
                    response_text = await self._ollama.generate(
                        protocol_messages,
                        current_model_name,
                        current_profile,
                        num_gpu=self._resolve_num_gpu(current_profile.mode),
                    )
                except (OllamaUnavailableError, OllamaModelError, OllamaResponseError, RuntimeError) as err:
                    if last_tool_name is not None and last_tool_result is not None:
                        logger.warning(
                            "[Tool loop] Ollama unavailable after tool '%s'; using deterministic fallback: %s",
                            last_tool_name,
                            err,
                        )
                        self._ollama_available = False
                        self._initialization_error = str(err)
                        return self._tool_only_fallback_text(last_tool_name, last_tool_result)
                    raise
                parsed = self._extract_tool_call(response_text)
                if parsed is None and not deterministic_fallback_used:
                    fallback = self._deterministic_tool_fallback(request_text)
                    if fallback is not None:
                        parsed = fallback
                        deterministic_fallback_used = True
                        logger.info("[Tool fallback] Deterministic dispatch: %s(%s)", parsed[0], parsed[1])

            if parsed is None:
                final_text = response_text.strip()
                if last_tool_name is not None and last_tool_result is not None:
                    return self._enforce_claim_integrity(last_tool_name, last_tool_result, final_text)
                if strict_command_mode:
                    return (
                        "I could not map that command to an available tool yet. "
                        "Please restate it with a clear action and target (for example: open Spotify, set volume to 40, or search web for ...)."
                    )
                return final_text

            tool_name, tool_args = parsed
            if not tool_registry.has_tool(tool_name):
                working_messages.append(
                    {"role": "assistant", "content": f"Tool '{tool_name}' is unavailable."}
                )
                if unknown_tool_retry_used:
                    return (
                        "I could not map that action to an available tool. "
                        "Please rephrase with a specific action, for example: open_application for apps or play_spotify for playback."
                    )
                working_messages.append({
                    "role": "user",
                    "content": self._unknown_tool_correction_prompt(tool_name),
                })
                unknown_tool_retry_used = True
                continue

            metadata = tool_registry.get_metadata(tool_name)
            risk_level = str(metadata.get("risk_level") or "low").lower()

            decision = await self._assess_tool_call(
                tool_name=tool_name,
                active_mode=self._settings.get_mode(),
            )
            if not decision.allowed:
                return self._guardian_message(decision)

            # Risk gate: medium/high-risk tools pause for user confirmation.
            # No send_event_fn in the non-WS tool loop — auto-approve low risk,
            # warn for medium/high via the response text instead.
            if risk_level in ("medium", "high"):
                # In the non-streaming path, we cannot pause for WS confirmation;
                # the LLM has already warned the user via its text (system prompt).
                # Log and proceed — WS path handles interactive confirmation.
                logger.info(
                    "Tool '%s' risk_level='%s' — proceeding (non-interactive path).",
                    tool_name, risk_level,
                )

            try:
                tool_result = await tool_registry.execute(tool_name, tool_args)
            except (KeyError, ValueError, RuntimeError) as err:
                tool_result = {
                    "status": "failed",
                    "verified": False,
                    "evidence": f"Tool execution failed: {err}",
                    "error": str(err),
                }
                working_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Tool '{tool_name}' failed with error: {err}",
                    }
                )
            last_tool_name = tool_name
            last_tool_result = tool_result
            working_messages.append(
                {
                    "role": "user",
                    "content": self._tool_outcome_prompt(tool_name, tool_result),
                }
            )

        final_response = "I could not complete the request safely within the local tool-call limit."
        return final_response

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

        model_name, _ = self._resolve_model_and_profile()
        final_text_from_callback: dict[str, str] = {"text": ""}

        async def on_sentence_audio_noop(
            _sentence_index: int,
            _sentence_text: str,
            _audio_payload: AssistantAudioPayload | None,
            _error_message: str | None,
        ) -> None:
            return None

        async def on_final_text_capture(final_text: str, _last_tool_outcome: dict[str, Any] | None) -> None:
            final_text_from_callback["text"] = final_text

        try:
            await on_stream_start(model_name)
            response_text = await self.handle_http_chat_streaming(
                session_id=session_id,
                user_text=user_text,
                auto_speak=False,
                num_ctx_override=None,
                on_text_chunk=on_stream_delta,
                on_sentence_audio=on_sentence_audio_noop,
                on_final_text=on_final_text_capture,
            )
            if final_text_from_callback["text"]:
                response_text = final_text_from_callback["text"]
            await on_stream_end(response_text)
        except (OllamaUnavailableError, OllamaModelError, OllamaResponseError, RuntimeError) as err:
            return BrainResponse(text=str(err), tool_schemas=self.tool_schemas)

        return await self._attach_audio(BrainResponse(text=response_text, tool_schemas=self.tool_schemas))

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
        on_final_text: Callable[[str, dict[str, Any] | None], Awaitable[None]] | None = None,
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
            self._personal_profile.ingest_user_text(user_text)
            self._memory.append_user(session_id, user_text)
            model_name, profile = self._resolve_model_and_profile()
            http_num_thread = HTTP_CHAT_NUM_THREAD.get(profile.mode, profile.num_thread)
            effective_profile = PERFORMANCE_PROFILE.__class__(
                mode=profile.mode,
                num_ctx=profile.num_ctx,
                num_thread=http_num_thread,
                num_gpu=profile.num_gpu,
                description=profile.description,
            )

            try:
                # Build messages with tool protocol injected
                session_messages = [
                    dict(message) for message in (self._memory.get_messages(session_id) or [])
                ]
                strict_command_mode = self._is_direct_command(user_text)
                personalization_prompt = self._build_personalization_prompt(
                    user_text,
                    strict_command_mode=strict_command_mode,
                )
                pending_tool_call = self._deterministic_tool_fallback(user_text)
                if pending_tool_call is not None:
                    logger.info(
                        "[HTTP streaming] Command-first dispatch: %s(%s)",
                        pending_tool_call[0],
                        pending_tool_call[1],
                    )
                    final_text = ""
                else:
                    protocol_messages = self._inject_tool_protocol(session_messages, force=strict_command_mode)
                    protocol_messages = self._inject_system_context(protocol_messages, personalization_prompt)

                    final_text = await self._ollama.generate_streaming(
                        protocol_messages,
                        model_name,
                        effective_profile,
                        on_delta=streamer.on_delta,
                        num_gpu=self._resolve_num_gpu(profile.mode),
                        num_ctx_override=num_ctx_override,
                    )

                    # Flush trailing text and dispatch trailing TTS sentence
                    trailing = streamer.get_trailing_sentence_buffer()
                    await streamer.flush_final(trailing_buffer=trailing)

                    final_text = final_text.strip()

                # ── Tool execution loop ──────────────────────────────────────
                # The streaming path may produce a tool-call JSON. If so,
                # execute the tool and generate a follow-up streaming response.
                working_messages = [dict(m) for m in session_messages]
                if final_text:
                    working_messages.append({"role": "assistant", "content": final_text})
                unknown_tool_retry_used = False
                deterministic_fallback_used = pending_tool_call is not None
                last_tool_name: str | None = None
                last_tool_result: dict[str, Any] | None = None

                for _tool_round in range(_MAX_TOOL_CALL_ROUNDS):
                    if pending_tool_call is not None:
                        parsed = pending_tool_call
                        pending_tool_call = None
                    else:
                        parsed = self._extract_tool_call(final_text)

                    if parsed is None and not deterministic_fallback_used:
                        fallback = self._deterministic_tool_fallback(user_text)
                        if fallback is not None:
                            parsed = fallback
                            deterministic_fallback_used = True
                            logger.info("[HTTP streaming] Deterministic fallback tool call: %s(%s)", parsed[0], parsed[1])

                    if parsed is None:
                        if strict_command_mode and last_tool_name is None:
                            final_text = (
                                "I could not map that command to an available tool yet. "
                                "Please restate it with a clear action and target."
                            )
                        break  # No tool call — we're done

                    tool_name, tool_args = parsed
                    logger.info("[HTTP streaming] Tool call detected: %s(%s)", tool_name, tool_args)

                    if not tool_registry.has_tool(tool_name):
                        working_messages.append(
                            {"role": "assistant", "content": f"Tool '{tool_name}' is unavailable."}
                        )
                        if unknown_tool_retry_used:
                            final_text = (
                                "I could not map that action to an available tool. "
                                "Please rephrase with a specific supported command."
                            )
                            break
                        working_messages.append({
                            "role": "user",
                            "content": self._unknown_tool_correction_prompt(tool_name),
                        })
                        unknown_tool_retry_used = True
                    else:
                        metadata = tool_registry.get_metadata(tool_name)
                        risk_level = str(metadata.get("risk_level") or "low").lower()

                        # Memory guardian check
                        decision = await self._assess_tool_call(
                            tool_name=tool_name,
                            active_mode=self._settings.get_mode(),
                        )
                        if not decision.allowed:
                            final_text = self._guardian_message(decision)
                            break

                        # Risk gate logging
                        if risk_level in ("medium", "high"):
                            logger.info(
                                "Tool '%s' risk_level='%s' — proceeding (HTTP streaming path).",
                                tool_name, risk_level,
                            )

                        # Execute the tool
                        try:
                            tool_result = await tool_registry.execute(tool_name, tool_args)
                            result_text = self._format_tool_result(tool_result)
                            logger.info("[HTTP streaming] Tool '%s' result: %s", tool_name, result_text[:200])
                        except (KeyError, ValueError, RuntimeError) as err:
                            tool_result = {
                                "status": "failed",
                                "verified": False,
                                "evidence": f"Tool execution failed: {err}",
                                "error": str(err),
                            }
                            logger.warning("[HTTP streaming] Tool '%s' failed: %s", tool_name, err)

                        last_tool_name = tool_name
                        last_tool_result = tool_result

                        working_messages.append(
                            {"role": "user", "content": self._tool_outcome_prompt(tool_name, tool_result)}
                        )

                    # Generate follow-up streaming response with tool result
                    protocol_messages = self._inject_tool_protocol(working_messages, force=True)
                    protocol_messages = self._inject_system_context(protocol_messages, personalization_prompt)
                    try:
                        final_text = await self._ollama.generate_streaming(
                            protocol_messages,
                            model_name,
                            effective_profile,
                            on_delta=streamer.on_delta,
                            num_gpu=self._resolve_num_gpu(effective_profile.mode),
                            num_ctx_override=num_ctx_override,
                        )
                        trailing = streamer.get_trailing_sentence_buffer()
                        await streamer.flush_final(trailing_buffer=trailing)
                        final_text = final_text.strip()
                        if last_tool_name is not None and last_tool_result is not None:
                            final_text = self._enforce_claim_integrity(last_tool_name, last_tool_result, final_text)
                    except (OllamaUnavailableError, OllamaModelError, OllamaResponseError, RuntimeError) as err:
                        if last_tool_name is not None and last_tool_result is not None:
                            logger.warning(
                                "[HTTP streaming] Ollama unavailable after tool '%s'; using deterministic fallback: %s",
                                last_tool_name,
                                err,
                            )
                            self._ollama_available = False
                            self._initialization_error = str(err)
                            final_text = self._tool_only_fallback_text(last_tool_name, last_tool_result)
                            break
                        raise
                    working_messages.append({"role": "assistant", "content": final_text})

                self._memory.append_assistant(session_id, final_text)
                self._ollama_available = True
                self._initialization_error = None
                final_tool_outcome = self._summarize_tool_outcome(last_tool_name, last_tool_result)

                if on_final_text is not None:
                    try:
                        await on_final_text(final_text, final_tool_outcome)
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
                "message": (
                    "Switched to Deep mode for complex tasks."
                    if mode == "performance"
                    else "Switched to Quick mode for lighter tasks."
                ),
                "mode_switch_wait_ms": int(wait_seconds * 1000),
                "prewarm_attempted": prewarm,
                "prewarm_warning": prewarm_warning,
                "prewarm_started": False,
            }

    # ------------------------------------------------------------------
    # Internal — model / pressure resolution
    # ------------------------------------------------------------------

    def _resolve_model_and_profile(self) -> tuple[str, Any]:
        """Return (model_name, profile) for the currently selected task mode."""
        mode = self._settings.get_mode()
        model_name = self._settings.get_active_model()
        self._last_low_ram_force_eco_message = None

        return model_name, self._settings.get_profile(mode)

    def _resolve_num_gpu(self, mode: str) -> int:
        if not self._settings.intel_gpu_requested or self._gpu_soft_fallback:
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

    async def _prewarm_tts(self) -> None:
        """Warm up TTS by sending a short silent request."""
        try:
            await self._tts.synthesize(".")
            logger.info("TTS pre-warmed successfully.")
        except TtsEngineError as err:
            logger.warning("TTS prewarm failed: %s", err)

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
        if not url:
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
