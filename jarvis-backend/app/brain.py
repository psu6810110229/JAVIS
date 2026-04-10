from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import psutil
import speech_recognition as sr
from pydantic import BaseModel, Field, ValidationError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .config import SystemConfig, load_project_env
from .tts_engine import KokoroTtsEngine, TtsEngineError

logger = logging.getLogger(__name__)
DEFAULT_OLLAMA_MODEL_NAME = "scb10x/llama3.1-typhoon2-8b-instruct"
DEFAULT_FALLBACK_OLLAMA_MODEL_NAME = "scb10x/typhoon2.5-qwen3-4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300.0
DEFAULT_OLLAMA_PULL_TIMEOUT_SECONDS = 1800.0
DEFAULT_OLLAMA_TEMPERATURE = 0.7
DEFAULT_ECO_NUM_CTX = 1536
DEFAULT_PERFORMANCE_NUM_CTX = 1024
FALLBACK_OLLAMA_NUM_CTX = 512
DEFAULT_PREWARM_NUM_CTX = 64
DEFAULT_OLLAMA_KEEP_ALIVE = -1
DEFAULT_ECO_NUM_THREAD = 2
DEFAULT_PERFORMANCE_NUM_THREAD = 10
DEFAULT_HTTP_CHAT_NUM_THREAD_PERFORMANCE = 8
DEFAULT_HTTP_CHAT_NUM_THREAD_ECO = 4
DEFAULT_ECO_NUM_GPU = 0
DEFAULT_PERFORMANCE_NUM_GPU = 0
DEFAULT_OLLAMA_NUM_GPU_FALLBACK = 0
DEFAULT_PAGEFILE_GUARDRAIL_PERCENT = 10.0
DEFAULT_EOF_RETRY_DELAY_SECONDS = 0.5
DEFAULT_KV_CACHE_TYPE = "q4_0"
DEFAULT_NUM_BATCH_REQUESTED = 512
DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS = 0.35
DEFAULT_LLM_MEMORY_CAP_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_LOW_RAM_FORCE_ECO_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT = 35.0
DEFAULT_MODE_SWITCH_COOLDOWN_SECONDS = 0.5
DEFAULT_PERFORMANCE_RETRY_COOLDOWN_SECONDS = 300.0
DEFAULT_SENTENCE_TTS_CONCURRENCY = 2
DEFAULT_STT_LANGUAGE = "th-TH"
DEFAULT_STREAM_TEXT_FLUSH_SECONDS = 0.10
DEFAULT_STREAM_TEXT_FLUSH_MIN_CHARS = 64


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
        self._intel_gpu_requested = os.getenv("OLLAMA_INTEL_GPU", "0") == "1"
        self._gpu_soft_fallback = False
        self._gpu_preflight_checked = False
        self._gpu_preflight_status = "disabled"
        self._gpu_preflight_message = "Intel iGPU acceleration not requested."
        self._mode_switch_cooldown_seconds = DEFAULT_MODE_SWITCH_COOLDOWN_SECONDS
        self._last_mode_switch_monotonic = 0.0
        self._llm_memory_cap_bytes = DEFAULT_LLM_MEMORY_CAP_BYTES
        self._low_ram_force_eco_threshold_bytes = DEFAULT_LOW_RAM_FORCE_ECO_BYTES
        self._high_swap_force_eco_percent = DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT
        self._last_low_ram_force_eco_message: str | None = None
        self._performance_retry_cooldown_seconds = DEFAULT_PERFORMANCE_RETRY_COOLDOWN_SECONDS
        self._performance_downgrade_active = False
        self._last_performance_downgrade_at: str | None = None
        self._last_performance_downgrade_reason: str | None = None
        self._performance_retry_after_monotonic = 0.0
        self._pagefile_guardrail_percent = DEFAULT_PAGEFILE_GUARDRAIL_PERCENT
        self._last_pressure_error: str | None = None
        self._last_disk_io_bytes: int | None = None
        self._last_disk_io_monotonic = 0.0
        self._metrics_lock = asyncio.Lock()
        self._kv_cache_supported = True
        self._kv_cache_warning: str | None = None
        self._num_batch_requested = DEFAULT_NUM_BATCH_REQUESTED
        self._num_batch_supported = False
        self._num_batch_warning = "Ollama /api/chat does not expose num_batch controls in request options."
        self._host_optimizer_url = os.getenv("JARVIS_HOST_OPTIMIZER_URL", "").strip()
        timeout_raw = os.getenv("JARVIS_HOST_OPTIMIZER_TIMEOUT_SECONDS", str(DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS))
        try:
            self._host_optimizer_timeout_seconds = max(0.1, float(timeout_raw))
        except ValueError:
            self._host_optimizer_timeout_seconds = DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS
        self._host_optimizer_last_status = "disabled" if not self._host_optimizer_url else "idle"
        self._host_optimizer_last_message = (
            "Host optimizer callback is disabled."
            if not self._host_optimizer_url
            else f"Ready to call host optimizer at {self._host_optimizer_url}."
        )

    def _active_model_name(self) -> str:
        current_model = self._system_config.get_active_model()
        self._model_name = current_model
        return current_model

    def _active_mode(self) -> str:
        return self._system_config.get_mode()

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
                await self._run_intel_gpu_preflight()
                if self._ollama_available:
                    asyncio.create_task(self.prewarm_model_non_blocking(self._active_model_name()))
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

    def _resolve_mode_for_model(self, model_name: str) -> str:
        try:
            if model_name == self._system_config.resolve_model("performance"):
                return "performance"
        except ValueError:
            pass
        return "eco"

    def _resolve_num_ctx(self, mode: str, use_fallback_profile: bool, num_ctx_override: int | None) -> int:
        if isinstance(num_ctx_override, int) and num_ctx_override > 0:
            return num_ctx_override
        if use_fallback_profile:
            return FALLBACK_OLLAMA_NUM_CTX
        if mode == "performance":
            return DEFAULT_PERFORMANCE_NUM_CTX
        return DEFAULT_ECO_NUM_CTX

    def _resolve_num_thread(self, mode: str, num_thread_override: int | None) -> int:
        if isinstance(num_thread_override, int) and num_thread_override > 0:
            return num_thread_override
        if mode == "performance":
            return DEFAULT_PERFORMANCE_NUM_THREAD
        return DEFAULT_ECO_NUM_THREAD

    def _resolve_http_chat_num_thread(self, mode: str) -> int:
        if mode == "performance":
            return DEFAULT_HTTP_CHAT_NUM_THREAD_PERFORMANCE
        return DEFAULT_HTTP_CHAT_NUM_THREAD_ECO

    async def _resolve_chat_model_for_current_pressure(self, requested_model: str) -> tuple[str, str]:
        mode = self._resolve_mode_for_model(requested_model)
        self._last_low_ram_force_eco_message = None

        if mode != "performance":
            return requested_model, mode

        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        available_ram_bytes = virtual_memory.available
        host_swap_percent = float(swap_memory.percent)
        if (
            available_ram_bytes >= self._low_ram_force_eco_threshold_bytes
            and host_swap_percent <= self._high_swap_force_eco_percent
        ):
            return requested_model, mode

        try:
            eco_model = self._system_config.resolve_model("eco")
        except ValueError:
            return requested_model, mode

        if eco_model == requested_model:
            return requested_model, mode

        self._last_low_ram_force_eco_message = (
            "Low-memory latency guard switched this request to eco model "
            f"because available RAM is {available_ram_bytes / (1024**3):.2f}GB "
            f"(threshold {self._low_ram_force_eco_threshold_bytes / (1024**3):.2f}GB) "
            f"and host swap is {host_swap_percent:.1f}% "
            f"(threshold {self._high_swap_force_eco_percent:.1f}%)."
        )
        logger.warning(self._last_low_ram_force_eco_message)
        return eco_model, "eco"

    def _resolve_num_gpu(self, mode: str, use_fallback_profile: bool = False) -> int:
        if not self._intel_gpu_requested or self._gpu_soft_fallback:
            return 0
        base_gpu = DEFAULT_PERFORMANCE_NUM_GPU if mode == "performance" else DEFAULT_ECO_NUM_GPU
        if use_fallback_profile:
            return min(base_gpu, DEFAULT_OLLAMA_NUM_GPU_FALLBACK)
        return base_gpu

    def _resolve_emergency_model_fallback(self, current_model: str) -> str | None:
        try:
            eco_model = self._system_config.resolve_model("eco")
        except ValueError:
            return None

        if eco_model == current_model:
            return None

        return eco_model

    @staticmethod
    def _is_kv_cache_option_unsupported(detail: str) -> bool:
        normalized = detail.lower()
        mentions_kv = "kv_cache" in normalized or "kv_type" in normalized
        mentions_option_error = "unsupported" in normalized or "unknown" in normalized or "invalid option" in normalized
        return mentions_kv and mentions_option_error

    @staticmethod
    def _is_unexpected_eof_error(detail: str) -> bool:
        normalized = detail.lower()
        return "unexpected eof" in normalized or "incomplete" in normalized or "connection reset" in normalized

    def _build_ollama_options(
        self,
        *,
        num_ctx: int,
        num_thread: int,
        num_gpu: int,
        temperature: float,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_thread": num_thread,
            "num_gpu": num_gpu,
        }
        if self._kv_cache_supported:
            # Best-effort request for 4-bit KV cache; unsupported runtimes are handled gracefully.
            options["kv_cache_type"] = DEFAULT_KV_CACHE_TYPE
        return options

    async def _invoke_host_optimizer_if_needed(self, mode: str, model_name: str) -> None:
        if mode != "performance":
            return
        if not self._host_optimizer_url:
            self._host_optimizer_last_status = "disabled"
            self._host_optimizer_last_message = "Host optimizer callback is disabled."
            return

        payload = {
            "action": "pre_inference_flush",
            "mode": mode,
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            timeout = httpx.Timeout(self._host_optimizer_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self._host_optimizer_url, json=payload)
            if response.status_code >= 400:
                self._host_optimizer_last_status = "degraded"
                self._host_optimizer_last_message = (
                    f"Host optimizer callback failed with HTTP {response.status_code}."
                )
                return

            self._host_optimizer_last_status = "applied"
            self._host_optimizer_last_message = "Host optimizer callback completed."
        except (httpx.HTTPError, ValueError) as error:
            self._host_optimizer_last_status = "degraded"
            self._host_optimizer_last_message = f"Host optimizer callback error: {error}"

    def _activate_emergency_eco_mode(self, eco_model: str) -> None:
        try:
            self._system_config.set_mode("eco")
        except ValueError:
            return

        self._model_name = eco_model
        self._performance_downgrade_active = True
        self._last_performance_downgrade_at = datetime.utcnow().isoformat() + "Z"
        self._last_performance_downgrade_reason = "llama runner process terminated under performance load"
        self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown_seconds
        logger.warning(
            "Emergency fallback activated: switching active mode to eco model '%s' to prevent repeated runner crashes.",
            eco_model,
        )

    async def _maybe_restore_performance_mode(self) -> None:
        if not self._performance_downgrade_active:
            return

        if time.monotonic() < self._performance_retry_after_monotonic:
            return

        async with self._model_switch_lock:
            if not self._performance_downgrade_active:
                return
            if time.monotonic() < self._performance_retry_after_monotonic:
                return

            try:
                performance_model = self._system_config.resolve_model("performance")
                eco_model = self._system_config.resolve_model("eco")
            except ValueError:
                return

            if self._system_config.get_mode() != "eco" or self._model_name != eco_model:
                return

            healthy, payload_or_error = await self._fetch_ollama_tags()
            if not healthy:
                self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown_seconds
                logger.warning(
                    "Performance auto-retry deferred: Ollama tags unavailable (%s).",
                    payload_or_error,
                )
                return

            assert isinstance(payload_or_error, dict)
            available_models = self._extract_model_names(payload_or_error)
            if performance_model not in available_models:
                self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown_seconds
                logger.warning(
                    "Performance auto-retry deferred: model '%s' unavailable.",
                    performance_model,
                )
                return

            try:
                await self._prewarm_model(performance_model)
            except (OllamaUnavailableError, OllamaModelError) as error:
                self._performance_retry_after_monotonic = time.monotonic() + self._performance_retry_cooldown_seconds
                logger.warning(
                    "Performance auto-retry failed prewarm and will retry later: %s",
                    error,
                )
                return

            self._system_config.set_mode("performance")
            self._model_name = performance_model
            self._performance_downgrade_active = False
            self._last_performance_downgrade_reason = None
            self._performance_retry_after_monotonic = 0.0
            logger.info("Performance mode auto-recovered after cooldown.")

    @staticmethod
    def _is_runner_memory_pressure(detail: str) -> bool:
        normalized = detail.lower()
        return (
            "runner process has terminated" in normalized
            or "out of memory" in normalized
            or "cuda out of memory" in normalized
            or "failed to allocate" in normalized
        )

    async def _fetch_ollama_processes(self) -> tuple[bool, dict[str, Any] | str]:
        try:
            timeout = httpx.Timeout(self._ollama_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                response = await client.get("/api/ps")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            return False, f"unable to connect ({error})"
        except httpx.HTTPError as error:
            return False, f"HTTP error ({error})"

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"

        try:
            payload = response.json()
        except ValueError as error:
            return False, f"invalid process payload ({error})"

        return True, payload

    @staticmethod
    def _read_int_file(path: str) -> int | None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
        except OSError:
            return None

        if not value or value == "max":
            return None

        try:
            return int(value)
        except ValueError:
            return None

    async def _collect_runtime_metrics(self) -> dict[str, Any]:
        async with self._metrics_lock:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk_counters = psutil.disk_io_counters()

            now = time.monotonic()
            disk_io_mbps = 0.0
            if disk_counters is not None:
                total_bytes = int(disk_counters.read_bytes + disk_counters.write_bytes)
                if self._last_disk_io_bytes is not None and self._last_disk_io_monotonic > 0:
                    elapsed = max(now - self._last_disk_io_monotonic, 1e-6)
                    disk_io_mbps = max(0.0, (total_bytes - self._last_disk_io_bytes) / elapsed / (1024 * 1024))
                self._last_disk_io_bytes = total_bytes
                self._last_disk_io_monotonic = now

            cgroup_mem_current = self._read_int_file("/sys/fs/cgroup/memory.current")
            cgroup_mem_max = self._read_int_file("/sys/fs/cgroup/memory.max")
            cgroup_swap_current = self._read_int_file("/sys/fs/cgroup/memory.swap.current")
            cgroup_swap_max = self._read_int_file("/sys/fs/cgroup/memory.swap.max")

            container_remaining_bytes: int | None = None
            if cgroup_mem_current is not None and cgroup_mem_max is not None:
                container_remaining_bytes = max(0, cgroup_mem_max - cgroup_mem_current)

            available_ram_bytes = int(vm.available)
            if container_remaining_bytes is not None:
                available_ram_bytes = min(available_ram_bytes, container_remaining_bytes)

            cgroup_swap_percent: float | None = None
            if cgroup_swap_current is not None and cgroup_swap_max is not None and cgroup_swap_max > 0:
                cgroup_swap_percent = (cgroup_swap_current / cgroup_swap_max) * 100.0

            host_swap_percent = float(swap.percent)
            pagefile_usage_percent = cgroup_swap_percent if cgroup_swap_percent is not None else host_swap_percent
            pressure_guardrail_triggered = (
                cgroup_swap_percent is not None and cgroup_swap_percent > self._pagefile_guardrail_percent
            )
            pagefile_proxy_source = "cgroup" if cgroup_swap_percent is not None else "host-informational"

            gpu_load_percent = 0
            healthy_ps, ps_payload_or_error = await self._fetch_ollama_processes()
            total_vram_bytes = 0
            if healthy_ps and isinstance(ps_payload_or_error, dict):
                models = ps_payload_or_error.get("models")
                if isinstance(models, list):
                    for model in models:
                        if isinstance(model, dict) and isinstance(model.get("size_vram"), int):
                            total_vram_bytes += int(model["size_vram"])

            if self._intel_gpu_requested and not self._gpu_soft_fallback and total_vram_bytes > 0:
                gpu_load_percent = max(1, min(100, int((total_vram_bytes / self._llm_memory_cap_bytes) * 100)))
            elif self._intel_gpu_requested and self._gpu_preflight_status == "active":
                gpu_load_percent = 95

            telemetry_source = "measured" if healthy_ps else "limited"

            return {
                "gpu_load_percent": gpu_load_percent,
                "ram_available_bytes": available_ram_bytes,
                "disk_io_mbps": round(disk_io_mbps, 2),
                "pagefile_usage_percent": round(pagefile_usage_percent, 2),
                "host_swap_percent": round(host_swap_percent, 2),
                "pagefile_guardrail_percent": self._pagefile_guardrail_percent,
                "pressure_guardrail_triggered": pressure_guardrail_triggered,
                "pagefile_proxy_source": pagefile_proxy_source,
                "telemetry_source": telemetry_source,
            }

    async def _enforce_pressure_guardrail(self, model_name: str) -> None:
        metrics = await self._collect_runtime_metrics()
        if not metrics["pressure_guardrail_triggered"]:
            self._last_pressure_error = None
            return

        error = (
            "Memory pressure guardrail triggered before inference for model "
            f"'{model_name}' (pagefile proxy {metrics['pagefile_usage_percent']}% > "
            f"{metrics['pagefile_guardrail_percent']}%). Flush memory and retry."
        )
        self._last_pressure_error = error
        raise OllamaUnavailableError(error)

    async def _run_intel_gpu_preflight(self) -> None:
        if self._gpu_preflight_checked:
            return

        self._gpu_preflight_checked = True
        if not self._intel_gpu_requested:
            self._gpu_preflight_status = "disabled"
            self._gpu_preflight_message = "Intel iGPU acceleration not requested."
            return

        if self._model_name:
            try:
                await self._prewarm_model(self._model_name)
            except (OllamaUnavailableError, OllamaModelError) as error:
                self._gpu_soft_fallback = True
                self._gpu_preflight_status = "cpu-fallback"
                self._gpu_preflight_message = (
                    "Intel iGPU preflight failed while preparing model. "
                    f"Falling back to CPU: {error}"
                )
                logger.warning(self._gpu_preflight_message)
                return

        healthy, payload_or_error = await self._fetch_ollama_processes()
        if not healthy:
            self._gpu_soft_fallback = True
            self._gpu_preflight_status = "cpu-fallback"
            self._gpu_preflight_message = (
                "Intel iGPU preflight could not inspect Ollama process metadata. "
                f"Falling back to CPU: {payload_or_error}"
            )
            logger.warning(self._gpu_preflight_message)
            return

        assert isinstance(payload_or_error, dict)
        models = payload_or_error.get("models")
        if not isinstance(models, list):
            models = []

        total_model_bytes = 0
        total_vram_bytes = 0
        for model in models:
            if not isinstance(model, dict):
                continue
            size_bytes = model.get("size")
            if isinstance(size_bytes, int):
                total_model_bytes += size_bytes
            size_vram = model.get("size_vram")
            if isinstance(size_vram, int):
                total_vram_bytes += size_vram

        if total_model_bytes > self._llm_memory_cap_bytes:
            self._gpu_soft_fallback = True
            self._gpu_preflight_status = "cpu-fallback"
            self._gpu_preflight_message = (
                "LLM process footprint exceeded 8GB shared memory safety cap during preflight. "
                "Falling back to CPU."
            )
            logger.warning(self._gpu_preflight_message)
            return

        if total_vram_bytes <= 0:
            self._gpu_soft_fallback = True
            self._gpu_preflight_status = "cpu-fallback"
            self._gpu_preflight_message = (
                "Intel iGPU preflight could not confirm Vulkan/oneAPI acceleration from Ollama runtime metrics. "
                "Falling back to CPU."
            )
            logger.warning(self._gpu_preflight_message)
            return

        self._gpu_preflight_status = "active"
        self._gpu_preflight_message = "Intel iGPU acceleration active (Vulkan/oneAPI heuristic passed)."
        logger.info(self._gpu_preflight_message)

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

        await self._maybe_restore_performance_mode()
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
        active_mode = self._active_mode()
        await self._invoke_host_optimizer_if_needed(active_mode, model_name)
        await self._enforce_pressure_guardrail(model_name)
        primary_num_gpu = self._resolve_num_gpu(active_mode, use_fallback_profile=False)
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": self._build_ollama_options(
                num_ctx=self._resolve_num_ctx(active_mode, use_fallback_profile=False, num_ctx_override=None),
                num_thread=self._resolve_num_thread(active_mode, num_thread_override=None),
                num_gpu=primary_num_gpu,
                temperature=DEFAULT_OLLAMA_TEMPERATURE,
            ),
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

            if self._is_kv_cache_option_unsupported(detail) and self._kv_cache_supported:
                self._kv_cache_supported = False
                self._kv_cache_warning = (
                    "Ollama runtime does not support 4-bit KV cache option. Continuing without KV quantization override."
                )
                logger.warning(self._kv_cache_warning)
                return await self._generate_ollama_response(messages, model_name)

            # Retry once with a smaller context when Ollama runner fails to load 8B model reliably.
            if self._is_runner_memory_pressure(detail):
                fallback_num_gpu = self._resolve_num_gpu(active_mode, use_fallback_profile=True)
                retry_payload = {
                    **request_payload,
                    "options": self._build_ollama_options(
                        num_ctx=self._resolve_num_ctx(active_mode, use_fallback_profile=True, num_ctx_override=None),
                        num_thread=self._resolve_num_thread(active_mode, num_thread_override=None),
                        num_gpu=fallback_num_gpu,
                        temperature=DEFAULT_OLLAMA_TEMPERATURE,
                    ),
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
                    emergency_model = self._resolve_emergency_model_fallback(model_name)
                    if emergency_model is not None:
                        self._activate_emergency_eco_mode(emergency_model)
                        logger.warning(
                            "Primary model '%s' failed under memory pressure. Falling back to eco model '%s'.",
                            model_name,
                            emergency_model,
                        )
                        return await self._generate_ollama_response(messages, emergency_model)

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
        num_thread_override: int | None = None,
        use_fallback_profile: bool = False,
        allow_model_fallback: bool = True,
        num_ctx_override: int | None = None,
        allow_eof_retry: bool = True,
    ) -> str:
        active_mode = self._active_mode()
        await self._invoke_host_optimizer_if_needed(active_mode, model_name)
        await self._enforce_pressure_guardrail(model_name)
        num_gpu = self._resolve_num_gpu(active_mode, use_fallback_profile=use_fallback_profile)
        num_ctx = self._resolve_num_ctx(active_mode, use_fallback_profile, num_ctx_override)
        num_thread = self._resolve_num_thread(active_mode, num_thread_override)
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": self._build_ollama_options(
                num_ctx=num_ctx,
                num_thread=num_thread,
                num_gpu=num_gpu,
                temperature=DEFAULT_OLLAMA_TEMPERATURE,
            ),
        }

        try:
            timeout = httpx.Timeout(self._ollama_timeout)
            async with httpx.AsyncClient(base_url=self._ollama_base_url, timeout=timeout) as client:
                async with client.stream("POST", "/api/chat", json=request_payload) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", errors="ignore").strip()
                        detail = detail or f"HTTP {response.status_code}"
                        if self._is_kv_cache_option_unsupported(detail) and self._kv_cache_supported:
                            self._kv_cache_supported = False
                            self._kv_cache_warning = (
                                "Ollama runtime does not support 4-bit KV cache option. Continuing without KV quantization override."
                            )
                            logger.warning(self._kv_cache_warning)
                            return await self._generate_ollama_response_streaming(
                                messages=messages,
                                model_name=model_name,
                                on_delta=on_delta,
                                num_thread_override=num_thread_override,
                                use_fallback_profile=use_fallback_profile,
                                allow_model_fallback=allow_model_fallback,
                                num_ctx_override=num_ctx_override,
                                allow_eof_retry=allow_eof_retry,
                            )
                        if self._is_runner_memory_pressure(detail) and not use_fallback_profile:
                            return await self._generate_ollama_response_streaming(
                                messages=messages,
                                model_name=model_name,
                                on_delta=on_delta,
                                num_thread_override=num_thread_override,
                                use_fallback_profile=True,
                                allow_model_fallback=allow_model_fallback,
                                allow_eof_retry=allow_eof_retry,
                            )
                        if self._is_runner_memory_pressure(detail) and use_fallback_profile and allow_model_fallback:
                            emergency_model = self._resolve_emergency_model_fallback(model_name)
                            if emergency_model is not None:
                                self._activate_emergency_eco_mode(emergency_model)
                                logger.warning(
                                    "Streaming load failed for '%s' under memory pressure. Falling back to eco model '%s'.",
                                    model_name,
                                    emergency_model,
                                )
                                return await self._generate_ollama_response_streaming(
                                    messages=messages,
                                    model_name=emergency_model,
                                    on_delta=on_delta,
                                    num_thread_override=num_thread_override,
                                    use_fallback_profile=True,
                                    allow_model_fallback=False,
                                    allow_eof_retry=allow_eof_retry,
                                )
                        if response.status_code in (400, 404):
                            lower_detail = detail.lower()
                            if "model" in lower_detail and ("not found" in lower_detail or "missing" in lower_detail):
                                raise OllamaModelError(detail)
                        raise OllamaUnavailableError(detail)

                    accumulated = ""
                    try:
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
                    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.DecodingError) as error:
                        if allow_eof_retry and self._is_unexpected_eof_error(str(error)):
                            await asyncio.sleep(DEFAULT_EOF_RETRY_DELAY_SECONDS)
                            return await self._generate_ollama_response_streaming(
                                messages=messages,
                                model_name=model_name,
                                on_delta=on_delta,
                                num_thread_override=num_thread_override,
                                use_fallback_profile=False,
                                allow_model_fallback=allow_model_fallback,
                                num_ctx_override=FALLBACK_OLLAMA_NUM_CTX,
                                allow_eof_retry=False,
                            )
                        raise OllamaUnavailableError(str(error)) from error

                    if not accumulated.strip():
                        if allow_eof_retry:
                            await asyncio.sleep(DEFAULT_EOF_RETRY_DELAY_SECONDS)
                            return await self._generate_ollama_response_streaming(
                                messages=messages,
                                model_name=model_name,
                                on_delta=on_delta,
                                num_thread_override=num_thread_override,
                                use_fallback_profile=False,
                                allow_model_fallback=allow_model_fallback,
                                num_ctx_override=FALLBACK_OLLAMA_NUM_CTX,
                                allow_eof_retry=False,
                            )
                        raise OllamaResponseError("assistant message content is empty")

                    return accumulated.strip()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            raise OllamaUnavailableError(str(error)) from error
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(str(error)) from error

    async def _prewarm_model(self, model_name: str) -> None:
        mode = self._resolve_mode_for_model(model_name)
        num_gpu = self._resolve_num_gpu(mode, use_fallback_profile=False)
        prewarm_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": self._build_ollama_options(
                num_ctx=DEFAULT_PREWARM_NUM_CTX,
                num_thread=self._resolve_num_thread(mode, num_thread_override=None),
                num_gpu=num_gpu,
                temperature=0,
            ),
        }
        prewarm_payload["options"]["num_predict"] = 1
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
            wait_seconds = 0.0
            now_monotonic = time.monotonic()
            elapsed = now_monotonic - self._last_mode_switch_monotonic
            if self._last_mode_switch_monotonic > 0 and elapsed < self._mode_switch_cooldown_seconds:
                wait_seconds = self._mode_switch_cooldown_seconds - elapsed
                await asyncio.sleep(wait_seconds)

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
            self._last_mode_switch_monotonic = time.monotonic()

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
                "mode_switch_wait_ms": int(wait_seconds * 1000),
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

        await self._maybe_restore_performance_mode()
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
            requested_model_name = self._active_model_name()
            model_name, effective_mode = await self._resolve_chat_model_for_current_pressure(requested_model_name)
            stream_num_thread = self._resolve_http_chat_num_thread(effective_mode)

            try:
                await on_stream_start(model_name)
                response_text = await self._generate_ollama_response_streaming(
                    messages=messages,
                    model_name=model_name,
                    on_delta=on_stream_delta,
                    num_thread_override=stream_num_thread,
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
        num_ctx_override: int | None,
        on_text_chunk: Callable[[str], Awaitable[None]],
        on_sentence_audio: Callable[[int, str, AssistantAudioPayload | None, str | None], Awaitable[None]],
        on_final_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        if not user_text.strip():
            raise RuntimeError("I did not receive any text to process.")

        await self._maybe_restore_performance_mode()
        await self.create_session(session_id)
        messages = self._sessions.get(session_id)
        if messages is None:
            raise RuntimeError("Jarvis session state is unavailable. Please restart the session.")

        session_lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        sentence_buffer = ""
        text_emit_buffer = ""
        last_text_emit_monotonic = 0.0
        sentence_index = 0
        sentence_tasks: list[asyncio.Task[None]] = []

        async def flush_text_emit_buffer(force: bool = False) -> None:
            nonlocal text_emit_buffer, last_text_emit_monotonic
            if not text_emit_buffer:
                return

            now = time.monotonic()
            should_flush = (
                force
                or len(text_emit_buffer) >= DEFAULT_STREAM_TEXT_FLUSH_MIN_CHARS
                or (now - last_text_emit_monotonic) >= DEFAULT_STREAM_TEXT_FLUSH_SECONDS
            )

            if not should_flush:
                return

            chunk = text_emit_buffer
            text_emit_buffer = ""
            last_text_emit_monotonic = now
            await on_text_chunk(chunk)

        async def on_delta(delta: str) -> None:
            nonlocal sentence_buffer, sentence_index, text_emit_buffer
            if not delta:
                return

            text_emit_buffer += delta
            await flush_text_emit_buffer(force=False)

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
            requested_model_name = self._active_model_name()
            model_name, effective_mode = await self._resolve_chat_model_for_current_pressure(requested_model_name)
            http_num_thread = self._resolve_http_chat_num_thread(effective_mode)

            try:
                final_text = await self._generate_ollama_response_streaming(
                    messages=messages,
                    model_name=model_name,
                    on_delta=on_delta,
                    num_thread_override=http_num_thread,
                    num_ctx_override=num_ctx_override,
                )

                await flush_text_emit_buffer(force=True)

                final_text = final_text.strip()
                messages.append({"role": "assistant", "content": final_text})
                self._ollama_available = True
                self._initialization_error = None

                if on_final_text is not None:
                    try:
                        await on_final_text(final_text)
                    except RuntimeError as error:
                        logger.warning("Final text callback failed: %s", error)

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
        runtime_metrics = await self._collect_runtime_metrics()
        tts_runtime = await self._tts_engine.get_runtime_status()

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
                "GPU_Load": runtime_metrics["gpu_load_percent"],
                "RAM_Available": round(runtime_metrics["ram_available_bytes"] / (1024**3), 2),
                "RAM_Available_Bytes": runtime_metrics["ram_available_bytes"],
                "Disk_IO": runtime_metrics["disk_io_mbps"],
                "Disk_IO_MBps": runtime_metrics["disk_io_mbps"],
                "pagefile_usage_percent": runtime_metrics["pagefile_usage_percent"],
                "pagefile_guardrail_percent": runtime_metrics["pagefile_guardrail_percent"],
                "pressure_guardrail_triggered": runtime_metrics["pressure_guardrail_triggered"],
                "telemetry_source": runtime_metrics["telemetry_source"],
            },
            "accelerator": {
                "intel_gpu_requested": self._intel_gpu_requested,
                "gpu_soft_fallback": self._gpu_soft_fallback,
                "preflight_status": self._gpu_preflight_status,
                "preflight_message": self._gpu_preflight_message,
                "kv_cache_target": DEFAULT_KV_CACHE_TYPE,
                "kv_cache_supported": self._kv_cache_supported,
                "kv_cache_warning": self._kv_cache_warning,
                "performance_downgrade_active": self._performance_downgrade_active,
                "last_performance_downgrade_at": self._last_performance_downgrade_at,
                "last_performance_downgrade_reason": self._last_performance_downgrade_reason,
                "performance_retry_after_ms": max(
                    0,
                    int((self._performance_retry_after_monotonic - time.monotonic()) * 1000),
                )
                if self._performance_downgrade_active
                else 0,
            },
            "guardrail": {
                "last_pressure_error": self._last_pressure_error,
                "low_ram_force_eco_message": self._last_low_ram_force_eco_message,
                "low_ram_force_eco_threshold_bytes": self._low_ram_force_eco_threshold_bytes,
                "high_swap_force_eco_percent": self._high_swap_force_eco_percent,
            },
            "profile": {
                "eco": {
                    "num_ctx": DEFAULT_ECO_NUM_CTX,
                    "num_thread": DEFAULT_ECO_NUM_THREAD,
                    "num_gpu": DEFAULT_ECO_NUM_GPU,
                },
                "performance": {
                    "num_ctx": DEFAULT_PERFORMANCE_NUM_CTX,
                    "num_thread": DEFAULT_PERFORMANCE_NUM_THREAD,
                    "num_gpu": DEFAULT_PERFORMANCE_NUM_GPU,
                },
                "active_mode_profile": {
                    "num_ctx": self._resolve_num_ctx(active_mode, False, None),
                    "num_thread": self._resolve_num_thread(active_mode, None),
                    "num_gpu": self._resolve_num_gpu(active_mode, False),
                },
            },
            "capabilities": {
                "num_batch_requested": self._num_batch_requested,
                "num_batch_supported": self._num_batch_supported,
                "num_batch_warning": self._num_batch_warning,
                "host_optimizer": {
                    "url_configured": bool(self._host_optimizer_url),
                    "status": self._host_optimizer_last_status,
                    "message": self._host_optimizer_last_message,
                },
            },
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
            "intel_gpu_requested": str(self._intel_gpu_requested).lower(),
            "gpu_soft_fallback": str(self._gpu_soft_fallback).lower(),
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
