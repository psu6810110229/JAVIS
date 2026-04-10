"""config/settings.py — Centralised configuration, env vars, and hardware profiles.

All constants that were scattered across brain.py are consolidated here.
Hardware profiles are immutable dataclasses so they can be compared safely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_project_env() -> None:
    """Load .env from the project root exactly once per process."""
    backend_dir = Path(__file__).resolve().parent.parent.parent  # app/config/ → jarvis-backend/
    project_root = backend_dir.parent  # jarvis-backend/ → project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareProfile:
    """Immutable hardware configuration profile for a specific mode."""

    mode: str
    num_ctx: int
    num_thread: int
    num_gpu: int
    description: str = ""


# Strict 1024 ctx for performance mode — the Memory Shield requirement.
# E-Cores are excluded; we target only P-Cores (4 on i5-13420H) × 2 threads/core.
PERFORMANCE_PROFILE = HardwareProfile(
    mode="performance",
    num_ctx=1024,
    num_thread=8,   # All P-Cores on i5-13420H
    num_gpu=0,      # CPU-only; set OLLAMA_INTEL_GPU=1 for iGPU offload
    description="Typhoon 8B — full P-Core burst, 1K context window",
)

ECO_PROFILE = HardwareProfile(
    mode="eco",
    num_ctx=1536,   # Slightly larger ctx for the lighter 4B model
    num_thread=4,   # More threads to keep eco model responsive
    num_gpu=0,
    description="Typhoon 4B — balanced power mode",
)

FALLBACK_PROFILE = HardwareProfile(
    mode="fallback",
    num_ctx=512,    # Emergency reduced window after OOM
    num_thread=4,
    num_gpu=0,
    description="Emergency fallback after memory pressure",
)

PREWARM_PROFILE = HardwareProfile(
    mode="prewarm",
    num_ctx=64,
    num_thread=2,
    num_gpu=0,
    description="Minimal load to warm model weights into VRAM/RAM",
)

# HTTP chat uses more threads because it is the primary user-facing path.
HTTP_CHAT_NUM_THREAD: dict[str, int] = {
    "performance": 8,
    "eco": 4,
}


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300.0  # 5 minutes — 9B model on CPU needs time
DEFAULT_OLLAMA_PULL_TIMEOUT_SECONDS = 1800.0
DEFAULT_OLLAMA_TEMPERATURE = 0.7
DEFAULT_OLLAMA_KEEP_ALIVE = -1
DEFAULT_KV_CACHE_TYPE = "q4_0"

# Memory guardrails
DEFAULT_LLM_MEMORY_CAP_BYTES = 8 * 1024 * 1024 * 1024      # 8 GB
DEFAULT_LOW_RAM_FORCE_ECO_BYTES = 4 * 1024 * 1024 * 1024   # 4 GB available RAM floor
DEFAULT_HIGH_SWAP_FORCE_ECO_PERCENT = 35.0                  # Swap % above which eco is forced
DEFAULT_PAGEFILE_GUARDRAIL_PERCENT = 10.0

# Streaming/TTS pipeline
DEFAULT_SENTENCE_TTS_CONCURRENCY = 2
DEFAULT_STREAM_TEXT_FLUSH_SECONDS = 0.10
DEFAULT_STREAM_TEXT_FLUSH_MIN_CHARS = 64
STREAM_TOKEN_DISPATCH_THRESHOLD = 6  # dispatch to TTS after this many tokens

# Retry / cooldown
DEFAULT_EOF_RETRY_DELAY_SECONDS = 0.5
DEFAULT_MODE_SWITCH_COOLDOWN_SECONDS = 0.5
DEFAULT_PERFORMANCE_RETRY_COOLDOWN_SECONDS = 300.0
DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS = 0.35

# STT
DEFAULT_STT_LANGUAGE = "th-TH"


# ---------------------------------------------------------------------------
# Dynamic singleton settings (runtime-modifiable)
# ---------------------------------------------------------------------------

class Settings:
    """Thread-safe singleton for runtime-modifiable settings.

    Model names and the active mode can change at runtime (via /v1/system/model).
    Everything else is read from env at startup and is effectively immutable.
    """

    _instance: "Settings | None" = None
    _instance_lock: RLock = RLock()

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        load_project_env()
        self._lock: RLock = RLock()

        eco_model = (
            os.getenv("JARVIS_MODEL_ECO", "").strip()
            or "scb10x/typhoon2.5-qwen3-4b"
        )
        performance_model = (
            os.getenv("JARVIS_MODEL_PERFORMANCE", "").strip()
            or os.getenv("JARVIS_MODEL_NAME", "").strip()
            or "scb10x/llama3.1-typhoon2-8b-instruct"
        )

        self._models: dict[str, str] = {
            "eco": self._normalize_model_name(eco_model),
            "performance": self._normalize_model_name(performance_model),
        }
        self._mode: str = "eco"
        self._active_model: str = self._models[self._mode]

        # Read-only env settings (parsed once)
        self.ollama_base_url: str = (
            os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
            or DEFAULT_OLLAMA_BASE_URL
        )
        self.intel_gpu_requested: bool = os.getenv("OLLAMA_INTEL_GPU", "0") == "1"
        self.host_optimizer_url: str = os.getenv("JARVIS_HOST_OPTIMIZER_URL", "").strip()

        timeout_raw = os.getenv(
            "JARVIS_HOST_OPTIMIZER_TIMEOUT_SECONDS",
            str(DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS),
        )
        try:
            self.host_optimizer_timeout_seconds: float = max(0.1, float(timeout_raw))
        except ValueError:
            self.host_optimizer_timeout_seconds = DEFAULT_HOST_OPTIMIZER_TIMEOUT_SECONDS

        # Hardware profiles (can be overridden by env later)
        self.performance_profile: HardwareProfile = PERFORMANCE_PROFILE
        self.eco_profile: HardwareProfile = ECO_PROFILE
        self.fallback_profile: HardwareProfile = FALLBACK_PROFILE
        self.prewarm_profile: HardwareProfile = PREWARM_PROFILE

    # ------------------------------------------------------------------
    # Model / mode API
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        stripped = model_name.strip()
        if not stripped:
            return stripped
        last_segment = stripped.rsplit("/", 1)[-1]
        if ":" in last_segment:
            return stripped
        return f"{stripped}:latest"

    def get_models(self) -> dict[str, str]:
        with self._lock:
            return dict(self._models)

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def get_active_model(self) -> str:
        with self._lock:
            return self._active_model

    def resolve_model(self, mode: str) -> str:
        with self._lock:
            if mode not in self._models:
                raise ValueError(f"Unsupported mode '{mode}'.")
            return self._models[mode]

    def set_mode(self, mode: str) -> str:
        with self._lock:
            if mode not in self._models:
                raise ValueError(f"Unsupported mode '{mode}'.")
            self._mode = mode
            self._active_model = self._models[mode]
            return self._active_model

    def get_profile(self, mode: str) -> HardwareProfile:
        """Return the hardware profile for the given mode."""
        with self._lock:
            if mode == "performance":
                return self.performance_profile
            if mode == "eco":
                return self.eco_profile
            return self.fallback_profile
