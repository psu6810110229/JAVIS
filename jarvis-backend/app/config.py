from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_project_env() -> None:
    backend_dir = Path(__file__).resolve().parent.parent
    project_root = backend_dir.parent
    load_dotenv(project_root / ".env")


class SystemConfig:
    _instance: SystemConfig | None = None
    _instance_lock = RLock()

    def __new__(cls) -> SystemConfig:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._lock = RLock()
        eco_model = (
            os.getenv("JARVIS_MODEL_ECO", "").strip()
            or "scb10x/typhoon2.5-qwen3-4b"
        )
        performance_model = (
            os.getenv("JARVIS_MODEL_PERFORMANCE", "").strip()
            or os.getenv("JARVIS_MODEL_NAME", "").strip()
            or "scb10x/llama3.1-typhoon2-8b-instruct"
        )
        eco_model = self._normalize_model_name(eco_model)
        performance_model = self._normalize_model_name(performance_model)
        self._models = {
            "eco": eco_model,
            "performance": performance_model,
        }
        self._mode = "eco"
        self._active_model = self._models[self._mode]

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
