from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_project_env() -> None:
    backend_dir = Path(__file__).resolve().parent.parent
    project_root = backend_dir.parent
    load_dotenv(project_root / ".env")
