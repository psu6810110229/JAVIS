"""brain/personal_profile.py - lightweight persistent user profile memory.

Stores user preferences/goals in a local JSON file so Jarvis can personalize
responses across sessions without external dependencies.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class PersonalProfileStore:
    """Thread-safe profile store with simple rule-based preference extraction."""

    def __init__(self) -> None:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        project_root = backend_dir.parent
        data_dir = project_root / "jarvis-backend" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "user_profile.json"
        self._lock = RLock()
        self._profile: dict[str, Any] = self._default_profile()
        self._load()

    @staticmethod
    def _default_profile() -> dict[str, Any]:
        return {
            "name": "",
            "response_style": "adaptive",
            "command_priority": True,
            "tone": "professional-human",
            "goals": [],
            "likes": [],
            "notes": [],
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged = self._default_profile()
                merged.update(raw)
                self._profile = merged
        except (OSError, ValueError, TypeError) as err:
            logger.warning("Failed loading user profile: %s", err)

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._profile, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as err:
            logger.warning("Failed saving user profile: %s", err)

    @staticmethod
    def _append_unique(values: list[str], candidate: str, limit: int = 12) -> bool:
        candidate_clean = candidate.strip()
        if not candidate_clean:
            return False
        if candidate_clean in values:
            return False
        values.append(candidate_clean)
        if len(values) > limit:
            del values[0 : len(values) - limit]
        return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": str(self._profile.get("name") or "").strip(),
                "response_style": str(self._profile.get("response_style") or "adaptive"),
                "command_priority": bool(self._profile.get("command_priority", True)),
                "tone": str(self._profile.get("tone") or "professional-human"),
                "goals": list(self._profile.get("goals") or []),
                "likes": list(self._profile.get("likes") or []),
                "notes": list(self._profile.get("notes") or []),
            }

    def ingest_user_text(self, user_text: str) -> None:
        """Extract lightweight persistent preferences from user utterances."""
        normalized = user_text.strip()
        if not normalized:
            return

        lowered = normalized.lower()
        changed = False

        with self._lock:
            # Name capture
            match_name = re.search(r"\b(?:my name is|call me|i am)\s+([a-zA-Z][a-zA-Z0-9_\- ]{1,30})", normalized, re.IGNORECASE)
            if match_name:
                self._profile["name"] = match_name.group(1).strip()
                changed = True

            # Style preference
            if "short" in lowered and "long" in lowered:
                if self._profile.get("response_style") != "adaptive":
                    self._profile["response_style"] = "adaptive"
                    changed = True
            elif "short answer" in lowered or "be concise" in lowered:
                if self._profile.get("response_style") != "short":
                    self._profile["response_style"] = "short"
                    changed = True
            elif "detailed" in lowered or "long answer" in lowered:
                if self._profile.get("response_style") != "detailed":
                    self._profile["response_style"] = "detailed"
                    changed = True

            # Command priority preference
            if "do whatever i command" in lowered or "command first" in lowered:
                if not bool(self._profile.get("command_priority", True)):
                    self._profile["command_priority"] = True
                    changed = True

            # Goal extraction
            for pattern in [r"\bi want to\s+([^.!?\n]+)", r"\bmy goal is to\s+([^.!?\n]+)"]:
                for match in re.finditer(pattern, normalized, re.IGNORECASE):
                    if self._append_unique(self._profile["goals"], match.group(1)):
                        changed = True

            # Likes extraction
            for match in re.finditer(r"\bi like\s+([^.!?\n]+)", normalized, re.IGNORECASE):
                if self._append_unique(self._profile["likes"], match.group(1)):
                    changed = True

            # Persistent high-level notes from explicit preference lines
            if "very personal" in lowered or "know exactly about me" in lowered:
                if self._append_unique(
                    self._profile["notes"],
                    "User wants highly personalized responses grounded in known preferences.",
                ):
                    changed = True

            if changed:
                self._save()
