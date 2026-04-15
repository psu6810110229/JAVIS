"""Deterministic route classifier for cloud vs local paths."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    LOCAL_TOOL = "local_tool"
    CLOUD_MULTISTEP = "cloud_multistep"
    CLOUD_SEARCH = "cloud_search"
    CLOUD_REASONING = "cloud_reasoning"
    LOCAL_PASSTHROUGH = "local_passthrough"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    confidence: float
    reason: str
    extracted_query: str = ""


_MULTI_STEP_PATTERNS = [
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bafter\s+that\b", re.IGNORECASE),
    re.compile(r"\band\s+then\b", re.IGNORECASE),
    re.compile(r"\bfollowed\s+by\b", re.IGNORECASE),
    re.compile(r"\bfinally\b", re.IGNORECASE),
    re.compile(r"แล้ว"),
    re.compile(r"จากนั้น"),
    re.compile(r"ต่อ(?:จาก(?:นั้น)?)?"),
]

_ACTION_VERBS = re.compile(
    r"\b(?:create|make|open|launch|start|close|set|adjust|change|move|copy|"
    r"delete|remove|rename|notify|remind|play|pause|skip|mute|unmute|"
    r"lock|shutdown|restart|type|press|click|drag|screenshot|"
    r"สร้าง|เปิด|ปิด|ลบ|ย้าย|คัดลอก|เปลี่ยน|เตือน)\b",
    re.IGNORECASE,
)

_SEARCH_PATTERNS = [
    re.compile(r"\b(?:search|find|lookup|look up|who is|what is)\b", re.IGNORECASE),
    re.compile(r"(?:ค้นหา|หาข้อมูล|หาให้|ใครคือ|อะไรคือ)"),
]

_REASONING_PATTERNS = [
    re.compile(r"\b(?:explain|analyze|compare|summarize|help\s+me)\b", re.IGNORECASE),
    re.compile(r"\b(?:pros?\s+and\s+cons?|trade-?off|strategy)\b", re.IGNORECASE),
]

_LOCAL_TOOL_TRIGGERS = [
    re.compile(r"\b(?:open|launch|start)\s+\w+", re.IGNORECASE),
    re.compile(r"\b(?:set|change|adjust)\s+(?:the\s+)?volume\b", re.IGNORECASE),
    re.compile(r"\b(?:mute|unmute)\b", re.IGNORECASE),
    re.compile(r"\b(?:play|pause|next|previous|skip)\b.*\bspotify\b", re.IGNORECASE),
    re.compile(r"\b(?:create|make)\b.*\b(folder|directory)\b", re.IGNORECASE),
    re.compile(r"\b(?:create|make|write)\b.*\b(docx|report|document)\b", re.IGNORECASE),
    re.compile(r"\b(?:remind|reminder|notification|notify)\b", re.IGNORECASE),
    re.compile(r"\b(?:search|find|list)\b.*\b(file|files|folder|folders|path|paths|project|workspace)\b", re.IGNORECASE),
    re.compile(r"\b(?:open)\b.*\b(file\s+explorer|explorer)\b", re.IGNORECASE),
]

_LOCAL_INTENT_STRONG_PATTERNS = [
    re.compile(r"\b(?:volume|mute|unmute|brightness)\b", re.IGNORECASE),
    re.compile(r"\b(?:spotify|track|playlist|now\s+playing)\b", re.IGNORECASE),
    re.compile(r"\b(?:notification|reminder|clipboard|screenshot)\b", re.IGNORECASE),
    re.compile(r"\b(?:file\s+explorer|open\s+path|search\s+file|in\s+this\s+project|workspace)\b", re.IGNORECASE),
]

_IDENTITY_PASSTHROUGH = re.compile(
    r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|how\s+are\s+you|คุณเป็นใคร|คุณช่วยอะไรได้บ้าง)\b",
    re.IGNORECASE,
)


class HeuristicRouter:
    def __init__(self, *, cloud_enabled: bool = True) -> None:
        self._cloud_enabled = cloud_enabled

    @property
    def cloud_enabled(self) -> bool:
        return self._cloud_enabled

    @cloud_enabled.setter
    def cloud_enabled(self, value: bool) -> None:
        self._cloud_enabled = value

    def classify(self, user_text: str) -> RouteDecision:
        text = user_text.strip()
        if not text:
            return RouteDecision(Route.LOCAL_PASSTHROUGH, 1.0, "empty_input")

        normalized = text.lower()

        if _IDENTITY_PASSTHROUGH.search(text):
            return RouteDecision(Route.LOCAL_PASSTHROUGH, 1.0, "identity_or_greeting")

        local_signal = any(p.search(text) for p in _LOCAL_TOOL_TRIGGERS) or any(
            p.search(text) for p in _LOCAL_INTENT_STRONG_PATTERNS
        )
        if local_signal:
            return RouteDecision(Route.LOCAL_TOOL, 0.93, "local_tool_intent")

        has_multi_step = any(p.search(text) for p in _MULTI_STEP_PATTERNS)
        action_count = len(set(_ACTION_VERBS.findall(normalized)))
        if has_multi_step and action_count >= 2:
            if not self._cloud_enabled:
                return RouteDecision(Route.LOCAL_PASSTHROUGH, 0.6, "multistep_but_cloud_disabled")
            return RouteDecision(Route.CLOUD_MULTISTEP, 0.95, f"multi_step_signal+{action_count}_actions")

        if any(p.search(text) for p in _REASONING_PATTERNS):
            if not self._cloud_enabled:
                return RouteDecision(Route.LOCAL_PASSTHROUGH, 0.5, "reasoning_but_cloud_disabled")
            return RouteDecision(Route.CLOUD_REASONING, 0.85, "reasoning_pattern")

        if any(p.search(text) for p in _SEARCH_PATTERNS):
            if self._cloud_enabled:
                return RouteDecision(
                    Route.CLOUD_SEARCH,
                    0.82,
                    "search_intent",
                    extracted_query=self._extract_search_query(text),
                )
            return RouteDecision(Route.LOCAL_PASSTHROUGH, 0.6, "search_but_cloud_disabled")

        if any(pattern.search(text) for pattern in _LOCAL_TOOL_TRIGGERS):
            return RouteDecision(Route.LOCAL_TOOL, 0.88, "tool_like_intent")

        word_count = len(re.findall(r"\w+", text))
        if word_count >= 12 and self._cloud_enabled:
            return RouteDecision(Route.CLOUD_REASONING, 0.65, f"long_input_{word_count}_words")

        return RouteDecision(Route.LOCAL_PASSTHROUGH, 0.5, "no_pattern_matched")

    @staticmethod
    def _extract_search_query(text: str) -> str:
        query = text.strip()
        prefixes = [
            r"^(?:hey\s+)?(?:jarvis\s*[,:]?\s*)?",
            r"^(?:please\s+)?(?:can\s+you\s+)?",
            r"^(?:search\s+(?:for|about)\s+)",
            r"^(?:look\s+up\s+)",
            r"^(?:find\s+(?:out\s+)?(?:about\s+)?)",
            r"^(?:who\s+is\s+)",
            r"^(?:what\s+is\s+)",
            r"^(?:ค้นหา\s*)",
            r"^(?:หาข้อมูล\s*(?:เกี่ยวกับ\s*)?)",
            r"^(?:ใครคือ\s*)",
            r"^(?:อะไรคือ\s*)",
        ]
        cleaned = query
        for prefix in prefixes:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.rstrip("?!.")
        return cleaned.strip() if cleaned.strip() else query.rstrip("?!.").strip()
