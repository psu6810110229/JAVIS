"""actuators/web_tools.py — Web search tools powered by DuckDuckGo (free, no API key).

Uses ``duckduckgo-search`` (DDGS) which provides:
  - Text search  → web results (title, URL, snippet)
  - News search  → recent headlines
  - Instant answers → facts, conversions, calculations

All network calls run in a thread pool via asyncio.to_thread to avoid
blocking the event loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.actuators.registry import tool

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_HARD_CAP = 10


# ── search_web ────────────────────────────────────────────────────────────────

@tool(
    name="search_web",
    description=(
        "Search the web using DuckDuckGo and return the top results. "
        "Each result includes a title, URL, and a short description snippet. "
        "Use this for any question that requires up-to-date or external information."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Number of results to return (default {_DEFAULT_MAX_RESULTS}, max {_MAX_RESULTS_HARD_CAP}).",
            },
        },
        "required": ["query"],
    },
    risk_level="low",
    category="web",
    timeout_seconds=15,
)
async def search_web(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """Perform a DuckDuckGo web search and return structured results."""
    n = max(1, min(_MAX_RESULTS_HARD_CAP, max_results))

    def _search() -> list[dict[str, str]]:
        from duckduckgo_search import DDGS  # type: ignore[import]

        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=n):
                results.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("href", ""),
                    "snippet": hit.get("body", ""),
                })
        return results

    try:
        results = await asyncio.to_thread(_search)
    except Exception as err:  # noqa: BLE001
        logger.warning("DuckDuckGo web search failed: %s", err)
        return {"query": query, "results": [], "error": str(err)}

    return {"query": query, "results": results, "count": len(results)}


# ── search_news ───────────────────────────────────────────────────────────────

@tool(
    name="search_news",
    description=(
        "Search for recent news articles using DuckDuckGo News. "
        "Returns headlines with title, URL, source, and publication date."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "News topic to search for.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Number of results (default {_DEFAULT_MAX_RESULTS}, max {_MAX_RESULTS_HARD_CAP}).",
            },
        },
        "required": ["query"],
    },
    risk_level="low",
    category="web",
    timeout_seconds=15,
)
async def search_news(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """Search DuckDuckGo News for recent headlines."""
    n = max(1, min(_MAX_RESULTS_HARD_CAP, max_results))

    def _search() -> list[dict[str, str]]:
        from duckduckgo_search import DDGS  # type: ignore[import]

        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for hit in ddgs.news(query, max_results=n):
                results.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("url", ""),
                    "source": hit.get("source", ""),
                    "date": hit.get("date", ""),
                    "snippet": hit.get("body", ""),
                })
        return results

    try:
        results = await asyncio.to_thread(_search)
    except Exception as err:  # noqa: BLE001
        logger.warning("DuckDuckGo news search failed: %s", err)
        return {"query": query, "results": [], "error": str(err)}

    return {"query": query, "results": results, "count": len(results)}


# ── instant_answer ────────────────────────────────────────────────────────────

@tool(
    name="instant_answer",
    description=(
        "Get a DuckDuckGo instant answer for a factual question, unit conversion, "
        "or calculation. Returns a concise direct answer when available."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The factual question or conversion (e.g. 'capital of France', '100 USD to THB').",
            }
        },
        "required": ["query"],
    },
    risk_level="low",
    category="web",
    timeout_seconds=10,
)
async def instant_answer(query: str) -> dict[str, Any]:
    """Fetch a DuckDuckGo instant answer."""
    def _fetch() -> dict[str, Any]:
        from duckduckgo_search import DDGS  # type: ignore[import]

        with DDGS() as ddgs:
            answers = list(ddgs.answers(query))
        if not answers:
            return {"query": query, "answer": None, "found": False}
        top = answers[0]
        return {
            "query": query,
            "answer": top.get("text", ""),
            "url": top.get("url", ""),
            "found": True,
        }

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as err:  # noqa: BLE001
        logger.warning("DuckDuckGo instant answer failed: %s", err)
        return {"query": query, "answer": None, "found": False, "error": str(err)}
