"""actuators/registry.py — Phase-4 Tool Registry with decorator pattern.

Usage
-----
    from app.actuators.registry import tool, tool_registry

    @tool(
        name="my_action",
        description="Does something useful.",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Some input"}
            },
            "required": ["param1"],
        },
    )
    async def my_action(param1: str) -> dict:
        return {"result": param1}

After decoration, `my_action` is auto-registered and available via
`tool_registry.get_schemas()` and `tool_registry.execute("my_action", {...})`.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import types
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Union, get_args, get_origin, get_type_hints

from app.brain.models import ToolSchema

logger = logging.getLogger(__name__)

# Internal type alias
_ToolFn = Callable[..., Any]


class ToolRegistry:
    """Global registry for callable tools exposed to the LLM function-calling engine."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolFn] = {}
        self._schemas: dict[str, ToolSchema] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        fn: _ToolFn,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None,
        timeout_seconds: float | None = None,
        estimated_ram_mb: int = 64,
        required_mode: str = "either",
        risk_level: str = "low",
        category: str = "general",
    ) -> _ToolFn:
        """Register *fn* as a tool with the given schema. Returns *fn* unchanged."""
        if name in self._tools:
            logger.warning("Tool '%s' is already registered. Overwriting.", name)

        resolved_description = description.strip() if description.strip() else self._infer_description(fn)
        resolved_parameters = parameters if parameters is not None else self._infer_parameters_schema(fn)
        tool_metadata = {
            "timeout_seconds": timeout_seconds,
            "estimated_ram_mb": max(16, int(estimated_ram_mb)),
            "required_mode": required_mode,
            "risk_level": risk_level,
            "category": category,
            "is_async": inspect.iscoroutinefunction(fn),
        }
        resolved_parameters = {
            **resolved_parameters,
            "x-jarvis-meta": tool_metadata,
        }

        self._tools[name] = fn
        self._schemas[name] = ToolSchema(
            name=name,
            description=resolved_description,
            parameters=resolved_parameters,
        )
        self._metadata[name] = tool_metadata
        logger.debug("Registered tool '%s'.", name)
        return fn

    @staticmethod
    def _infer_description(fn: _ToolFn) -> str:
        doc = inspect.getdoc(fn) or ""
        first_line = doc.splitlines()[0].strip() if doc else ""
        return first_line or f"Execute {fn.__name__}."

    @staticmethod
    def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
        if annotation is inspect._empty:
            return {"type": "string"}
        if annotation is Any:
            return {"type": "object"}

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (types.UnionType, Union):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return ToolRegistry._annotation_to_schema(non_none[0])
            return {"type": "string"}

        if origin in (list, tuple, set):
            item_schema = ToolRegistry._annotation_to_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item_schema}

        if origin is dict:
            return {"type": "object"}

        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}

        return {"type": "string"}

    @staticmethod
    def _infer_parameters_schema(fn: _ToolFn) -> dict[str, Any]:
        signature = inspect.signature(fn)
        type_hints = get_type_hints(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name.startswith("_"):
                continue

            annotation = type_hints.get(name, param.annotation)
            schema_entry = ToolRegistry._annotation_to_schema(annotation)
            schema_entry.setdefault("description", f"Parameter '{name}'.")
            properties[name] = schema_entry
            if param.default is inspect._empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[ToolSchema]:
        """Return all registered tool schemas (safe copy)."""
        return list(self._schemas.values())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return execution metadata for a registered tool."""
        if name not in self._metadata:
            raise KeyError(f"Tool '{name}' is not registered.")
        return dict(self._metadata[name])

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_result(name: str, raw_result: Any, elapsed_ms: int) -> dict[str, Any]:
        if isinstance(raw_result, dict):
            normalized: dict[str, Any] = dict(raw_result)
        else:
            normalized = {"result": raw_result}

        normalized.setdefault("status", "ok")
        normalized.setdefault("verified", None)
        normalized.setdefault("evidence", None)
        normalized.setdefault("error", None)
        normalized.setdefault("warning", None)
        normalized["tool_name"] = name
        normalized["elapsed_ms"] = elapsed_ms
        normalized["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return normalized

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute the named tool with the supplied *arguments* dict.

        Supports both sync and async tool functions.

        Raises
        ------
        KeyError
            If no tool with *name* is registered.
        ValueError
            If *arguments* contains unexpected keys (basic validation).
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")

        schema = self._schemas[name]
        expected_props = set(
            schema.parameters.get("properties", {}).keys()
        )
        unexpected = set(arguments.keys()) - expected_props
        if unexpected:
            raise ValueError(
                f"Tool '{name}' received unexpected arguments: {unexpected}. "
                f"Expected properties: {expected_props}."
            )

        required = set(schema.parameters.get("required", []))
        missing = required - set(arguments.keys())
        if missing:
            raise ValueError(
                f"Tool '{name}' is missing required arguments: {missing}."
            )

        fn = self._tools[name]
        logger.info("Executing tool '%s' with args: %s", name, arguments)
        metadata = self._metadata.get(name, {})
        timeout_seconds = metadata.get("timeout_seconds")
        started = time.monotonic()

        try:
            if inspect.iscoroutinefunction(fn):
                call = fn(**arguments)
            else:
                call = asyncio.to_thread(fn, **arguments)

            if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
                result = await asyncio.wait_for(call, timeout=float(timeout_seconds))
            else:
                result = await call
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                f"Tool '{name}' timed out after {timeout_seconds:.2f} seconds."
            ) from error
        except Exception as error:
            logger.exception("Tool '%s' raised an exception: %s", name, error)
            raise

        elapsed_ms = int((time.monotonic() - started) * 1000)
        return self._normalize_result(name=name, raw_result=result, elapsed_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Module-level singleton and decorator
# ---------------------------------------------------------------------------

#: The single global registry instance imported by all modules.
tool_registry = ToolRegistry()


def tool(
    name: str,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    *,
    timeout_seconds: float | None = None,
    estimated_ram_mb: int = 64,
    required_mode: str = "either",
    risk_level: str = "low",
    category: str = "general",
) -> Callable[[_ToolFn], _ToolFn]:
    """Decorator that registers a function as a Jarvis tool.

    Parameters
    ----------
    name:
        Unique identifier the LLM uses to call this tool.
    description:
        Natural-language description shown to the LLM.
    parameters:
        JSON Schema object describing the function's parameters.

    Example
    -------
    ::

        @tool(
            name="get_weather",
            description="Get current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        )
        async def get_weather(city: str) -> dict:
            ...
    """
    def decorator(fn: _ToolFn) -> _ToolFn:
        tool_registry.register(
            fn,
            name=name,
            description=description,
            parameters=parameters,
            timeout_seconds=timeout_seconds,
            estimated_ram_mb=estimated_ram_mb,
            required_mode=required_mode,
            risk_level=risk_level,
            category=category,
        )
        return fn

    return decorator
