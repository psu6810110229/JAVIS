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
from collections.abc import Awaitable, Callable
from typing import Any

from app.brain.models import ToolSchema

logger = logging.getLogger(__name__)

# Internal type alias
_ToolFn = Callable[..., Any]


class ToolRegistry:
    """Global registry for callable tools exposed to the LLM function-calling engine."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolFn] = {}
        self._schemas: dict[str, ToolSchema] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        fn: _ToolFn,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> _ToolFn:
        """Register *fn* as a tool with the given schema. Returns *fn* unchanged."""
        if name in self._tools:
            logger.warning("Tool '%s' is already registered. Overwriting.", name)
        self._tools[name] = fn
        self._schemas[name] = ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
        )
        logger.debug("Registered tool '%s'.", name)
        return fn

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[ToolSchema]:
        """Return all registered tool schemas (safe copy)."""
        return list(self._schemas.values())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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

        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**arguments)
            else:
                result = await asyncio.to_thread(fn, **arguments)
        except Exception as error:
            logger.exception("Tool '%s' raised an exception: %s", name, error)
            raise

        return result


# ---------------------------------------------------------------------------
# Module-level singleton and decorator
# ---------------------------------------------------------------------------

#: The single global registry instance imported by all modules.
tool_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
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
        tool_registry.register(fn, name=name, description=description, parameters=parameters)
        return fn

    return decorator
