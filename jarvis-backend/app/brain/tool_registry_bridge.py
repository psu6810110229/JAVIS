"""Safe tool execution wrapper for standby controller."""
from __future__ import annotations

from typing import Any

from app.actuators.registry import ToolRegistry


class SafeToolRegistry:
    """Allowlisted adapter over ToolRegistry for standby controller usage."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tools: set[str],
        denied_tools: set[str] | None = None,
        policy_label: str = "standby allowlist",
    ) -> None:
        self._registry = registry
        self._allowed_tools = set(allowed_tools)
        self._denied_tools = set(denied_tools or set())
        self._policy_label = policy_label

    @property
    def allowed_tools(self) -> set[str]:
        return set(self._allowed_tools)

    @property
    def denied_tools(self) -> set[str]:
        return set(self._denied_tools)

    def is_allowed(self, tool_name: str) -> bool:
        if tool_name in self._denied_tools:
            return False
        return tool_name in self._allowed_tools

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("Tool args must be a JSON object.")
        if tool_name in self._denied_tools:
            raise PermissionError(f"Tool '{tool_name}' is blocked by {self._policy_label} denylist.")
        if not self.is_allowed(tool_name):
            raise PermissionError(f"Tool '{tool_name}' is not in {self._policy_label}.")
        if not self._registry.has_tool(tool_name):
            raise KeyError(f"Tool '{tool_name}' is not registered.")
        result = await self._registry.execute(tool_name, args)
        if isinstance(result, dict):
            return result
        return {"status": "ok", "result": result}
