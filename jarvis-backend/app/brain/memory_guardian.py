"""brain/memory_guardian.py — pre-tool resource guardrail for Phase 4.1.

This layer checks runtime pressure before executing a tool call so JAVIS can
avoid swap thrashing and ask for mode confirmation when the host is unsafe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardianDecision:
    """Decision returned by MemoryGuardian for a proposed tool call."""

    allowed: bool
    reason: str
    requires_mode_confirmation: bool = False
    suggested_mode: str | None = None
    available_ram_bytes: int = 0
    pagefile_percent: float = 0.0
    projected_required_ram_bytes: int = 0
    tool_name: str | None = None


class MemoryGuardian:
    """Resource-aware guard before tool execution."""

    def __init__(
        self,
        *,
        low_ram_force_eco_bytes: int,
        high_swap_force_eco_percent: float,
        pagefile_guardrail_percent: float,
        safety_reserve_bytes: int = 768 * 1024 * 1024,
    ) -> None:
        self._low_ram_force_eco_bytes = low_ram_force_eco_bytes
        self._high_swap_force_eco_percent = high_swap_force_eco_percent
        self._pagefile_guardrail_percent = pagefile_guardrail_percent
        self._safety_reserve_bytes = max(256 * 1024 * 1024, safety_reserve_bytes)

    def assess(
        self,
        *,
        tool_name: str,
        tool_metadata: dict[str, Any],
        active_mode: str,
        runtime_metrics: dict[str, Any],
    ) -> GuardianDecision:
        """Assess whether a tool call is safe under current runtime pressure."""
        available_ram = int(runtime_metrics.get("ram_available_bytes") or 0)
        pagefile_percent = float(runtime_metrics.get("pagefile_usage_percent") or 0.0)
        swap_percent = float(runtime_metrics.get("host_swap_percent") or pagefile_percent)

        estimated_ram_mb = int(tool_metadata.get("estimated_ram_mb") or 64)
        projected_required = (estimated_ram_mb * 1024 * 1024) + self._safety_reserve_bytes
        required_mode = str(tool_metadata.get("required_mode") or "either").lower()

        if pagefile_percent > self._pagefile_guardrail_percent:
            return GuardianDecision(
                allowed=False,
                reason=(
                    f"I cannot run '{tool_name}' right now because pagefile pressure is "
                    f"{pagefile_percent:.1f}% (limit {self._pagefile_guardrail_percent:.1f}%)."
                ),
                available_ram_bytes=available_ram,
                pagefile_percent=pagefile_percent,
                projected_required_ram_bytes=projected_required,
                tool_name=tool_name,
            )

        if available_ram < projected_required:
            return GuardianDecision(
                allowed=False,
                reason=(
                    f"I cannot run '{tool_name}' safely with current free RAM "
                    f"({available_ram / (1024**3):.2f} GB available)."
                ),
                available_ram_bytes=available_ram,
                pagefile_percent=pagefile_percent,
                projected_required_ram_bytes=projected_required,
                tool_name=tool_name,
            )

        if required_mode == "performance" and active_mode != "performance":
            return GuardianDecision(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' requires the deep model mode. "
                    "I need your confirmation before switching modes."
                ),
                requires_mode_confirmation=True,
                suggested_mode="performance",
                available_ram_bytes=available_ram,
                pagefile_percent=pagefile_percent,
                projected_required_ram_bytes=projected_required,
                tool_name=tool_name,
            )

        return GuardianDecision(
            allowed=True,
            reason="Tool execution is safe.",
            available_ram_bytes=available_ram,
            pagefile_percent=pagefile_percent,
            projected_required_ram_bytes=projected_required,
            tool_name=tool_name,
        )
