"""Deterministic no-hardware body controller for tests and examples."""

from __future__ import annotations

from dataclasses import dataclass, field

from reachy_agent.runtime.core.body import BodyAction, BodyActionResult, BodyStatus
from reachy_agent.runtime.core.safety import SafeMovementPolicy, SafetyConfig


@dataclass
class MockBodyController:
    """A fake body controller that records accepted actions and never touches hardware."""

    policy: SafeMovementPolicy = field(
        default_factory=lambda: SafeMovementPolicy(SafetyConfig(body_enabled=True, live_movement_enabled=False))
    )
    actions: list[BodyAction] = field(default_factory=list)
    stopped: bool = False

    async def execute(self, action: BodyAction) -> BodyActionResult:
        safe = self.policy.validate(action, await self.status())
        if safe.action == "stop":
            return await self.stop("stop action")

        self.actions.append(safe)
        self.stopped = False
        return BodyActionResult(
            ok=True,
            action=safe.action,
            turn_id=safe.turn_id,
            message="mock action accepted",
            readback={"mock": True, "recorded_actions": len(self.actions)},
        )

    async def stop(self, reason: str | None = None) -> BodyActionResult:
        self.actions.clear()
        self.stopped = True
        return BodyActionResult(
            ok=True,
            action="stop",
            message=reason or "mock stopped",
            stopped=True,
            readback={"mock": True, "recorded_actions": 0},
        )

    async def status(self) -> BodyStatus:
        return BodyStatus(
            available=False,
            motors_enabled=False,
            current_actions=tuple(action.action for action in self.actions),
            media_available=False,
            error=None,
        )
