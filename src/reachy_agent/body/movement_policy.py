"""Curated, side-effect-free movement planning for AGENT-controlled Reachy (M3).

Ported as-is from the prior-art official-app-adapter (`agent_movement_policy.py`) — it is
pure (stdlib only) and encodes the v0.1 safety allowlist: AGENT may only request a small
set of curated head intents, never arbitrary motion. Planning is separate from execution
so the plan can be inspected/logged before any robot side effect.

Exposed:
- ``AgentMovementIntent`` — the allowlist (look_left/right/up/down/front, stop_motion).
- ``plan_agent_movement(intent)`` — normalise a requested intent into a ``AgentMovementPlan``
  (``blocked`` for anything outside the allowlist; never raises on bad input).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AgentMovementIntent(StrEnum):
    """Curated v0.1 movement intents exposed to AGENT. Anything else is blocked."""

    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    LOOK_FRONT = "look_front"
    STOP_MOTION = "stop_motion"


@dataclass(frozen=True)
class AgentMovementPlan:
    """Side-effect-free movement plan produced before any execution."""

    status: str  # "planned" | "blocked"
    intent: str
    selected_tool: str | None  # "agent_safe_head_motion" | "clear_move_queue" | None
    tool_args: dict[str, Any]
    reason: str
    side_effects: list[str]
    requires_live_execute: bool


_INTENT_TO_DIRECTION = {
    AgentMovementIntent.LOOK_LEFT: "left",
    AgentMovementIntent.LOOK_RIGHT: "right",
    AgentMovementIntent.LOOK_UP: "up",
    AgentMovementIntent.LOOK_DOWN: "down",
    AgentMovementIntent.LOOK_FRONT: "front",
}


def plan_agent_movement(intent: str, *, source: str = "agent", reason: str = "") -> AgentMovementPlan:
    """Normalise a requested movement intent into a curated plan.

    Returns a ``blocked`` plan (no side effects, ``requires_live_execute=False``) for any
    intent outside the allowlist — never raises. ``stop_motion`` maps to the halt seam
    (``clear_move_queue``); the look_* intents map to ``agent_safe_head_motion`` with a
    bounded direction.
    """
    _ = source
    normalized = (intent or "").strip().lower()
    try:
        parsed_intent = AgentMovementIntent(normalized)
    except ValueError:
        return AgentMovementPlan(
            status="blocked",
            intent=normalized,
            selected_tool=None,
            tool_args={},
            reason=reason,
            side_effects=[],
            requires_live_execute=False,
        )

    if parsed_intent == AgentMovementIntent.STOP_MOTION:
        return AgentMovementPlan(
            status="planned",
            intent=parsed_intent.value,
            selected_tool="clear_move_queue",
            tool_args={},
            reason=reason,
            side_effects=[],
            requires_live_execute=True,
        )

    return AgentMovementPlan(
        status="planned",
        intent=parsed_intent.value,
        selected_tool="agent_safe_head_motion",
        tool_args={"direction": _INTENT_TO_DIRECTION[parsed_intent]},
        reason=reason,
        side_effects=[],
        requires_live_execute=True,
    )
