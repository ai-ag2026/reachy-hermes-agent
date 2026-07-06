"""Safe, no-hardware movement/body action policy for public-candidate runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .body import BodyAction, BodyActionError, BodyStatus

MOVEMENT_LIKE_ACTIONS = frozenset({"look", "dance", "head_tracking"})


@dataclass(frozen=True)
class SafetyConfig:
    """Conservative defaults for public examples and CI."""

    body_enabled: bool = False
    live_movement_enabled: bool = False
    max_action_seconds: float = 2.0
    max_yaw_delta: float = 0.20
    max_pitch_delta: float = 0.12
    require_readback: bool = True


class SafeMovementPolicy:
    """Validate/clamp body actions before any adapter can touch hardware."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def validate(self, action: BodyAction, status: BodyStatus | None = None) -> BodyAction:
        """Return a safe action or raise ``BodyActionError``.

        ``stop`` is always allowed. Every other action requires ``body_enabled``.
        Movement-like actions also require ``live_movement_enabled`` when the status
        indicates a real/available body. With mock/unavailable bodies, tests can still
        validate schema behavior without implying hardware execution.
        """
        if action.action == "stop":
            return action

        if not self.config.body_enabled:
            raise BodyActionError("body actions are disabled")

        if action.duration_s is not None and action.duration_s > self.config.max_action_seconds:
            raise BodyActionError("duration_s exceeds max_action_seconds")

        body_available = bool(status.available) if status is not None else False
        if action.action in MOVEMENT_LIKE_ACTIONS and body_available and not self.config.live_movement_enabled:
            raise BodyActionError("live movement is disabled")

        if action.action == "look":
            if action.direction is None:
                raise BodyActionError("look action requires direction")
            # Keep neutral core policy simple: expose the configured bounds in metadata for adapters.
            metadata = dict(action.metadata)
            metadata.setdefault("max_yaw_delta", self.config.max_yaw_delta)
            metadata.setdefault("max_pitch_delta", self.config.max_pitch_delta)
            return replace(action, metadata=metadata)

        return action
