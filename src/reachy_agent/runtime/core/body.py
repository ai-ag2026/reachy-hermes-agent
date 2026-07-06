"""Neutral body-action contracts for Reachy agent runtimes.

No agent-framework imports, no robot SDK, no network. Adapters translate these models to real systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

ALLOWED_ACTIONS = frozenset({"chirp", "look", "emote", "dance", "stop", "head_tracking"})
ALLOWED_DIRECTIONS = frozenset({"left", "right", "center", "front", "up", "down"})
_SIMPLE_METADATA_TYPES = (str, int, float, bool)


class BodyActionError(ValueError):
    """Raised when a body action is invalid before any adapter can execute it."""


def _validate_metadata(metadata: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise BodyActionError("metadata keys must be non-empty strings")
        if not isinstance(value, _SIMPLE_METADATA_TYPES):
            raise BodyActionError("metadata values must be strings, numbers or booleans")
        clean[key] = value
    return clean


@dataclass(frozen=True)
class BodyAction:
    """A neutral request for a robot/avatar body cue.

    The model is deliberately small and serializable. It does not imply live hardware,
    camera/mic use, audible playback or persistence.
    """

    action: str
    turn_id: str | None = None
    name: str | None = None
    emotion: str | None = None
    direction: str | None = None
    intensity: float = 1.0
    duration_s: float | None = None
    enabled: bool | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = (self.action or "").strip().lower()
        if normalized not in ALLOWED_ACTIONS:
            raise BodyActionError(f"unknown body action: {self.action!r}")
        object.__setattr__(self, "action", normalized)

        if self.direction is not None:
            direction = self.direction.strip().lower()
            if direction not in ALLOWED_DIRECTIONS:
                raise BodyActionError(f"unknown body direction: {self.direction!r}")
            object.__setattr__(self, "direction", direction)

        intensity = float(self.intensity)
        if intensity < 0.0:
            intensity = 0.0
        if intensity > 1.0:
            intensity = 1.0
        object.__setattr__(self, "intensity", intensity)

        if self.duration_s is not None and float(self.duration_s) < 0:
            raise BodyActionError("duration_s must not be negative")
        if self.duration_s is not None:
            object.__setattr__(self, "duration_s", float(self.duration_s))

        if self.turn_id is not None and not isinstance(self.turn_id, str):
            raise BodyActionError("turn_id must be a string when provided")
        if self.name is not None and not isinstance(self.name, str):
            raise BodyActionError("name must be a string when provided")
        if self.emotion is not None:
            if not isinstance(self.emotion, str):
                raise BodyActionError("emotion must be a string when provided")
            emotion = self.emotion.strip().lower()
            if not emotion:
                raise BodyActionError("emotion must not be empty when provided")
            object.__setattr__(self, "emotion", emotion)

        object.__setattr__(self, "metadata", _validate_metadata(self.metadata))


@dataclass(frozen=True)
class BodyActionResult:
    """Structured, sanitized body-action result."""

    ok: bool
    action: str
    turn_id: str | None = None
    message: str | None = None
    stopped: bool = False
    readback: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", (self.action or "").strip().lower())
        object.__setattr__(self, "readback", _validate_metadata(self.readback))


@dataclass(frozen=True)
class BodyStatus:
    """Small status surface for policy checks and fake/live readbacks."""

    available: bool
    motors_enabled: bool | None = None
    current_actions: tuple[str, ...] = ()
    media_available: bool | None = None
    error: str | None = None


class BodyController(Protocol):
    """Execution seam implemented by mock or real body adapters."""

    async def execute(self, action: BodyAction) -> BodyActionResult: ...

    async def stop(self, reason: str | None = None) -> BodyActionResult: ...

    async def status(self) -> BodyStatus: ...
