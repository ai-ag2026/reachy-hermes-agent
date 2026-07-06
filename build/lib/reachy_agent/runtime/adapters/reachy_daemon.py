"""Reachy daemon body adapter with no live defaults.

This adapter translates the neutral body contracts to the Reachy daemon HTTP API. It is safe for tests because the HTTP client is injectable and all live behavior remains gated by ``SafeMovementPolicy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import httpx

from reachy_agent.runtime.core.body import BodyAction, BodyActionError, BodyActionResult, BodyStatus
from reachy_agent.runtime.core.safety import SafeMovementPolicy, SafetyConfig

_DIRECTION_TO_DELTA = {
    "left": ("yaw", 1.0),
    "right": ("yaw", -1.0),
    "center": ("yaw", 0.0),
    "front": ("yaw", 0.0),
    "up": ("pitch", 1.0),
    "down": ("pitch", -1.0),
}


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float | str):
        return float(value)
    raise BodyActionError("movement bound metadata must be numeric")


@dataclass
class ReachyDaemonBodyController:
    """HTTP adapter for Reachy daemon body actions.

    Defaults are deliberately conservative: body actions are disabled unless a caller passes an enabling ``SafetyConfig``. Tests should inject an ``httpx.AsyncClient`` with ``MockTransport``.
    """

    base_url: str
    client: httpx.AsyncClient | None = None
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sound_map: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float = 3.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._policy = SafeMovementPolicy(self.safety)
        self._owns_client = self.client is None
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout_s)

    async def aclose(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    async def execute(self, action: BodyAction) -> BodyActionResult:
        status = await self.status()
        if action.action == "head_tracking":
            raise BodyActionError(
                "daemon head_tracking is explicitly parked until a generic daemon API contract exists"
            )
        safe = self._policy.validate(action, status)
        if safe.action == "stop":
            return await self.stop("stop action")
        if safe.action == "chirp":
            return await self._chirp(safe)
        if safe.action == "look":
            return await self._look(safe)
        raise BodyActionError(f"Reachy daemon adapter does not implement action: {safe.action}")

    async def stop(self, reason: str | None = None) -> BodyActionResult:
        await self._request("POST", "/api/move/stop", json={})
        running = await self._get_json("/api/move/running")
        running_count = len(running) if isinstance(running, list) else 0
        return BodyActionResult(
            ok=True,
            action="stop",
            message=reason or "daemon stop requested",
            stopped=True,
            readback={"running_actions": running_count},
        )

    async def status(self) -> BodyStatus:
        daemon = await self._get_json("/api/daemon/status")
        motors = await self._get_json("/api/motors/status")
        running = await self._get_json("/api/move/running")
        media = await self._get_json("/api/media/status")

        daemon_ok = isinstance(daemon, dict) and daemon.get("status") == "running"
        error = daemon.get("error") if isinstance(daemon, dict) else None
        motor_mode = motors.get("mode") if isinstance(motors, dict) else None
        motors_enabled = motor_mode == "enabled" if motor_mode is not None else None
        current_actions = tuple(str(item) for item in running) if isinstance(running, list) else ()
        media_available = bool(media.get("available")) if isinstance(media, dict) else None

        return BodyStatus(
            available=daemon_ok and not error,
            motors_enabled=motors_enabled,
            current_actions=current_actions,
            media_available=media_available,
            error=str(error) if error else None,
        )

    async def _chirp(self, action: BodyAction) -> BodyActionResult:
        if not action.name:
            raise BodyActionError("chirp action requires name")
        file_name = self.sound_map.get(action.name)
        if not file_name:
            raise BodyActionError("chirp name is not configured")
        response = await self._get_json("/api/media/play_sound", method="POST", json={"file": file_name})
        status = response.get("status") if isinstance(response, dict) else None
        ok = status == "ok"
        return BodyActionResult(
            ok=ok,
            action="chirp",
            turn_id=action.turn_id,
            message="daemon sound accepted" if ok else "daemon sound failed",
            readback={"daemon_status": str(status), "file": file_name},
        )

    async def _look(self, action: BodyAction) -> BodyActionResult:
        if action.direction is None:
            raise BodyActionError("look action requires direction")
        axis, sign = _DIRECTION_TO_DELTA[action.direction]
        max_yaw = _as_float(action.metadata.get("max_yaw_delta"), self.safety.max_yaw_delta)
        max_pitch = _as_float(action.metadata.get("max_pitch_delta"), self.safety.max_pitch_delta)
        payload: dict[str, float] = {"duration": action.duration_s or self.safety.max_action_seconds}
        if axis == "yaw":
            payload["yaw"] = sign * max_yaw * action.intensity
        else:
            payload["pitch"] = sign * max_pitch * action.intensity
        response = await self._get_json("/api/move/goto", method="POST", json=payload)
        return BodyActionResult(
            ok=True,
            action="look",
            turn_id=action.turn_id,
            message="daemon move accepted",
            readback={"daemon_response": str(response.get("uuid") if isinstance(response, dict) else response)},
        )

    async def _get_json(self, path: str, *, method: str = "GET", json: object | None = None) -> object:
        response = await self._request(method, path, json=json)
        return response.json()

    async def _request(self, method: str, path: str, *, json: object | None = None) -> httpx.Response:
        assert self.client is not None
        response = await self.client.request(method, f"{self.base_url}{path}", json=json)
        response.raise_for_status()
        return response
