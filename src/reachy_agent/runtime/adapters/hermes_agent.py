"""Optional Hermes/AGENT adapter for neutral body actions.

This module does not import Hermes runtime code. It receives an injected async callable that represents the private/local ``reachy_body`` tool surface.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Mapping

from reachy_agent.runtime.core.body import BodyAction, BodyActionError, BodyActionResult, BodyStatus

ReachyBodyResponse = Mapping[str, object] | str
ReachyBodyCall = Callable[[dict[str, object]], Awaitable[ReachyBodyResponse]]


@dataclass
class HermesAgentBodyController:
    """Map neutral body actions to the private/local ``reachy_body`` tool schema."""

    call_reachy_body: ReachyBodyCall

    async def execute(self, action: BodyAction) -> BodyActionResult:
        if action.action == "stop":
            return await self.stop()
        payload = self._to_payload(action)
        response = _normalize_response(await self.call_reachy_body(payload))
        return self._result_from_response(action, response)

    async def stop(self, reason: str | None = None) -> BodyActionResult:
        payload: dict[str, object] = {"action": "stop"}
        if reason:
            payload["reason"] = reason
        response = _normalize_response(await self.call_reachy_body(payload))
        return BodyActionResult(
            ok=_response_ok(response),
            action="stop",
            message=_message_from_response(response),
            stopped=bool(response.get("stopped", True)),
            readback=_clean_readback(response.get("readback")),
        )

    async def status(self) -> BodyStatus:
        return BodyStatus(available=False, error="status is not exposed by the injected reachy_body callable")

    def _to_payload(self, action: BodyAction) -> dict[str, object]:
        if action.action == "chirp" and not action.name:
            raise BodyActionError("chirp action requires name")

        payload: dict[str, object] = {"action": action.action}
        if action.turn_id is not None:
            payload["turn_id"] = action.turn_id
        if action.name is not None:
            payload["name"] = action.name
        if action.emotion is not None:
            payload["emotion"] = action.emotion
        if action.direction is not None:
            payload["direction"] = _to_reachy_body_direction(action.direction)
        if action.intensity != 1.0:
            payload["intensity"] = action.intensity
        if action.duration_s is not None:
            payload["duration_s"] = action.duration_s
        if action.enabled is not None:
            payload["enabled"] = action.enabled
        if action.metadata:
            payload["metadata"] = dict(action.metadata)
        return payload

    def _result_from_response(self, action: BodyAction, response: Mapping[str, object]) -> BodyActionResult:
        return BodyActionResult(
            ok=_response_ok(response),
            action=action.action,
            turn_id=action.turn_id,
            message=_message_from_response(response),
            stopped=bool(response.get("stopped", False)),
            readback=_clean_readback(response.get("readback")),
        )


def _normalize_response(response: ReachyBodyResponse) -> Mapping[str, object]:
    if isinstance(response, str):
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError as exc:
            raise BodyActionError("invalid JSON response from reachy_body callable") from exc
        if not isinstance(decoded, dict):
            raise BodyActionError("reachy_body JSON response must be an object")
        return decoded
    return response


def _response_ok(response: Mapping[str, object]) -> bool:
    if response.get("error"):
        return False
    return bool(response.get("ok", True))


def _message_from_response(response: Mapping[str, object]) -> str | None:
    if response.get("message") is not None:
        return _optional_str(response.get("message"))
    if response.get("error") is not None:
        return _optional_str(response.get("error"))
    return None


def _to_reachy_body_direction(direction: str) -> str:
    if direction == "center":
        return "front"
    return direction


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _clean_readback(value: object) -> dict[str, str | int | float | bool]:
    if not isinstance(value, Mapping):
        return {}
    clean: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str | int | float | bool):
            clean[key] = item
    return clean
