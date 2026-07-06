"""Minimal local S2S event helpers for the AGENT/Reachy MVP path.

The local S2S lane intentionally mirrors the small Realtime-style surface we need
for the first embodied AGENT value proof. It is not a robot controller and does
not touch audio devices, cameras, motors, or network services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class LocalS2SEventType(StrEnum):
    """Supported local S2S events for the MVP bridge."""

    TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_CANCELLED = "response.cancelled"


@dataclass(frozen=True)
class LocalS2SEvent:
    """A validated local S2S event payload.

    `payload` holds the event-specific fields. Keeping the top-level shape small
    makes the first bridge smokes deterministic while still allowing future
    adapters to preserve source-specific metadata under `payload`.
    """

    type: LocalS2SEventType
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return an OpenAI/HF-realtime-like dictionary payload."""
        return {"type": self.type.value, **dict(self.payload)}


def parse_local_s2s_event(event: Mapping[str, Any]) -> LocalS2SEvent:
    """Parse and validate a local S2S event dictionary."""
    raw_type = event.get("type")
    if not isinstance(raw_type, str) or not raw_type:
        raise ValueError("local S2S event requires a non-empty string 'type'")
    try:
        event_type = LocalS2SEventType(raw_type)
    except ValueError as exc:
        raise ValueError(f"unsupported local S2S event type: {raw_type}") from exc

    payload = {key: value for key, value in event.items() if key != "type"}
    _validate_payload(event_type, payload)
    return LocalS2SEvent(type=event_type, payload=payload)


def make_transcription_completed(*, transcript: str, item_id: str | None = None) -> LocalS2SEvent:
    """Create a completed-input-transcription event."""
    payload: dict[str, Any] = {"transcript": _require_non_empty_string(transcript, "transcript")}
    if item_id is not None:
        payload["item_id"] = _require_non_empty_string(item_id, "item_id")
    return LocalS2SEvent(LocalS2SEventType.TRANSCRIPTION_COMPLETED, payload)


def make_output_text_delta(*, delta: str, response_id: str | None = None) -> LocalS2SEvent:
    """Create a streamed output-text delta event."""
    payload: dict[str, Any] = {"delta": _require_string(delta, "delta")}
    if response_id is not None:
        payload["response_id"] = _require_non_empty_string(response_id, "response_id")
    return LocalS2SEvent(LocalS2SEventType.OUTPUT_TEXT_DELTA, payload)


def make_response_completed(*, response_id: str | None = None) -> LocalS2SEvent:
    """Create a response-completed event."""
    payload: dict[str, Any] = {}
    if response_id is not None:
        payload["response_id"] = _require_non_empty_string(response_id, "response_id")
    return LocalS2SEvent(LocalS2SEventType.RESPONSE_COMPLETED, payload)


def make_response_cancelled(*, reason: str | None = None, response_id: str | None = None) -> LocalS2SEvent:
    """Create a response-cancelled event."""
    payload: dict[str, Any] = {}
    if reason is not None:
        payload["reason"] = _require_non_empty_string(reason, "reason")
    if response_id is not None:
        payload["response_id"] = _require_non_empty_string(response_id, "response_id")
    return LocalS2SEvent(LocalS2SEventType.RESPONSE_CANCELLED, payload)


def _validate_payload(event_type: LocalS2SEventType, payload: Mapping[str, Any]) -> None:
    if event_type == LocalS2SEventType.TRANSCRIPTION_COMPLETED:
        _require_non_empty_string(payload.get("transcript"), "transcript")
        if "item_id" in payload:
            _require_non_empty_string(payload.get("item_id"), "item_id")
        return
    if event_type == LocalS2SEventType.OUTPUT_TEXT_DELTA:
        _require_string(payload.get("delta"), "delta")
        if "response_id" in payload:
            _require_non_empty_string(payload.get("response_id"), "response_id")
        return
    if event_type == LocalS2SEventType.RESPONSE_COMPLETED:
        if "response_id" in payload:
            _require_non_empty_string(payload.get("response_id"), "response_id")
        return
    if event_type == LocalS2SEventType.RESPONSE_CANCELLED:
        if "reason" in payload:
            _require_non_empty_string(payload.get("reason"), "reason")
        if "response_id" in payload:
            _require_non_empty_string(payload.get("response_id"), "response_id")
        return
    raise ValueError(f"unsupported local S2S event type: {event_type}")


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    text = _require_string(value, field_name)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text
