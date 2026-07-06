from __future__ import annotations

import pytest

from reachy_agent.local_s2s.events import (
    LocalS2SEvent,
    LocalS2SEventType,
    make_output_text_delta,
    make_response_cancelled,
    make_response_completed,
    make_transcription_completed,
    parse_local_s2s_event,
)


def test_builds_transcription_completed_event() -> None:
    """Build the required completed-transcription event shape."""
    event = make_transcription_completed(item_id="item-1", transcript="Was bist du?")

    assert event == LocalS2SEvent(
        type=LocalS2SEventType.TRANSCRIPTION_COMPLETED,
        payload={"item_id": "item-1", "transcript": "Was bist du?"},
    )
    assert event.to_dict() == {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-1",
        "transcript": "Was bist du?",
    }


def test_builds_output_text_delta_event() -> None:
    """Build the required streamed text-delta event shape."""
    event = make_output_text_delta(response_id="resp-1", delta="Ich bin AGENT")

    assert event.to_dict() == {
        "type": "response.output_text.delta",
        "response_id": "resp-1",
        "delta": "Ich bin AGENT",
    }


def test_output_text_delta_allows_empty_flush_delta() -> None:
    """Allow empty text deltas for flush/heartbeat-style chunks."""
    event = make_output_text_delta(delta="")

    assert event.to_dict() == {"type": "response.output_text.delta", "delta": ""}
    assert parse_local_s2s_event(event.to_dict()) == event


def test_builds_response_terminal_events() -> None:
    """Build response terminal events used by the MVP pipeline."""
    assert make_response_completed(response_id="resp-1").to_dict() == {
        "type": "response.completed",
        "response_id": "resp-1",
    }
    assert make_response_cancelled(response_id="resp-1", reason="barge_in").to_dict() == {
        "type": "response.cancelled",
        "response_id": "resp-1",
        "reason": "barge_in",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "item-1", "transcript": "Hi"},
        {"type": "response.output_text.delta", "response_id": "resp-1", "delta": "Hallo"},
        {"type": "response.completed", "response_id": "resp-1"},
        {"type": "response.cancelled", "response_id": "resp-1", "reason": "barge_in"},
    ],
)
def test_parse_roundtrips_supported_event(payload: dict[str, str]) -> None:
    """Parse and serialize every supported event without shape drift."""
    event = parse_local_s2s_event(payload)

    assert event.to_dict() == payload


@pytest.mark.parametrize(
    "payload,error",
    [
        ({}, "requires"),
        ({"type": "unknown.event"}, "unsupported"),
        ({"type": "conversation.item.input_audio_transcription.completed"}, "transcript"),
        ({"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}, "transcript"),
        ({"type": "response.output_text.delta"}, "delta"),
        ({"type": "response.output_text.delta", "delta": None}, "delta"),
        ({"type": "response.completed", "response_id": ""}, "response_id"),
        ({"type": "response.cancelled", "reason": ""}, "reason"),
    ],
)
def test_parse_rejects_invalid_event(payload: dict[str, object], error: str) -> None:
    """Reject malformed or unsupported event payloads early."""
    with pytest.raises(ValueError, match=error):
        parse_local_s2s_event(payload)
