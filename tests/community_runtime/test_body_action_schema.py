import pytest

from reachy_agent.runtime.core.body import BodyAction, BodyActionError, BodyActionResult


def test_stop_action_accepts_minimal_fields():
    action = BodyAction(action=" stop ")
    assert action.action == "stop"
    assert action.intensity == 1.0


def test_unknown_action_rejects():
    with pytest.raises(BodyActionError, match="unknown body action"):
        BodyAction(action="backflip")


def test_intensity_is_clamped():
    assert BodyAction(action="chirp", intensity=2.5).intensity == 1.0
    assert BodyAction(action="chirp", intensity=-1).intensity == 0.0


def test_negative_duration_rejects():
    with pytest.raises(BodyActionError, match="duration_s"):
        BodyAction(action="look", direction="left", duration_s=-0.1)


def test_direction_is_validated_and_normalized():
    assert BodyAction(action="look", direction=" LEFT ").direction == "left"
    assert BodyAction(action="look", direction="front").direction == "front"
    with pytest.raises(BodyActionError, match="unknown body direction"):
        BodyAction(action="look", direction="orbit")


def test_emotion_is_first_class_and_normalized():
    assert BodyAction(action="emote", emotion=" HAPPY ").emotion == "happy"
    with pytest.raises(BodyActionError, match="emotion must not be empty"):
        BodyAction(action="emote", emotion="  ")


def test_metadata_must_be_simple_values():
    assert BodyAction(action="chirp", metadata={"count": 1, "ok": True}).metadata["count"] == 1
    with pytest.raises(BodyActionError, match="metadata values"):
        BodyAction(action="chirp", metadata={"nested": {"bad": True}})


def test_result_preserves_turn_id_and_sanitized_readback():
    result = BodyActionResult(ok=True, action="CHIRP", turn_id="turn-1", readback={"mock": True})
    assert result.action == "chirp"
    assert result.turn_id == "turn-1"
    assert result.readback == {"mock": True}
