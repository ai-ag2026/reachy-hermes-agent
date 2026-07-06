import pytest

from reachy_agent.runtime.core.body import BodyAction, BodyActionError, BodyStatus
from reachy_agent.runtime.core.safety import SafeMovementPolicy, SafetyConfig


def test_body_disabled_rejects_non_stop_actions():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=False))
    with pytest.raises(BodyActionError, match="disabled"):
        policy.validate(BodyAction(action="chirp"), BodyStatus(available=False))


def test_stop_allowed_when_body_disabled():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=False))
    action = BodyAction(action="stop")
    assert policy.validate(action, BodyStatus(available=True)) is action


def test_live_movement_disabled_rejects_available_real_body_movement():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=True, live_movement_enabled=False))
    with pytest.raises(BodyActionError, match="live movement"):
        policy.validate(BodyAction(action="look", direction="left"), BodyStatus(available=True))


def test_live_movement_disabled_still_allows_fake_unavailable_body_for_ci():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=True, live_movement_enabled=False))
    safe = policy.validate(BodyAction(action="look", direction="left"), BodyStatus(available=False))
    assert safe.metadata["max_yaw_delta"] == 0.20
    assert safe.metadata["max_pitch_delta"] == 0.12


def test_duration_above_max_rejects():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=True, max_action_seconds=1.0))
    with pytest.raises(BodyActionError, match="max_action_seconds"):
        policy.validate(BodyAction(action="look", direction="left", duration_s=1.5), BodyStatus(available=False))


def test_look_requires_direction():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=True))
    with pytest.raises(BodyActionError, match="requires direction"):
        policy.validate(BodyAction(action="look"), BodyStatus(available=False))


def test_enabled_live_movement_accepts_available_body_look():
    policy = SafeMovementPolicy(SafetyConfig(body_enabled=True, live_movement_enabled=True))
    action = policy.validate(BodyAction(action="look", direction="center"), BodyStatus(available=True))
    assert action.direction == "center"
