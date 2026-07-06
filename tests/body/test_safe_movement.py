"""Unit tests for the safe_movement executor (pure; no reachy_mini, no hardware).

Uses a FakeMover and an injected numpy create_head_pose so the policy->execution mapping, joint
preservation, the absolute-recenter fix for look_front, stop_motion halting, and the ManagerMover /
SdkMover adapters are all testable in the SSoT venv.
"""

import numpy as np

from reachy_agent.body.safe_movement import (
    ManagerMover,
    apply_safe_movement,
)


def fake_chp(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True):
    """Encode yaw->translation[0], pitch->translation[1] so composition is assertable."""
    m = np.eye(4)
    m[0, 3] = yaw
    m[1, 3] = pitch
    return m


class FakeMover:
    def __init__(self, head=None, body_yaw=0.3, antennas=(0.1, -0.2)):
        self.head = head if head is not None else np.eye(4)
        self.body_yaw = body_yaw
        self.antennas = antennas
        self.queued = []
        self.stopped = 0

    def current_head_pose(self):
        return self.head

    def current_body_yaw_antennas(self):
        return self.body_yaw, self.antennas

    def queue_goto(self, **kw):
        self.queued.append(kw)

    def stop(self):
        self.stopped += 1


def test_look_left_relative_delta_preserves_joints():
    m = FakeMover()
    res = apply_safe_movement(m, "look_left", duration=0.5, head_pose_factory=fake_chp)
    assert res["status"] == "executed" and res["direction"] == "left"
    assert res["bounded_degrees"]["yaw"] == 5
    assert res["preserved"] == ["antennas", "body_yaw"]
    assert m.stopped == 0 and len(m.queued) == 1
    q = m.queued[0]
    # start = identity, delta yaw=5 -> target translation[0]=5 (relative compose)
    assert q["target_head_pose"][0, 3] == 5
    assert q["antennas"] == (0.1, -0.2) and q["body_yaw"] == 0.3
    assert q["duration"] == 0.5


def test_look_down_uses_pitch_delta():
    m = FakeMover()
    apply_safe_movement(m, "look_down", head_pose_factory=fake_chp)
    assert m.queued[0]["target_head_pose"][1, 3] == 4  # down = +4 pitch


def test_look_front_is_absolute_recenter_not_relative_noop():
    # start at a non-neutral pose; front must recenter to absolute neutral (identity), not start.
    start = np.eye(4)
    start[0, 3] = 99  # clearly off-neutral
    m = FakeMover(head=start)
    res = apply_safe_movement(m, "look_front", head_pose_factory=fake_chp)
    assert res["status"] == "executed"
    assert res["bounded_degrees"]["recenter"] == "absolute_neutral"
    np.testing.assert_array_equal(m.queued[0]["target_head_pose"], np.eye(4))  # absolute neutral
    assert m.queued[0]["body_yaw"] == 0.3 and m.queued[0]["antennas"] == (0.1, -0.2)  # preserved


def test_stop_motion_halts_without_queue():
    m = FakeMover()
    res = apply_safe_movement(m, "stop_motion", head_pose_factory=fake_chp)
    assert res["status"] == "executed" and res["side_effects"] == ["movement_stopped"]
    assert m.stopped == 1 and m.queued == []


def test_blocked_intent_does_nothing():
    m = FakeMover()
    res = apply_safe_movement(m, "do_a_backflip", head_pose_factory=fake_chp)
    assert res["status"] == "blocked"
    assert m.stopped == 0 and m.queued == []


class _FakeReachy:
    def __init__(self):
        self.goto_calls = []

    def get_current_head_pose(self):
        return np.eye(4)

    def get_current_joint_positions(self):
        return ([0.25], (0.4, -0.4))  # (body_yaw_list, antennas)

    def goto_target(self, **kw):
        self.goto_calls.append(kw)


def test_sdk_mover_backcompat_raw_reachy():
    r = _FakeReachy()
    res = apply_safe_movement(r, "look_right", duration=0.6, head_pose_factory=fake_chp)
    assert res["status"] == "executed"
    assert len(r.goto_calls) == 1
    c = r.goto_calls[0]
    assert c["head"][0, 3] == -5  # right = -5 yaw
    assert c["body_yaw"] == 0.25 and list(c["antennas"]) == [0.4, -0.4]
    assert c["duration"] == 0.6


def test_manager_mover_enqueues_via_factory_preserving_joints():
    r = _FakeReachy()
    mm_calls = {"queue": [], "moving": []}

    class _MM:
        def queue_move(self, move):
            mm_calls["queue"].append(move)

        def set_moving_state(self, d):
            mm_calls["moving"].append(d)

        def clear_move_queue(self):
            mm_calls.setdefault("cleared", 0)
            mm_calls["cleared"] = mm_calls.get("cleared", 0) + 1

    factory_args = {}

    def fake_goto_factory(**kw):
        factory_args.update(kw)
        return ("GotoQueueMove", kw)

    mover = ManagerMover(r, _MM(), fake_goto_factory)
    res = apply_safe_movement(mover, "look_left", duration=0.7, head_pose_factory=fake_chp)
    assert res["status"] == "executed"
    assert len(mm_calls["queue"]) == 1 and mm_calls["moving"] == [0.7]
    # antennas/body_yaw preserved: start==target on those axes
    assert factory_args["target_antennas"] == factory_args["start_antennas"] == (0.4, -0.4)
    assert factory_args["target_body_yaw"] == factory_args["start_body_yaw"] == 0.25
    assert factory_args["duration"] == 0.7
    assert factory_args["target_head_pose"][0, 3] == 5  # left yaw delta


def test_manager_mover_stop_clears_queue():
    r = _FakeReachy()
    cleared = {"n": 0}

    class _MM:
        def clear_move_queue(self):
            cleared["n"] += 1

    mover = ManagerMover(r, _MM(), lambda **k: None)
    res = apply_safe_movement(mover, "stop_motion", head_pose_factory=fake_chp)
    assert res["status"] == "executed" and cleared["n"] == 1


# --- audit 2026-07-02 hardening ------------------------------------------------


def test_mover_exception_returns_error_status_not_raise():
    class _Boom(FakeMover):
        def current_head_pose(self):
            raise ConnectionError("daemon gone")

    res = apply_safe_movement(_Boom(), "look_left", head_pose_factory=fake_chp)
    assert res["status"] == "error"
    assert "daemon gone" in res["error"]
    assert res["side_effects"] == []


def test_stop_exception_returns_error_status():
    class _Boom(FakeMover):
        def stop(self):
            raise RuntimeError("zenoh timeout")

    res = apply_safe_movement(_Boom(), "stop_motion", head_pose_factory=fake_chp)
    assert res["status"] == "error" and "zenoh timeout" in res["error"]


def test_non_finite_current_pose_aborts_with_error():
    start = np.eye(4)
    start[0, 3] = np.nan
    res = apply_safe_movement(FakeMover(head=start), "look_left", head_pose_factory=fake_chp)
    assert res["status"] == "error" and "non-finite" in res["error"]
    # nothing queued
    assert res["side_effects"] == []


def test_duration_is_clamped():
    for bad, want in [(0.0, 0.15), (-5.0, 0.15), (99.0, 2.0), (float("nan"), 0.3)]:
        m = FakeMover()
        res = apply_safe_movement(m, "look_left", duration=bad, head_pose_factory=fake_chp)
        assert res["status"] == "executed"
        assert m.queued[0]["duration"] == want
