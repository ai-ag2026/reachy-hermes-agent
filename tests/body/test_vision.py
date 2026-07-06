"""Unit tests for the one-shot vision safety wrapper (fake capture/describe seams)."""

import numpy as np

from reachy_agent.body.vision import one_shot_vision, vision_policy


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _describe_ok(frame, question):
    return f"A {frame.shape[1]}x{frame.shape[0]} scene; neutral lighting."


def test_person_identification_allowed_by_default():
    # the operator: local model, personal robot -> person/face recognition is allowed by default.
    for q in ["Who is this?", "Wer steht da?", "What is the person's name?", "identify the face"]:
        res = one_shot_vision(q, capture_frame=_frame, describe=_describe_ok)
        assert res["status"] == "ok", q
        assert res["raw_image_returned"] is False  # still never leaks the raw image


def test_person_identification_blocked_when_env_set(monkeypatch):
    monkeypatch.setenv("AGENT_VISION_BLOCK_PERSON_ID", "1")
    captured = {"n": 0}

    def capture():
        captured["n"] += 1
        return _frame()

    res = one_shot_vision("Wer ist das?", capture_frame=capture, describe=_describe_ok)
    assert res["status"] == "blocked"
    assert captured["n"] == 0  # blocked before any capture


def test_safe_question_returns_description_without_raw_image():
    res = one_shot_vision("What colours are visible?", capture_frame=_frame, describe=_describe_ok)
    assert res["status"] == "ok"
    assert "image_description" in res and isinstance(res["image_description"], str)
    assert res["frame_shape"] == [48, 64, 3]
    assert res["image_persisted"] is False
    assert res["raw_image_returned"] is False
    assert "image" not in {k for k in res if k in ("image", "frame", "raw_image")}


def test_no_frame_and_empty_question_are_errors():
    assert one_shot_vision("", capture_frame=_frame, describe=_describe_ok)["status"] == "error"
    assert one_shot_vision("What is there?", capture_frame=lambda: None, describe=_describe_ok)["status"] == "error"


def test_describe_failure_is_handled_and_no_raw_leak():
    def boom(frame, q):
        raise RuntimeError("vlm down")

    res = one_shot_vision("Describe the shapes", capture_frame=_frame, describe=boom)
    assert res["status"] == "error"
    assert res["image_persisted"] is False and res["raw_image_returned"] is False


def test_policy_allows_everything_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_VISION_BLOCK_PERSON_ID", raising=False)
    assert vision_policy("How many objects are on the table?")["blocked"] is False
    assert vision_policy("Wer ist das?")["blocked"] is False  # person-ID allowed by default
    monkeypatch.setenv("AGENT_VISION_BLOCK_PERSON_ID", "1")
    assert vision_policy("Wer ist das?")["blocked"] is True  # legacy denylist re-enabled via env
