"""Tests for the parallel vision-context seam (pure, no robot/network)."""

from __future__ import annotations

from reachy_agent.voice.vision_context import is_visual_query, scene_context


def test_is_visual_query_triggers_on_visual_intent() -> None:
    for q in [
        "Was siehst du?",
        "Welche Farbe hat das?",
        "Wie viele Objekte sind da?",
        "Schau mal nach vorne.",
        "What do you see?",
        "How many cups are on the table?",
        "Was ist das hier?",
    ]:
        assert is_visual_query(q), q


def test_is_visual_query_ignores_non_visual() -> None:
    for q in [
        "Erzähl mir etwas über schwarze Löcher.",
        "Wie spät ist es?",
        "Mach weiter.",
        "Was hältst du von Politik?",  # 'hältst' without sight intent stays out (no 'in der hand')
        "",
    ]:
        assert not is_visual_query(q), q


def test_scene_context_visual_returns_labeled_context() -> None:
    calls = []

    def fake_describe(frame, prompt):
        calls.append((frame, prompt))
        return "Ein roter Becher auf einem Holztisch."

    out = scene_context("Welche Farbe hat der Becher?", frame=object(), describe=fake_describe)
    assert out is not None
    assert "roter Becher" in out
    assert out.startswith("[Was du gerade durch deine Kamera siehst:")
    assert len(calls) == 1


def test_scene_context_non_visual_skips_describe() -> None:
    called = False

    def fake_describe(frame, prompt):
        nonlocal called
        called = True
        return "x"

    assert scene_context("Erzähl mir einen Witz.", frame=object(), describe=fake_describe) is None
    assert called is False  # gate prevents a wasted VLM call


def test_scene_context_best_effort_on_failure_and_empty() -> None:
    def raising(frame, prompt):
        raise RuntimeError("vlm down")

    assert scene_context("Was siehst du?", frame=object(), describe=raising) is None
    assert scene_context("Was siehst du?", frame=object(), describe=lambda f, p: "  ") is None
    assert scene_context("Was siehst du?", frame=None, describe=lambda f, p: "x") is None


def test_scene_context_force_bypasses_gate_and_truncates() -> None:
    long = "A" * 1000
    out = scene_context("irgendwas", frame=object(), describe=lambda f, p: long, force=True)
    assert out is not None and out.endswith("…]")
    assert len(out) < 1000  # description hard-capped (_MAX_CONTEXT_CHARS) + short label


def test_vision_mode_routing() -> None:
    from reachy_agent.voice.vision_context import vision_mode

    assert vision_mode("Erzähl mir einen Witz.") == "none"
    assert vision_mode("Was siehst du gerade?") == "describe"  # simple still -> gemma
    assert vision_mode("Wie viele Tassen stehen da?") == "native"  # counting -> the agent native
    assert vision_mode("Vergleiche die zwei Objekte.") == "native"
    assert vision_mode("Lies mal vor, was da steht.") == "native"
    assert vision_mode("Was passiert gerade?") == "video"  # motion -> gemma video
    assert vision_mode("Schau mal zu, was sich bewegt.") == "video"


def test_person_identification_is_now_visual_not_blocked() -> None:
    # Person/face questions are valid visual turns now (route to vision, not blocked).
    from reachy_agent.voice.vision_context import is_visual_query

    assert is_visual_query("Wer ist das vor dir?")  # 'wer'+'vor dir' -> visual
