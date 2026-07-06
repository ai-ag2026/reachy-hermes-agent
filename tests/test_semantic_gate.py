"""Unit tests for the semantic barge-in gate (pure heuristic + injected helper-LLM)."""

from __future__ import annotations

import pytest

from reachy_agent.voice.semantic_gate import (
    barge_decision,
    classify_interrupt,
    is_bare_stop,
    semantic_decision,
)


@pytest.mark.parametrize("text", ["", "   ", None, "ja", "okay", "mhm", "genau", "ja ja", "achso", "hm"])
def test_backchannel_and_empty_resume(text):
    assert barge_decision(text) == "resume"


@pytest.mark.parametrize("text", ["a", "ok", "huh", "yo"])
def test_very_short_resume(text):
    # length <= 3 chars after stripping punctuation -> resume
    assert barge_decision(text) == "resume"


@pytest.mark.parametrize("text", ["stopp", "halt", "nein", "warte", "moment"])
def test_lone_command_word_commits(text):
    assert barge_decision(text) == "commit"


@pytest.mark.parametrize("text", ["Quatsch", "trotzdem", "vielleicht"])
def test_lone_noncommand_word_resumes(text):
    # a single non-command word is most likely backchannel / noise
    assert barge_decision(text) == "resume"


@pytest.mark.parametrize(
    "text",
    [
        "warte mal kurz",
        "nein das stimmt nicht",
        "erzähl mir lieber etwas anderes",
        "wie spät ist es",
    ],
)
def test_substantive_utterance_commits(text):
    assert barge_decision(text) == "commit"


def test_trailing_punctuation_ignored():
    assert barge_decision("ja.") == "resume"
    assert barge_decision("stopp!") == "commit"


# --- semantic_decision: heuristic short-circuit (no LLM call) ---


def test_semantic_resume_shortcircuits_without_llm():
    calls = []

    def never(_t):  # must NOT be called for a heuristic-resume
        calls.append(_t)
        return "STOP"

    assert semantic_decision("mhm", ask_stop_continue=never) == "resume"
    assert calls == []


# --- semantic_decision: helper-LLM stage ---


def test_semantic_llm_weiter_resumes():
    assert semantic_decision("ach das meinst du", ask_stop_continue=lambda _t: "WEITER") == "resume"


def test_semantic_llm_stop_commits():
    assert semantic_decision("nein das ist falsch", ask_stop_continue=lambda _t: "STOP") == "commit"


def test_semantic_llm_ambiguous_commits():
    # neither a clean WEITER -> default to commit (never drop a real interrupt)
    assert semantic_decision("warte mal eben", ask_stop_continue=lambda _t: "hmm unklar") == "commit"


def test_semantic_llm_error_defaults_commit():
    def boom(_t):
        raise RuntimeError("network down")

    assert semantic_decision("erklär das nochmal anders", ask_stop_continue=boom) == "commit"


def test_semantic_llm_case_insensitive():
    assert semantic_decision("ja aber warte", ask_stop_continue=lambda _t: "weiter") == "resume"


# --- is_bare_stop ---


@pytest.mark.parametrize(
    "text", ["stopp", "stop", "halt", "stopp bitte", "halt mal", "stopp jetzt", "sei still", "ruhe"]
)
def test_bare_stop_true(text):
    assert is_bare_stop(text) is True


@pytest.mark.parametrize(
    "text",
    ["stopp wechsle das thema", "halt, erzähl was anderes", "nein das stimmt nicht", "wie spät ist es", "", "ja"],
)
def test_bare_stop_false(text):
    assert is_bare_stop(text) is False


# --- classify_interrupt (3-way) ---


@pytest.mark.parametrize("text", ["ja", "mhm", "okay", "genau", "", "achso"])
def test_classify_backchannel_ignore(text):
    # heuristic resume -> ignore, no LLM
    assert classify_interrupt(text, ask_classify=lambda _t: "INHALT") == "ignore"


@pytest.mark.parametrize("text", ["stopp", "halt mal", "stopp bitte", "ruhe"])
def test_classify_bare_stop(text):
    # bare stop -> stop, no LLM
    assert classify_interrupt(text, ask_classify=lambda _t: "INHALT") == "stop"


def test_classify_stop_with_content_commits_via_llm():
    assert classify_interrupt("stopp, wechsle das Thema zu Katzen", ask_classify=lambda _t: "INHALT") == "commit"


def test_classify_new_question_commits():
    assert classify_interrupt("erzähl mir lieber etwas über Katzen", ask_classify=lambda _t: "INHALT") == "commit"


def test_classify_llm_weiter_ignores():
    assert classify_interrupt("hintergrund gemurmel hier", ask_classify=lambda _t: "WEITER") == "ignore"


def test_classify_llm_stop_only():
    assert classify_interrupt("jetzt ist aber wirklich genug", ask_classify=lambda _t: "STOP") == "stop"


def test_classify_llm_error_defaults_commit():
    def boom(_t):
        raise RuntimeError("down")

    assert classify_interrupt("erklär das nochmal anders", ask_classify=boom) == "commit"


def test_backchannel_with_inner_punctuation_resumes():
    """Review 2026-07-02 round 2, P3: 'Ja, ja.' fell through to commit because only edge
    punctuation was stripped before the backchannel set lookup."""
    from reachy_agent.voice.semantic_gate import barge_decision

    assert barge_decision("Ja, ja.") == "resume"
    assert barge_decision("ok, ok") == "resume"
    assert barge_decision("mhm, mhm.") == "resume"
    # substantive utterances still commit
    assert barge_decision("nein warte, nimm die andere Datei") == "commit"
