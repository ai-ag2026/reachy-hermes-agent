"""Lexicon transcript correction — proper-noun mishearing fixes."""

import pytest

from reachy_agent.voice.lexicon import correct_transcript


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        # the default map only fixes the robot's own product name
        ("Ricci, dreh deinen Kopf nach links.", "Reachy, dreh deinen Kopf nach links."),
        ("Hey Ritchie, wach auf!", "Hey Reachy, wach auf!"),
        # case-insensitive match, canonical casing out
        ("RICHIE und ricci sind wach.", "Reachy und Reachy sind wach."),
    ],
)
def test_known_mishearings_corrected(heard, expected):
    assert correct_transcript(heard) == expected


def test_whole_word_only_no_substring_damage():
    # ordinary words are never touched; a default alias only matches as a whole word.
    assert correct_transcript("Das Thema bleibt unverändert.") == "Das Thema bleibt unverändert."
    assert correct_transcript("Reachyerweiterung") == "Reachyerweiterung"


def test_correct_names_pass_through():
    assert correct_transcript("Frag Reachy nach dem Status.") == "Frag Reachy nach dem Status."


def test_empty_and_none_safe():
    assert correct_transcript("") == ""


def test_env_aliases_extend_and_override(monkeypatch):
    # users add their own persona/household names without code, via env
    monkeypatch.setenv("AGENT_STT_ALIASES", "thars,thas=Assistant; amalon=Amalun")
    assert (
        correct_transcript("Hallo Thars, Status vom Amalon-Projekt?")
        == "Hallo Assistant, Status vom Amalun-Projekt?"
    )
    # the default (Reachy) map stays active alongside env entries
    assert correct_transcript("Ricci schläft.") == "Reachy schläft."


def test_exotic_casefold_input_does_not_raise(monkeypatch):
    """An IGNORECASE match whose .lower() is not the alias key (exotic Unicode
    case-folding, e.g. ſ long-s) must not raise or drop the utterance."""
    monkeypatch.delenv("AGENT_STT_ALIASES", raising=False)
    out = correct_transcript("hallo ricſi und ricci")
    assert "Reachy" in out  # the normal alias still corrects
