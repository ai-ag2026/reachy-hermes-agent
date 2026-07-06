"""Deterministic transcript lexicon — fix STT mishearings of proper nouns.

General ASR models have never seen your robot's or household's names, so non-English
turns often come back with the wrong spelling of exactly those proper nouns. No served
model currently offers context biasing, so this is a post-STT correction map applied at
the single STT chokepoint (`AgentSttFrontend._finish_utterance`) — engine-agnostic and cheap.

The default map only fixes the robot's own product name ("Reachy"). Add your own names
without code via env ``AGENT_STT_ALIASES``: semicolon-separated entries, each
``wrong1,wrong2=Correct`` (case-insensitive, whole-word).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

# Only the robot's own product name is fixed by default; it is the one proper noun every
# deployment shares. Deliberately conservative: only phonetic variants implausible as real
# words. Add persona/household names for your own setup via the AGENT_STT_ALIASES env var.
_DEFAULT_ALIASES: dict[str, str] = {
    "ricci": "Reachy",
    "reachie": "Reachy",
    "ritchie": "Reachy",
    "richie": "Reachy",
    "riechie": "Reachy",
}


def _env_aliases(spec: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        wrongs, _, right = entry.partition("=")
        right = right.strip()
        if not right:
            continue
        for wrong in wrongs.split(","):
            wrong = wrong.strip().lower()
            if wrong:
                aliases[wrong] = right
    return aliases


@lru_cache(maxsize=4)
def _compiled(spec: str) -> tuple[re.Pattern[str], dict[str, str]]:
    aliases = dict(_DEFAULT_ALIASES)
    aliases.update(_env_aliases(spec))
    # Longest-first so "reachie" wins over a hypothetical shorter overlap. Lookarounds instead
    # of \b: word boundaries silently never match for aliases with non-word edge characters
    # (".net", "'agent"), which would make such env entries a no-op.
    keys = sorted(aliases, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)",
        re.IGNORECASE,
    )
    return pattern, aliases


def correct_transcript(text: str) -> str:
    """Replace known mishearings (whole-word, case-insensitive); everything else untouched."""
    if not text:
        return text
    pattern, aliases = _compiled(os.getenv("AGENT_STT_ALIASES", ""))
    # .get with the original as fallback: exotic Unicode case-folds (e.g. long s) can match
    # IGNORECASE without .lower() equaling the alias key — a KeyError here silently dropped the
    # whole utterance (review 2026-07-02 round 2, P3).
    return pattern.sub(lambda m: aliases.get(m.group(0).lower(), m.group(0)), text)
