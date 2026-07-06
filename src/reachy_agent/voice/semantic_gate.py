"""Semantic barge-in gate: decide whether a user utterance captured *while AGENT is speaking* is a
real interruption (COMMIT — stop and treat it as the next turn) or a backchannel / background noise
(RESUME — keep speaking the rest of the reply).

Two stages, cheapest first:

1. ``barge_decision`` — a pure heuristic (no network). Backchannels ("ja", "mhm", "okay"), empties
   and lone non-command words → RESUME; a known interrupt command word or a multi-word utterance →
   COMMIT. It leans to RESUME so background chatter doesn't stop AGENT, but a single clear "stopp"
   still commits.

2. ``semantic_decision`` — the hybrid gate used at runtime. If the heuristic already says RESUME we
   return immediately (no LLM call). Otherwise we ask a helper LLM (the AGENT/Qwopus endpoint with
   ``reasoning_effort="none"``, max_tokens 4) to classify STOP vs WEITER. The LLM latency is hidden
   behind the onset-pause (AGENT has already gone quiet). On any error we default to COMMIT — never
   drop a real interruption.

The LLM call is injected via ``ask_stop_continue`` so the gate is unit-testable without a network.
Ported from scripts/reachy_agent_mvp_dialogue.py (commit 2abbc9a), which proved the design live.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Literal

logger = logging.getLogger(__name__)

Decision = Literal["commit", "resume"]
# 3-way interrupt outcome (the operator's design):
#   ignore  — backchannel / not directed at AGENT -> AGENT was never interrupted, keep talking
#   stop    — a BARE stop ("stopp", "halt") with no further instruction -> stop, forward nothing
#   commit  — a real interrupt WITH content (a stop+instruction, or a new question/command)
#             -> stop and forward the WHOLE transcript as AGENT's next input
Interrupt = Literal["ignore", "stop", "commit"]

# Short acknowledgements that should NOT interrupt AGENT — the user is just signalling "I'm listening".
_BACKCHANNEL: frozenset[str] = frozenset(
    {
        "ja",
        "okay",
        "ok",
        "mhm",
        "hm",
        "aha",
        "genau",
        "stimmt",
        "jaja",
        "mm",
        "ah",
        "oh",
        "ja ja",
        "yeah",
        "yep",
        "uh",
        "uh huh",
        "hmm",
        "soso",
        "ach",
        "achso",
    }
)

# Single words that, on their own, ARE a real interrupt (otherwise a lone word leans to backchannel).
_INTERRUPT_CMDS: frozenset[str] = frozenset(
    {
        "stopp",
        "stop",
        "halt",
        "warte",
        "nein",
        "moment",
        "ruhe",
        "ende",
        "schweig",
        "still",
        "aufhören",
        "pause",
        "stille",
    }
)

# Trivial words that don't add an instruction — a stop padded only with these is still a BARE stop.
_FILLER: frozenset[str] = frozenset(
    {
        "mal",
        "jetzt",
        "bitte",
        "kurz",
        "doch",
        "agent",
        "reachy",
        "und",
        "ähm",
        "äh",
        "so",
        "also",
        "sei",
        "bist",
        "du",
        "sein",
        "ein",
        "kurzer",
        "moment",
    }
)

_GATE_SYS = (
    "Ein Roboter spricht gerade. Klassifiziere die Nutzer-Äußerung: echte Unterbrechung "
    "(Einwand/Korrektur/Stopp/neue Frage oder Anweisung AN den Roboter) -> STOP, oder nur "
    "Backchannel/Zustimmung (ja, mhm, okay, genau) bzw. Hintergrund-Gerede -> WEITER. "
    "Antworte mit genau einem Wort: STOP oder WEITER."
)


def barge_decision(text: str | None) -> Decision:
    """Pure heuristic stage. RESUME on backchannel / empty / very-short / lone non-command word;
    COMMIT on a known command word or a substantive multi-word utterance."""
    s = (text or "").strip().lower().strip(".!?,…")
    words = s.split()
    # Normalize INNER punctuation for the set lookups too: "Ja, ja." stripped only at the edges
    # -> "ja, ja" missed _BACKCHANNEL and fell through to commit / an LLM call
    # (review 2026-07-02 round 2, P3).
    s_norm = re.sub(r"[^\wäöüß ]+", "", s).strip()
    s_norm = re.sub(r"\s+", " ", s_norm)
    if not s or len(s) <= 3 or s in _BACKCHANNEL or s_norm in _BACKCHANNEL:
        return "resume"
    # a doubled backchannel ("ja ja", "ok ok") is still a backchannel
    parts = s_norm.split()
    if parts and len(set(parts)) == 1 and parts[0] in _BACKCHANNEL:
        return "resume"
    if len(words) <= 1 and s not in _INTERRUPT_CMDS:
        # a lone non-command word is most likely a backchannel / mis-transcribed noise
        return "resume"
    return "commit"


def _resolve_gate_endpoint(base_url: str | None, model: str | None, api_key: str | None) -> tuple[str, str, str]:
    """Resolve the helper-LLM endpoint for the gate. Prefers a dedicated fast gate model
    (AGENT_GATE_BASE_URL / AGENT_GATE_MODEL — e.g. a small model on llama-swap) so the gate is quick
    while AGENT's brain stays the big model; falls back to the AGENT_* brain endpoint. The gate auth
    key comes from AGENT_GATE_API_KEY (default none — llama-swap needs no auth) when a gate URL is set,
    else from API_SERVER_KEY (the AGENT brain)."""
    gate_url = os.getenv("AGENT_GATE_BASE_URL", "").strip()
    if base_url:
        url = base_url
    elif gate_url:
        url = gate_url
    else:
        url = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8642/v1")
    mdl = model or os.getenv("AGENT_GATE_MODEL", "").strip() or os.getenv("AGENT_MODEL", "AGENT")
    if api_key is not None:
        key = api_key
    elif gate_url:
        key = os.getenv("AGENT_GATE_API_KEY", "").strip()
    else:
        key = os.getenv("API_SERVER_KEY", "").strip()
    return url, mdl, key


def _default_ask_stop_continue(
    text: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    system: str = _GATE_SYS,
    max_tokens: int = 4,
) -> str:
    """Call the helper LLM (OpenAI-compatible /chat/completions) and return its raw reply text.
    reasoning_effort=none keeps it fast; a few tokens are enough for the one-word verdict."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        # Disable the gate model's chain-of-thought so it answers in one word in ~100ms instead of
        # reasoning for seconds (Qwopus-9B is a thinking model; the 35B-A3B chat template honors this).
        # Unknown to non-thinking backends (the AGENT brain fallback), which ignore it.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }
    r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def semantic_decision(
    text: str | None,
    *,
    ask_stop_continue: Callable[[str], str] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 20.0,
) -> Decision:
    """Hybrid gate: heuristic RESUME short-circuit, else ask the helper LLM STOP/WEITER.

    ``ask_stop_continue`` lets tests (and alternative backends) inject the classifier; when omitted it
    calls the AGENT/Qwopus endpoint resolved from the args or env (AGENT_BASE_URL/AGENT_MODEL/
    API_SERVER_KEY). Returns COMMIT on any error so a real interruption is never silently dropped."""
    if barge_decision(text) == "resume":
        return "resume"
    assert text is not None  # barge_decision returns "resume" for None, so we never reach here with None

    if ask_stop_continue is None:
        resolved_url, resolved_model, resolved_key = _resolve_gate_endpoint(base_url, model, api_key)

        def ask_stop_continue(t: str) -> str:  # type: ignore[misc]
            return _default_ask_stop_continue(
                t, base_url=resolved_url, model=resolved_model, api_key=resolved_key, timeout=timeout
            )

    try:
        out = (ask_stop_continue(text) or "").upper()
    except Exception as exc:  # network / parse / timeout — never drop a real interrupt
        logger.warning("semantic_decision helper failed, defaulting to commit: %r", exc)
        return "commit"
    # WEITER (and not STOP) => resume; anything else (incl. ambiguous) => commit
    return "resume" if "WEITER" in out and "STOP" not in out else "commit"


def is_bare_stop(text: str | None) -> bool:
    """True if the utterance is ONLY a stop/command word (plus trivial filler) with no further
    instruction — e.g. "stopp", "halt mal", "stopp bitte". A stop carrying content ("stopp, wechsle
    das Thema") is NOT bare. Used to decide whether to forward the transcript to AGENT after stopping."""
    words = re.findall(r"[a-zäöüß]+", (text or "").lower())
    if not words or not any(w in _INTERRUPT_CMDS for w in words):
        return False
    content = [w for w in words if w not in _INTERRUPT_CMDS and w not in _FILLER]
    return len(content) == 0


_CLASSIFY_SYS = (
    "Ein Roboter (AGENT) spricht gerade. Klassifiziere die Nutzer-Äußerung in GENAU EIN Wort:\n"
    "WEITER = nur Backchannel/Zustimmung/Hintergrund-Gerede, nicht an den Roboter gerichtet.\n"
    "STOP = der Nutzer will den Roboter NUR stoppen, ohne weitere Anweisung.\n"
    "INHALT = echte Unterbrechung MIT Inhalt: eine neue Frage, ein Kommando oder ein Stopp samt "
    "Anweisung (z. B. 'hör auf und erzähl mir etwas anderes').\n"
    "Antworte mit genau einem Wort: WEITER, STOP oder INHALT."
)


def classify_interrupt(
    text: str | None,
    *,
    ask_classify: Callable[[str], str] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 20.0,
) -> Interrupt:
    """3-way gate for the operator's non-pausing barge-in. Heuristic first (no network):
    backchannel/non-directed -> ignore; bare stop -> stop. For anything else ask the helper LLM
    (Qwobus) WEITER/STOP/INHALT; on error default to commit (never drop a real interrupt). On a
    commit the caller forwards the WHOLE transcript to AGENT as the next turn."""
    if barge_decision(text) == "resume":
        return "ignore"
    if is_bare_stop(text):
        return "stop"
    assert text is not None

    if ask_classify is None:
        resolved_url, resolved_model, resolved_key = _resolve_gate_endpoint(base_url, model, api_key)

        def ask_classify(t: str) -> str:  # type: ignore[misc]
            return _default_ask_stop_continue(
                t,
                base_url=resolved_url,
                model=resolved_model,
                api_key=resolved_key,
                timeout=timeout,
                system=_CLASSIFY_SYS,
                max_tokens=6,
            )

    try:
        out = (ask_classify(text) or "").upper()
    except Exception as exc:
        logger.warning("classify_interrupt helper failed, defaulting to commit: %r", exc)
        return "commit"
    if "WEITER" in out and "STOP" not in out and "INHALT" not in out:
        return "ignore"
    if "STOP" in out and "INHALT" not in out:
        return "stop"
    return "commit"
