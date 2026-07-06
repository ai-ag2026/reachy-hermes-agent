"""Parallel vision context for the AGENT voice turn.

Architecture (the operator's parallelism directive, 2026-06-26): the fast local 9B (Qwopus-9B :3447,
which also serves the VLM) runs IN PARALLEL with the the agent main brain. For vision the composition
is deliberately simple and ordered — the VLM's scene description is **input to** the agent (folded into
its prompt), never a second spoken stream to merge. So there is no contradiction/overlap risk: AGENT
"sees" via the description and then reasons/answers with it.

This module is pure and seam-injected (no robot, no network) so it is unit-testable:
- ``is_visual_query(transcript)`` — fast heuristic gate: does this turn plausibly need sight?
- ``scene_context(transcript, frame, describe)`` — if visual, call the injected ``describe`` seam and
  return a short context line to prepend to the agent's prompt, else ``None``.

The caller (AgentVoiceHandler) supplies the real camera frame (``camera_worker.get_latest_frame()``)
and ``describe`` (``vlm_client.smolvlm_describe`` → 9B VLM), and fires it as a task so the VLM latency
overlaps the user's speech instead of adding to the reply.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Strong visual-intent markers (DE + EN). Kept focused so non-visual turns don't trigger a look.
# Person identification is allowed (opt-in re-block via AGENT_VISION_BLOCK_PERSON_ID in
# body.vision) — this here is plain scene sight.
_VISUAL_RE = re.compile(
    r"(siehst du|sieh(st|\s|t)?\s|schau|guck|was siehst|welche farbe|wie sieht|"
    r"was ist (das|hier|da)\b|vor dir|in der hand|zeig|kamera|"
    r"\bbild\b|\bszene\b|\bfarbe\b|\bobjekt|"
    # reading text off a physical thing implies sight even without "siehst du" — keep these here
    # so the strict video/native gate (vision_mode) doesn't drop them (audit 2026-07-02)
    r"was steht|steht (da|drauf|dort)|welcher text|etikett|beschriftung|lies (mal|das|hier|was|vor)|vorlesen|"
    r"auf dem (tisch|schreibtisch|boden)|"
    r"do you see|can you see|what do you see|look at|what color|"
    r"what(?:'s| is) (this|that|here)|in front of you|on the (table|desk)|what does it say)",
    re.IGNORECASE,
)

# What the VLM is asked. Concrete scene description (no person identification — the VLM client's own
# system prompt also forbids it); the user's actual question is answered by the agent using this.
SCENE_PROMPT = (
    "Du bist Reachys Kamera-Auge. Beschreibe das Bild präzise und vollständig, sodass jemand ohne das "
    "Bild genau weiß, was vor der Kamera ist:\n"
    "1) Jedes erkennbare Objekt mit Farbe, Form/Material und Position (links/Mitte/rechts, vorne/hinten).\n"
    "2) Anzahl gleichartiger Objekte (zähle, wenn sinnvoll).\n"
    "3) Jeden lesbaren Text oder jedes Etikett WÖRTLICH wiedergeben.\n"
    "4) Räumliche Anordnung und auffällige Details (Anordnung, Größenverhältnisse, Zustand).\n"
    "5) Wenn Personen sichtbar sind, beschreibe sie (Aussehen, Kleidung, Haltung, Mimik) und nenne, "
    "wer es sein könnte, falls erkennbar.\n"
    "Antworte dicht und sachlich, keine Floskeln, nichts erfinden."
)

_MAX_CONTEXT_CHARS = 900


# A visual turn that needs real visual REASONING (compare / read text / count precisely / spatial
# relations / close analysis) is routed to the premium native path (the agent) instead
# of a text scene description — a small VLM's prose often loses exactly these details.
_COMPLEX_VISUAL_RE = re.compile(
    # Strong visual-reasoning cues only. Bare precision adverbs (genau/exakt/präzise/analysier) were
    # removed — they over-fired the costly native native-vision path on non-visual turns
    # ("erklär mir das genau").
    r"(vergleich|unterschied|lies\b|lies mal|vorlesen|vorles|"
    r"steht (da|drauf|dort)|was steht|welcher text|\bschrift|etikett|label|beschriftung|"
    r"wie viele|\bz(ä|ae)hl|\banzahl|"
    r"hinter|davor|dahinter|links neben|rechts neben|oberhalb|unterhalb|zwischen|"
    r"gr(ö|oe)(ß|ss)er|kleiner|n(ä|ae)her|weiter weg|schau (mal )?genau|genau hin|"
    r"compare|difference|\bread\b|exactly|how many|\bcount\b|behind|in front of|next to|larger|smaller)",
    re.IGNORECASE,
)


def is_visual_query(transcript: str) -> bool:
    """True if the turn plausibly needs sight (fast heuristic — err toward not triggering)."""
    return bool(transcript) and _VISUAL_RE.search(transcript) is not None


def is_complex_visual(transcript: str) -> bool:
    """True if a visual turn needs real visual reasoning -> route to native vision (premium),
    not the fast gemma scene description."""
    return bool(transcript) and _COMPLEX_VISUAL_RE.search(transcript) is not None


# A turn about MOTION / what is happening over time -> a short multi-frame clip to gemma (Gemma 3n
# does video). Video stays gemma-only (the agent).
_VIDEO_RE = re.compile(
    r"(video|\bclip\b|kontinuierlich|schau (mal )?zu|beobacht|was passiert|passiert (gerade|da)|"
    r"bewegt sich|bewegung|gerade jetzt|live\b|\bfilm|aufnahme|sequenz|nimm auf|"
    r"watch\b|what(?:'s| is) happening|moving|motion|record\b)",
    re.IGNORECASE,
)


def is_video_query(transcript: str) -> bool:
    """True if the turn asks about motion / what is happening over time -> short gemma video clip."""
    return bool(transcript) and _VIDEO_RE.search(transcript) is not None


# Deictic anchors: "wie viele Tassen stehen DA?" / "was passiert GERADE?" are visual even without
# an explicit sight verb — the here-and-now reference points at the scene in front of the robot.
_DEICTIC_RE = re.compile(
    r"\b(da|hier|dort|gerade|jetzt|drauf|davor|dahinter|daneben|vor dir|here|there|right now)\b",
    re.IGNORECASE,
)


def vision_mode(transcript: str) -> str:
    """Route a turn: 'none' | 'video' (gemma multi-frame) | 'native' (the agent) |
    'describe' (gemma single-frame description). Motion wins -> video; a visual turn that needs
    visual reasoning (read/compare/count/spatial) -> native; a plain visual turn -> fast gemma
    describe; else none.

    video/native additionally require a visual marker OR a deictic anchor (audit 2026-07-02):
    the complex/video regexes alone over-fired on everyday non-visual turns ("Wie viele Terabyte
    hat der NAS?" -> native sent a camera frame to the cloud; "Was passiert, wenn ich den Server
    neu starte?" -> video grabbed a 6-frame clip)."""
    anchored = is_visual_query(transcript) or (bool(transcript) and _DEICTIC_RE.search(transcript) is not None)
    if is_video_query(transcript) and anchored:
        return "video"
    if is_complex_visual(transcript) and anchored:
        return "native"
    return "describe" if is_visual_query(transcript) else "none"


def scene_context(
    transcript: str,
    frame: Any,
    describe: Callable[[Any, str], str],
    *,
    force: bool = False,
) -> Optional[str]:
    """Return a short scene-context line for the agent, or None if not a visual turn / unavailable.

    ``describe(frame, prompt) -> str`` is the injected VLM seam. Best-effort: any failure or empty
    description returns None (the turn proceeds without sight — never breaks). ``force`` bypasses the
    gate (e.g. when the caller already classified the turn as visual).
    """
    if not force and not is_visual_query(transcript):
        return None
    if frame is None:
        return None
    try:
        description = describe(frame, SCENE_PROMPT)
    except Exception as exc:  # best-effort: vision is additive, never fatal to the turn
        logger.info("vision_context: describe failed (%s) — proceeding without sight", type(exc).__name__)
        return None
    description = (description or "").strip()
    if not description:
        return None
    if len(description) > _MAX_CONTEXT_CHARS:
        description = description[:_MAX_CONTEXT_CHARS].rstrip() + "…"
    # Ordered composition: this is CONTEXT for the agent, clearly labeled, not a spoken answer.
    return f"[Was du gerade durch deine Kamera siehst: {description}]"
