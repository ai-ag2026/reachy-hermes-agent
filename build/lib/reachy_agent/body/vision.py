"""One-shot vision safety wrapper for AGENT-controlled Reachy (M3).

The AGENT-specific value over plain upstream vision is the **safety contract**, ported/adapted
from prior-art `agent_vision`:
- **person identification is ALLOWED by default** (the operator, 2026-06-26/2026-07-02: local model
  on a personal robot). An opt-in legacy denylist re-blocks who/name/face questions via
  ``AGENT_VISION_BLOCK_PERSON_ID=1`` (dashboard toggle "person");
- **no raw-image persistence and no raw-image return** (only a sanitized text description leaves
  this function);
- one frame, one answer (no streaming, no hidden capture).

Capture and description are **injected seams** so this is testable with fakes and adaptable to
either (a) the reachy_mini SDK camera + a VLM endpoint, or (b) the official app's
``camera_worker.get_latest_frame`` + ``vision_processor.process_image``:
    one_shot_vision(question, capture_frame=<()->ndarray|None>, describe=<(frame,question)->str>)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# OPT-IN legacy denylist (OFF by default — person ID is allowed; see module docstring).
# Best-effort only, NOT a hard guarantee. German compound-safe STEMS (no trailing \b) catch
# "Gesichtserkennung", "Erkennung", "Identifikation", "wiedererkennen"; plus strong identity
# phrasings. Keep identical to reachy_mini_conversation_app.tools.agent_vision.
_PERSON_IDENTIFICATION_RE = re.compile(
    r"(\bwho\b|\bwer\b|person|persona|\bname\b|identit|gesicht|\bface\b|"
    r"recogni[sz]e|erkenn|identifi|"
    r"\bwie (heißt|heisst|alt)\b|\bhow old\b|\b(his|her) name\b)",
    re.IGNORECASE,
)


def vision_policy(question: str) -> Dict[str, Any]:
    """Pure policy check. Returns {'blocked': bool, 'reason': str|None}.

    Person identification is ALLOWED by default (the operator: it's a local model on a personal robot —
    it can do anything; face/person recognition is a wanted embodiment feature). Set
    ``AGENT_VISION_BLOCK_PERSON_ID=1`` to re-enable the legacy denylist."""
    if os.getenv("AGENT_VISION_BLOCK_PERSON_ID", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ) and _PERSON_IDENTIFICATION_RE.search(question or ""):
        return {"blocked": True, "reason": "person_identification"}
    return {"blocked": False, "reason": None}


def _result(status: str, **fields: Any) -> Dict[str, Any]:
    # The no-raw-image contract is asserted on EVERY return path.
    fields.setdefault("image_persisted", False)
    fields.setdefault("raw_image_returned", False)
    fields["status"] = status
    return fields


def one_shot_vision(
    question: str,
    *,
    capture_frame: Callable[[], Optional[Any]],
    describe: Callable[[Any, str], str],
) -> Dict[str, Any]:
    """Answer one safe question about the current frame; never persist/return the raw image.

    ``capture_frame()`` returns a frame (numpy ndarray) or None. ``describe(frame, question)``
    returns a sanitized text description. Person-identification questions pass by default and
    are blocked before capture only when ``AGENT_VISION_BLOCK_PERSON_ID`` is enabled. All paths
    guarantee image_persisted=False, raw_image_returned=False.
    """
    question = (question or "").strip()
    qchars = len(question)
    if not question:
        return _result("error", error="question must be a non-empty string", question_chars=qchars)

    policy = vision_policy(question)
    if policy["blocked"]:
        logger.info("agent vision: blocked (person identification)")
        return _result("blocked", error="person identification is not allowed", question_chars=qchars, policy=policy)

    try:
        frame = capture_frame()
    except Exception:
        logger.warning("agent vision: capture failed")
        return _result("error", error="frame capture failed", question_chars=qchars, policy=policy)
    if frame is None:
        return _result("error", error="no frame available", question_chars=qchars, policy=policy)
    # ndarray-ish check without a hard numpy import dependency at module level
    shape = getattr(frame, "shape", None)
    if shape is None:
        return _result("error", error="frame is not an image array", question_chars=qchars, policy=policy)

    try:
        description = describe(frame, question)
    except Exception:
        logger.warning("agent vision: description failed")
        return _result(
            "error", error="vision processing failed", question_chars=qchars, frame_shape=list(shape), policy=policy
        )
    if not isinstance(description, str):
        return _result(
            "error", error="vision returned non-string", question_chars=qchars, frame_shape=list(shape), policy=policy
        )

    logger.info("agent vision: described frame %s (%d chars)", list(shape), len(description))
    return _result(
        "ok",
        image_description=description,
        question_chars=qchars,
        description_chars=len(description),
        frame_shape=list(shape),
        policy=policy,
    )
