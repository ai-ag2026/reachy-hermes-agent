"""Endpointing metadata helpers for local S2S smokes.

The first N4 strategy is intentionally conservative: fixed-window endpointing.
It records where turn detection would sit in the pipeline without keeping raw
transcript/audio data or touching live microphone paths.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixedWindowEndpointingConfig:
    """Metadata configuration for fixed-window endpointing."""

    capture_window_ms: int | None = None
    silence_tail_ms: int = 0
    audio_to_stt_submit_ms: int = 0


EndpointingMetadata = dict[str, int | str | None]


def fixed_window_endpointing_metadata(
    *,
    audio_path: Path,
    config: FixedWindowEndpointingConfig | None = None,
) -> EndpointingMetadata:
    """Return metadata-only fixed-window endpointing timings for an audio file."""
    cfg = config or FixedWindowEndpointingConfig()
    speech_duration_ms = _wav_duration_ms(audio_path)
    capture_window_ms = cfg.capture_window_ms if cfg.capture_window_ms is not None else speech_duration_ms
    endpoint_detected_ms = capture_window_ms
    return {
        "endpoint_strategy": "fixed_window",
        "capture_window_ms": capture_window_ms,
        "endpoint_detected_ms": endpoint_detected_ms,
        "audio_to_stt_submit_ms": cfg.audio_to_stt_submit_ms,
        "speech_duration_estimate_ms": speech_duration_ms,
        "silence_tail_ms": cfg.silence_tail_ms,
        "transcript_ready_ms": None,
    }


def endpointing_timings(metadata: EndpointingMetadata) -> dict[str, int]:
    """Extract integer timing fields for the smoke timing block."""
    timings: dict[str, int] = {}
    for key in (
        "capture_window_ms",
        "endpoint_detected_ms",
        "audio_to_stt_submit_ms",
        "speech_duration_estimate_ms",
        "silence_tail_ms",
        "transcript_ready_ms",
    ):
        value = metadata.get(key)
        if isinstance(value, int):
            timings[key] = value
    return timings


def finalize_endpointing_metadata(metadata: EndpointingMetadata, *, stt_total_ms: int) -> EndpointingMetadata:
    """Fill transcript-ready timing once STT has completed."""
    finalized = dict(metadata)
    endpoint_detected_ms = finalized.get("endpoint_detected_ms")
    audio_to_stt_submit_ms = finalized.get("audio_to_stt_submit_ms")
    if isinstance(endpoint_detected_ms, int) and isinstance(audio_to_stt_submit_ms, int):
        finalized["transcript_ready_ms"] = endpoint_detected_ms + audio_to_stt_submit_ms + stt_total_ms
    return finalized


def _wav_duration_ms(audio_path: Path) -> int:
    try:
        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                return 0
            return int(round((wav.getnframes() / rate) * 1000))
    except (OSError, wave.Error):
        return 0
