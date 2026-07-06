from __future__ import annotations

import wave
from pathlib import Path

from reachy_agent.local_s2s.endpointing import (
    FixedWindowEndpointingConfig,
    fixed_window_endpointing_metadata,
)
from reachy_agent.local_s2s.stt_source import FakeTranscriptSource, transcript_event_from_audio


def _wav_fixture(tmp_path: Path, *, seconds: float = 1.0, rate: int = 16_000, channels: int = 1) -> Path:
    path = tmp_path / "speech.wav"
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * frames * channels)
    return path


def test_fixed_window_endpointing_metadata_is_report_safe(tmp_path: Path) -> None:
    """Expose endpointing timing fields without raw audio or transcript payloads."""
    audio = _wav_fixture(tmp_path, seconds=1.25, channels=2)

    metadata = fixed_window_endpointing_metadata(
        audio_path=audio,
        config=FixedWindowEndpointingConfig(capture_window_ms=1_250, silence_tail_ms=180),
    )

    assert metadata == {
        "endpoint_strategy": "fixed_window",
        "capture_window_ms": 1250,
        "endpoint_detected_ms": 1250,
        "audio_to_stt_submit_ms": 0,
        "speech_duration_estimate_ms": 1250,
        "silence_tail_ms": 180,
        "transcript_ready_ms": None,
    }
    assert str(audio) not in str(metadata)


def test_transcript_event_from_audio_adds_endpointing_contract(tmp_path: Path) -> None:
    """STT handoff reports the input-side turn-detection contract."""
    audio = _wav_fixture(tmp_path, seconds=0.5)

    result = transcript_event_from_audio(
        audio,
        FakeTranscriptSource("Bereit."),
        endpointing=FixedWindowEndpointingConfig(capture_window_ms=500, silence_tail_ms=100),
    )

    assert result.status == "ok"
    assert result.endpointing is not None
    assert result.endpointing["endpoint_strategy"] == "fixed_window"
    assert result.endpointing["capture_window_ms"] == 500
    assert result.endpointing["endpoint_detected_ms"] == 500
    assert result.endpointing["speech_duration_estimate_ms"] == 500
    assert result.endpointing["silence_tail_ms"] == 100
    assert result.endpointing["transcript_ready_ms"] == result.timings_ms["stt_total_ms"] + 500
    assert result.timings_ms["endpoint_detected_ms"] == 500
    assert result.timings_ms["audio_to_stt_submit_ms"] == 0
    assert result.timings_ms["transcript_ready_ms"] == result.endpointing["transcript_ready_ms"]
