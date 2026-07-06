from __future__ import annotations

from pathlib import Path

from reachy_agent.local_s2s.events import LocalS2SEventType
from reachy_agent.local_s2s.stt_source import (
    FakeTranscriptSource,
    SttSourceResult,
    transcript_event_from_audio,
)


def _fixture_audio(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.wav"
    path.write_bytes(b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt ")
    return path


def test_fake_transcript_source_returns_local_s2s_event(tmp_path: Path) -> None:
    """Turn fixture audio plus fake STT into a transcript-completed event."""
    audio = _fixture_audio(tmp_path)
    source = FakeTranscriptSource("Wir testen Reachy per Fixture Audio.")

    result = transcript_event_from_audio(audio, source)

    assert isinstance(result, SttSourceResult)
    assert result.status == "ok"
    assert result.transcript_chars == len("Wir testen Reachy per Fixture Audio.")
    assert result.event == {
        "type": LocalS2SEventType.TRANSCRIPTION_COMPLETED.value,
        "transcript": "Wir testen Reachy per Fixture Audio.",
    }
    assert result.audio["exists"] is True
    assert result.audio["mime_type"] == "audio/wav"
    assert result.timings_ms["audio_inspect_ms"] >= 0
    assert result.timings_ms["stt_transcribe_ms"] >= 0
    assert result.timings_ms["stt_total_ms"] >= result.timings_ms["audio_inspect_ms"]
    assert source.calls == [audio]


def test_fake_transcript_source_fails_closed_on_empty_text(tmp_path: Path) -> None:
    """Do not emit a transcript event when STT output is empty."""
    audio = _fixture_audio(tmp_path)
    source = FakeTranscriptSource("   ")

    result = transcript_event_from_audio(audio, source)

    assert result.status == "error"
    assert result.event is None
    assert result.transcript_chars == 0
    assert result.error == "empty_transcript"
    assert result.timings_ms["stt_total_ms"] >= 0


def test_stt_source_result_keeps_audio_metadata_only(tmp_path: Path) -> None:
    """Keep fixture audio metadata without embedding audio bytes."""
    audio = _fixture_audio(tmp_path)
    result = transcript_event_from_audio(audio, FakeTranscriptSource("Hallo"))

    assert "content" not in result.audio
    assert "bytes" in result.audio
    assert "timings_ms" in result.__dataclass_fields__
