"""STT sources that produce local S2S transcript events."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from reachy_agent.local_s2s.endpointing import (
    EndpointingMetadata,
    FixedWindowEndpointingConfig,
    endpointing_timings,
    finalize_endpointing_metadata,
    fixed_window_endpointing_metadata,
)
from reachy_agent.local_s2s.events import make_transcription_completed
from reachy_agent.stt_client import SttClient, SttClientConfig, SttResult, inspect_audio_file


class TranscriptSource(Protocol):
    """Source that returns a completed transcript from audio input."""

    def transcribe(self, audio_path: Path) -> SttResult:
        """Transcribe the supplied audio file."""
        ...


@dataclass(frozen=True)
class SttSourceResult:
    """Metadata-only STT source result for local S2S handoff."""

    status: str
    event: dict[str, object] | None
    transcript_chars: int
    audio: dict[str, object]
    timings_ms: dict[str, int]
    endpointing: EndpointingMetadata | None = None
    language: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


class FakeTranscriptSource:
    """Deterministic transcript source for fixture and unit tests."""

    def __init__(self, transcript: str, *, language: str = "de") -> None:
        """Initialize with one transcript."""
        self.transcript = transcript
        self.language = language
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> SttResult:
        """Return the configured transcript and record the audio path."""
        self.calls.append(audio_path)
        cleaned = self.transcript.strip()
        if not cleaned:
            return SttResult(text="", status="error", error="empty_transcript")
        return SttResult(text=cleaned, language=self.language, status="ok")


class ParakeetTranscriptSource:
    """Parakeet/OpenAI-compatible STT transcript source."""

    def __init__(self, config: SttClientConfig | None = None) -> None:
        """Initialize with optional STT client config.

        Honors ``AGENT_STT_BASE_URL`` (preset-controlled, same as the live front-end) when no
        explicit config is given, so the preset toggle actually moves this lane's endpoint.
        """
        if config is None:
            import os

            base = os.getenv("AGENT_STT_BASE_URL", "http://127.0.0.1:5093").rstrip("/")
            config = SttClientConfig(base_url=base)
        self._client = SttClient(config)

    def transcribe(self, audio_path: Path) -> SttResult:
        """Transcribe audio through the configured STT endpoint."""
        return self._client.transcribe(audio_path)


def transcript_event_from_audio(
    audio_path: Path,
    source: TranscriptSource,
    *,
    endpointing: FixedWindowEndpointingConfig | None = None,
) -> SttSourceResult:
    """Transcribe an audio fixture and return a local S2S transcript event."""
    timings: dict[str, int] = {}
    total_start = time.perf_counter()
    inspect_start = time.perf_counter()
    audio = inspect_audio_file(audio_path)
    timings["audio_inspect_ms"] = _elapsed_ms(inspect_start)
    endpointing_meta: EndpointingMetadata | None = None
    if endpointing is not None:
        endpointing_meta = fixed_window_endpointing_metadata(audio_path=audio_path, config=endpointing)
        timings.update(endpointing_timings(endpointing_meta))
    stt_start = time.perf_counter()
    result = source.transcribe(audio_path)
    timings["stt_transcribe_ms"] = _elapsed_ms(stt_start)
    timings["stt_total_ms"] = _elapsed_ms(total_start)
    if endpointing_meta is not None:
        endpointing_meta = finalize_endpointing_metadata(endpointing_meta, stt_total_ms=timings["stt_total_ms"])
        timings.update(endpointing_timings(endpointing_meta))
    if result.status != "ok":
        return SttSourceResult(
            status=result.status,
            event=None,
            transcript_chars=0,
            audio=audio,
            timings_ms=timings,
            endpointing=endpointing_meta,
            language=result.language,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )
    event = make_transcription_completed(transcript=result.text).to_dict()
    return SttSourceResult(
        status="ok",
        event=event,
        transcript_chars=len(result.text),
        audio=audio,
        timings_ms=timings,
        endpointing=endpointing_meta,
        language=result.language,
        duration_seconds=result.duration_seconds,
    )


def _elapsed_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))
