"""Qwen/NullPlayback output bridge for local S2S text-delta events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Protocol

from reachy_agent.local_s2s.events import LocalS2SEventType, parse_local_s2s_event
from reachy_agent.playback_queue import (
    NoSpeakerPlaybackSmokeResult,
    NullPlaybackAdapter,
    PlaybackAdapter,
)
from reachy_agent.agent_voice_latency import LatencyEventRecorder
from reachy_agent.tts_chunk_queue import SpeechSynthesizer, run_tts_to_playback_smoke


class OutputMode(Protocol):
    """Marker protocol for future explicit speaker modes."""


@dataclass(frozen=True)
class TextChunkResult:
    """Chunking result for local S2S text deltas."""

    chunks: list[str]
    cancelled: bool = False
    cancellation_reason: str | None = None


class QwenOutputBridge:
    """Route local S2S text delta events to Qwen TTS and null playback."""

    def __init__(
        self,
        tts_client: SpeechSynthesizer,
        *,
        playback_adapter: PlaybackAdapter | None = None,
        latency_recorder: LatencyEventRecorder | None = None,
        turn_id: str = "local-s2s-qwen-output",
        max_chunk_chars: int = 160,
    ) -> None:
        """Initialize the bridge with injected clients and metadata sinks."""
        self._tts_client = tts_client
        self._playback_adapter = playback_adapter or NullPlaybackAdapter()
        self._latency_recorder = latency_recorder or _default_latency_recorder(turn_id)
        self._turn_id = turn_id
        self._max_chunk_chars = max_chunk_chars

    async def handle_events(self, events: Iterable[dict[str, object]]) -> NoSpeakerPlaybackSmokeResult:
        """Convert output text events into no-speaker playback metadata."""
        chunk_result = collect_speakable_chunks(events, max_chunk_chars=self._max_chunk_chars)
        if chunk_result.cancelled:
            await self._playback_adapter.cancel_current(chunk_result.cancellation_reason or "response.cancelled")
        if not chunk_result.chunks:
            return NoSpeakerPlaybackSmokeResult(
                chunk_count=0,
                total_audio_bytes=0,
                playback_side_effects=[],
                played_chunks=[],
            )
        return await run_tts_to_playback_smoke(
            phrase_chunks=chunk_result.chunks,
            tts_client=self._tts_client,
            playback_adapter=self._playback_adapter,
            latency_recorder=self._latency_recorder,
            turn_id=self._turn_id,
        )


def collect_speakable_chunks(events: Iterable[dict[str, object]], *, max_chunk_chars: int = 160) -> TextChunkResult:
    """Collect response text deltas into simple speakable chunks."""
    chunks: list[str] = []
    buffer = ""
    for raw_event in events:
        event = parse_local_s2s_event(raw_event)
        if event.type == LocalS2SEventType.RESPONSE_CANCELLED:
            return TextChunkResult(
                chunks=chunks, cancelled=True, cancellation_reason=_reason(event.payload.get("reason"))
            )
        if event.type == LocalS2SEventType.RESPONSE_COMPLETED:
            _flush_buffer(buffer, chunks)
            return TextChunkResult(chunks=chunks)
        if event.type != LocalS2SEventType.OUTPUT_TEXT_DELTA:
            continue

        delta = event.payload.get("delta")
        if not isinstance(delta, str):
            continue
        buffer += delta
        ready, buffer = _pop_ready_chunks(buffer, max_chunk_chars=max_chunk_chars)
        chunks.extend(ready)

    _flush_buffer(buffer, chunks)
    return TextChunkResult(chunks=chunks)


def _pop_ready_chunks(buffer: str, *, max_chunk_chars: int) -> tuple[list[str], str]:
    chunks: list[str] = []
    while True:
        boundary = _first_sentence_boundary(buffer)
        if boundary is None and len(buffer.strip()) < max_chunk_chars:
            return chunks, buffer
        if boundary is None:
            boundary = max_chunk_chars
        chunk = buffer[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        buffer = buffer[boundary:].lstrip()


def _first_sentence_boundary(text: str) -> int | None:
    for index, char in enumerate(text):
        if char in ".!?…":
            return index + 1
    return None


def _flush_buffer(buffer: str, chunks: list[str]) -> None:
    chunk = buffer.strip()
    if chunk:
        chunks.append(chunk)


def _reason(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _default_latency_recorder(turn_id: str) -> LatencyEventRecorder:
    path = Path(gettempdir()) / "reachy-local-s2s" / f"{turn_id}.jsonl"
    return LatencyEventRecorder(path, session_id="local-s2s")
