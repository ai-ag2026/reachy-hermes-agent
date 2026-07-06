from __future__ import annotations

import asyncio
from pathlib import Path

from reachy_agent.local_s2s.events import (
    make_output_text_delta,
    make_response_cancelled,
    make_response_completed,
)
from reachy_agent.local_s2s.qwen_output import QwenOutputBridge, collect_speakable_chunks
from reachy_agent.playback_queue import NullPlaybackAdapter
from reachy_agent.agent_voice_latency import LatencyEventRecorder


class FakeQwenTtsClient:
    """Fake Qwen TTS client that records text and returns deterministic bytes."""

    def __init__(self) -> None:
        """Initialize fake synthesis call log."""
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        """Return deterministic fake audio bytes for text."""
        self.calls.append(text)
        return f"WAV:{text}".encode()


def test_collects_deltas_into_speakable_chunks() -> None:
    """Collect text deltas into sentence-sized speakable chunks."""
    result = collect_speakable_chunks(
        [
            make_output_text_delta(delta="Ich bin AGENT. ").to_dict(),
            make_output_text_delta(delta="Ich laufe lokal").to_dict(),
            make_response_completed().to_dict(),
        ]
    )

    assert result.chunks == ["Ich bin AGENT.", "Ich laufe lokal"]
    assert result.cancelled is False


def test_qwen_output_sends_chunks_to_tts_and_null_playback(tmp_path: Path) -> None:
    """Send speakable chunks to injected Qwen client and null playback only."""
    tts = FakeQwenTtsClient()
    playback = NullPlaybackAdapter()
    recorder = LatencyEventRecorder(tmp_path / "latency.jsonl", session_id="test-session")
    bridge = QwenOutputBridge(tts, playback_adapter=playback, latency_recorder=recorder, turn_id="turn-1")

    result = asyncio.run(
        bridge.handle_events(
            [
                make_output_text_delta(delta="Ich bin AGENT. ").to_dict(),
                make_output_text_delta(delta="Bereit.").to_dict(),
                make_response_completed().to_dict(),
            ]
        )
    )

    assert tts.calls == ["Ich bin AGENT.", "Bereit."]
    assert result.chunk_count == 2
    assert result.total_audio_bytes == len("WAV:Ich bin AGENT.".encode()) + len("WAV:Bereit.".encode())
    assert result.playback_side_effects == []
    assert [chunk.text for chunk in result.played_chunks] == ["Ich bin AGENT.", "Bereit."]
    assert playback.played_chunks == result.played_chunks


def test_cancellation_stops_later_chunks_and_records_null_cancel(tmp_path: Path) -> None:
    """Stop consuming text after cancellation and avoid later TTS calls."""
    tts = FakeQwenTtsClient()
    playback = NullPlaybackAdapter()
    recorder = LatencyEventRecorder(tmp_path / "latency.jsonl", session_id="test-session")
    bridge = QwenOutputBridge(tts, playback_adapter=playback, latency_recorder=recorder, turn_id="turn-cancel")

    result = asyncio.run(
        bridge.handle_events(
            [
                make_output_text_delta(delta="Erster Satz. ").to_dict(),
                make_response_cancelled(reason="barge_in").to_dict(),
                make_output_text_delta(delta="Darf nicht sprechen.").to_dict(),
                make_response_completed().to_dict(),
            ]
        )
    )

    assert tts.calls == ["Erster Satz."]
    assert result.chunk_count == 1
    assert playback.cancelled_reasons == ["barge_in"]
    assert result.playback_side_effects == []


def test_no_text_events_return_empty_no_speaker_report(tmp_path: Path) -> None:
    """Return empty metadata without touching TTS when no text arrived."""
    tts = FakeQwenTtsClient()
    recorder = LatencyEventRecorder(tmp_path / "latency.jsonl", session_id="test-session")
    bridge = QwenOutputBridge(tts, latency_recorder=recorder)

    result = asyncio.run(bridge.handle_events([make_response_completed().to_dict()]))

    assert tts.calls == []
    assert result.chunk_count == 0
    assert result.total_audio_bytes == 0
    assert result.playback_side_effects == []
    assert result.played_chunks == []


def test_max_char_chunking_splits_without_sentence_boundary() -> None:
    """Split long unsentenced text by configured character budget."""
    result = collect_speakable_chunks(
        [make_output_text_delta(delta="abcdefghi").to_dict(), make_response_completed().to_dict()],
        max_chunk_chars=4,
    )

    assert result.chunks == ["abcd", "efgh", "i"]
