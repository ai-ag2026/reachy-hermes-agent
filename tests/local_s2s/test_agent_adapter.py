from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from reachy_agent.local_s2s.events import make_transcription_completed
from reachy_agent.local_s2s.agent_adapter import AgentS2SAdapter


class FakeStreamingAgentClient:
    """Fake streaming client that records calls and yields configured chunks."""

    def __init__(self, chunks: list[str]) -> None:
        """Initialize fake streamed response chunks."""
        self.chunks = chunks
        self.calls: list[str] = []
        self.body_side_effects = 0

    async def stream_fast(self, transcript: str) -> AsyncIterator[str]:
        """Yield configured chunks for one transcript."""
        self.calls.append(transcript)
        for chunk in self.chunks:
            yield chunk


class FakeAskAgentClient:
    """Fake non-streaming client compatible with HermesAgentClient.ask_fast."""

    def __init__(self, response: str) -> None:
        """Initialize one-shot response text."""
        self.response = response
        self.calls: list[str] = []

    async def ask_fast(self, transcript: str) -> str:
        """Return configured text and record the transcript."""
        self.calls.append(transcript)
        return self.response


def test_transcript_event_calls_streaming_agent_client_once() -> None:
    """Call injected streaming AGENT client once for a completed transcript."""
    client = FakeStreamingAgentClient(["Ich ", "bin AGENT."])
    adapter = AgentS2SAdapter(client)
    event = make_transcription_completed(transcript=" Was bist du? ").to_dict()

    result = asyncio.run(adapter.handle_event(event))

    assert client.calls == ["Was bist du?"]
    assert client.body_side_effects == 0
    assert result == [
        {"type": "response.output_text.delta", "delta": "Ich "},
        {"type": "response.output_text.delta", "delta": "bin AGENT."},
        {"type": "response.completed"},
    ]


def test_adapter_falls_back_to_ask_fast_client() -> None:
    """Support the current HermesAgentClient ask_fast shape before true streaming lands."""
    client = FakeAskAgentClient("Kurz und brauchbar.")
    adapter = AgentS2SAdapter(client)
    event = make_transcription_completed(transcript="Sag etwas").to_dict()

    result = asyncio.run(adapter.handle_event(event))

    assert client.calls == ["Sag etwas"]
    assert result == [
        {"type": "response.output_text.delta", "delta": "Kurz und brauchbar."},
        {"type": "response.completed"},
    ]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.output_text.delta", "delta": "ignore me"},
        {"type": "response.completed"},
    ],
)
def test_adapter_ignores_non_transcript_events(event: dict[str, object]) -> None:
    """Ignore events owned by later output/playback stages."""
    client = FakeAskAgentClient("should not be called")
    adapter = AgentS2SAdapter(client)

    assert asyncio.run(adapter.handle_event(event)) == []
    assert client.calls == []


def test_adapter_skips_whitespace_transcript_without_agent_call() -> None:
    """Treat empty STT output as no-op instead of calling AGENT."""
    client = FakeAskAgentClient("should not be called")
    adapter = AgentS2SAdapter(client)
    event = {"type": "conversation.item.input_audio_transcription.completed", "transcript": "   "}

    assert asyncio.run(adapter.handle_event(event)) == []
    assert client.calls == []


def test_adapter_rejects_non_string_stream_chunk() -> None:
    """Fail fast if a broken client yields non-text chunks."""

    class BrokenStreamingClient:
        def stream_fast(self, transcript: str) -> AsyncIterator[str]:
            async def _inner() -> AsyncIterator[str]:
                yield 123  # type: ignore[misc]

            return _inner()

    adapter = AgentS2SAdapter(BrokenStreamingClient())
    event = make_transcription_completed(transcript="Hallo").to_dict()

    with pytest.raises(TypeError, match="chunks must be strings"):
        asyncio.run(adapter.handle_event(event))
