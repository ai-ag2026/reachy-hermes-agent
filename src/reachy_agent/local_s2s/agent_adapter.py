"""Adapter from local S2S transcript events to AGENT text-delta events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Mapping, Protocol, runtime_checkable

from reachy_agent.local_s2s.events import (
    LocalS2SEventType,
    make_output_text_delta,
    make_response_completed,
    parse_local_s2s_event,
)


@runtime_checkable
class AgentFastClient(Protocol):
    """Minimal AGENT client surface consumed by the local S2S adapter."""

    async def ask_fast(self, transcript: str) -> str:
        """Return a short spoken AGENT response."""
        ...


@runtime_checkable
class AgentStreamingClient(Protocol):
    """Optional streaming AGENT client surface for lower-latency deltas."""

    def stream_fast(self, transcript: str) -> AsyncIterator[str]:
        """Yield short spoken AGENT response chunks."""
        ...


class AgentS2SAdapter:
    """Turn completed transcript events into AGENT response delta events."""

    def __init__(self, client: AgentFastClient | AgentStreamingClient) -> None:
        """Initialize the adapter with an injected AGENT client."""
        self._client = client

    async def handle_event(self, event: Mapping[str, object]) -> list[dict[str, object]]:
        """Handle one local S2S event and return response events.

        Non-transcript events are ignored by this adapter. Empty transcripts are
        an explicit no-op: the STT/VAD lane may emit silence/empty outcomes, but
        the adapter must not call AGENT for those.
        """
        parsed = parse_local_s2s_event(event)
        if parsed.type != LocalS2SEventType.TRANSCRIPTION_COMPLETED:
            return []

        raw_transcript = parsed.payload.get("transcript")
        transcript = raw_transcript.strip() if isinstance(raw_transcript, str) else ""
        if not transcript:
            return []

        output_events: list[dict[str, object]] = []
        async for chunk in self._response_chunks(transcript):
            if chunk:
                output_events.append(make_output_text_delta(delta=chunk).to_dict())
        output_events.append(make_response_completed().to_dict())
        return output_events

    async def _response_chunks(self, transcript: str) -> AsyncIterator[str]:
        if isinstance(self._client, AgentStreamingClient):
            async for chunk in self._client.stream_fast(transcript):
                yield _coerce_chunk(chunk)
            return

        if not isinstance(self._client, AgentFastClient):
            raise TypeError("AGENT S2S client must implement ask_fast() or stream_fast()")
        yield _coerce_chunk(await self._client.ask_fast(transcript))


def _coerce_chunk(chunk: Any) -> str:
    if not isinstance(chunk, str):
        raise TypeError("AGENT response chunks must be strings")
    return chunk
