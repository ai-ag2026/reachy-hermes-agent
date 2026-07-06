from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, Protocol

import numpy as np

from reachy_agent.playback_queue import (
    CancellablePlaybackQueue,
    NoSpeakerPlaybackSmokeResult,
    PlaybackAdapter,
    PlaybackChunk,
)
from reachy_agent.agent_voice_latency import LatencyEventRecorder

logger = logging.getLogger(__name__)


class AsyncPostClient(Protocol):
    """Minimal async POST client used by QwenTtsClient."""

    async def post(self, url: str, **kwargs: Any) -> Any:
        """Return an HTTP response object with content and raise_for_status()."""


class SpeechSynthesizer(Protocol):
    """Minimal async text-to-audio synthesizer interface."""

    async def synthesize(self, text: str) -> bytes:
        """Synthesize one speakable text chunk and return encoded audio bytes."""


@dataclass(frozen=True)
class QwenTtsConfig:
    """OpenAI-compatible `/v1/audio/speech` config for AI-VM Qwen3-TTS."""

    base_url: str = "http://127.0.0.1:7034/v1"
    model: str = "qwen3-tts"
    voice: str = "default"
    api_key_env: str = "AGENT_QWEN_TTS_API_KEY"
    response_format: str = "wav"
    timeout_seconds: float = 8.0

    @classmethod
    def from_env(cls) -> "QwenTtsConfig":
        """Build config from AGENT_QWEN_TTS_* environment variables."""
        defaults = cls()
        return cls(
            base_url=os.getenv("AGENT_QWEN_TTS_BASE_URL", defaults.base_url),
            model=os.getenv("AGENT_QWEN_TTS_MODEL", defaults.model),
            voice=os.getenv("AGENT_QWEN_TTS_VOICE", defaults.voice),
            api_key_env=os.getenv("AGENT_QWEN_TTS_API_KEY_ENV", defaults.api_key_env),
            response_format=os.getenv("AGENT_QWEN_TTS_RESPONSE_FORMAT", defaults.response_format),
            timeout_seconds=_env_float("AGENT_QWEN_TTS_TIMEOUT_SECONDS", defaults.timeout_seconds),
        )


@dataclass(frozen=True)
class TtsChunk:
    """Synthesized TTS chunk artifact kept in memory for no-speaker validation."""

    text: str
    audio_bytes: bytes
    sequence: int


@dataclass(frozen=True)
class TtsNoSpeakerSmokeResult:
    """No-speaker Pack 4 smoke artifact."""

    chunk_count: int
    total_audio_bytes: int
    playback_side_effects: list[str]
    chunks: list[TtsChunk]


class QwenTtsClient:
    """Small OpenAI-compatible client for short Qwen3-TTS phrase chunks."""

    def __init__(
        self,
        config: QwenTtsConfig | None = None,
        http_client: AsyncPostClient | None = None,
        *,
        stream_source: Optional[Callable[[dict[str, Any], dict[str, str]], AsyncIterator[bytes]]] = None,
    ) -> None:
        """Initialize with optional injectable config, HTTP client, and PCM-stream byte source.

        ``stream_source(payload, headers)`` is an async byte-chunk generator used by
        :meth:`stream_pcm` instead of a live HTTP stream — the test seam (no network).
        """
        self.config = config or QwenTtsConfig.from_env()
        self._http_client = http_client
        self._stream_source = stream_source

    async def synthesize(self, text: str) -> bytes:
        """Synthesize one text chunk via `/audio/speech` and return audio bytes."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return b""
        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": cleaned_text,
            "response_format": self.config.response_format,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        client = self._http_client
        if client is not None:
            return await self._post_speech(client, payload, headers)

        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError("httpx is required for live QwenTtsClient synthesis") from exc

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as live_client:
            return await self._post_speech(live_client, payload, headers)

    async def _post_speech(
        self,
        client: AsyncPostClient,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> bytes:
        url = f"{self.config.base_url.rstrip('/')}/audio/speech"
        response = await client.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        content = getattr(response, "content", b"")
        return content if isinstance(content, bytes) else bytes(content)

    async def stream_pcm(self, text: str, *, sample_rate: int = 24_000) -> AsyncIterator[tuple[int, np.ndarray]]:
        """Stream raw PCM audio for **low time-to-first-audio**, yielding (sample_rate, int16) chunks.

        Requests ``response_format=pcm, stream=True`` and yields signed-16-bit mono samples as bytes
        arrive — the first chunk lands ~8x sooner than the full-file ``wav`` path (bench: ~160ms vs
        ~1300ms first-byte; see docs/decisions/LATENCY_2026-06-23_tts-pcm-stream.md). Odd trailing
        bytes are carried to the next chunk so every yield is on a whole-sample boundary.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return
        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": cleaned_text,
            "response_format": "pcm",
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        leftover = b""
        async for raw in self._iter_stream_bytes(payload, headers):
            if not raw:
                continue
            buf = leftover + raw
            usable = len(buf) - (len(buf) % 2)  # whole int16 samples only
            if usable:
                yield sample_rate, np.frombuffer(buf[:usable], dtype="<i2").copy()
            leftover = buf[usable:]
        # A well-formed PCM stream ends on a sample boundary; drop any stray trailing byte.

    async def _iter_stream_bytes(self, payload: dict[str, Any], headers: dict[str, str]) -> AsyncIterator[bytes]:
        if self._stream_source is not None:
            async for chunk in self._stream_source(payload, headers):
                yield chunk
            return
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError("httpx is required for live QwenTtsClient streaming") from exc
        url = f"{self.config.base_url.rstrip('/')}/audio/speech"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk


class CancellableTtsChunkQueue:
    """Serialize phrase chunks through TTS while allowing pending chunks to be cancelled."""

    def __init__(
        self,
        tts_client: SpeechSynthesizer,
        *,
        latency_recorder: LatencyEventRecorder | None = None,
        turn_id: str = "tts-chunk-queue",
    ) -> None:
        """Initialize an in-memory no-speaker TTS queue."""
        self._tts_client = tts_client
        self._latency_recorder = latency_recorder
        self._turn_id = turn_id
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._completed_chunks: list[TtsChunk] = []
        self._sequence = 0
        self._stop_requested = False
        self._first_tts_request_recorded = False
        self._first_audio_recorded = False
        self._pending_cancel_reason: str | None = None

    @property
    def completed_chunks(self) -> list[TtsChunk]:
        """Return synthesized chunks completed so far."""
        return list(self._completed_chunks)

    async def enqueue(self, text: str) -> None:
        """Queue one speakable text chunk for synthesis."""
        cleaned_text = text.strip()
        if cleaned_text and not self._stop_requested:
            await self._queue.put(cleaned_text)

    def cancel_pending(self, *, reason: str = "cancelled") -> int:
        """Drop queued chunks that have not started synthesis yet."""
        cancelled = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                self._queue.put_nowait(None)
                self._queue.task_done()
                break
            cancelled += 1
            self._queue.task_done()
        self._pending_cancel_reason = reason
        return cancelled

    async def run(self) -> None:
        """Process queued chunks until stopped."""
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                await self._synthesize_one(item)
                if self._pending_cancel_reason is not None:
                    self._record("tts_queue_cancelled", reason=self._pending_cancel_reason)
                    self._pending_cancel_reason = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A TTS error (HTTP 500/timeout, non-WAV bytes) must not kill the worker: a dead
                # worker leaves later items unprocessed and join() hangs forever. Log, record, go on.
                logger.warning(
                    "TTS chunk synthesis failed (chunk_chars=%s): %r", len(item) if isinstance(item, str) else "?", exc
                )
                self._record("tts_chunk_failed", error=type(exc).__name__)
            finally:
                self._queue.task_done()

    async def join(self) -> None:
        """Wait until all current queue items have been processed or cancelled."""
        await self._queue.join()

    async def stop(self) -> None:
        """Request worker shutdown after current item."""
        self._stop_requested = True
        await self._queue.put(None)

    async def _synthesize_one(self, text: str) -> None:
        if not self._first_tts_request_recorded:
            self._record("first_tts_request", chunk_chars=len(text))
            self._first_tts_request_recorded = True
        audio = await self._tts_client.synthesize(text)
        if audio and not self._first_audio_recorded:
            self._record("first_audio_bytes", audio_bytes=len(audio), chunk_chars=len(text))
            self._first_audio_recorded = True
        chunk = TtsChunk(text=text, audio_bytes=audio, sequence=self._sequence)
        self._sequence += 1
        self._completed_chunks.append(chunk)
        self._record("tts_chunk_ready", audio_bytes=len(audio), chunk_chars=len(text), sequence=chunk.sequence)

    def _record(self, event_type: str, **payload: Any) -> None:
        if self._latency_recorder is None:
            return
        self._latency_recorder.record(event_type, turn_id=self._turn_id, **payload)


async def run_tts_no_speaker_smoke(
    *,
    phrase_chunks: list[str],
    tts_client: SpeechSynthesizer,
    latency_recorder: LatencyEventRecorder,
    turn_id: str = "tts-no-speaker-smoke",
) -> TtsNoSpeakerSmokeResult:
    """Synthesize phrase chunks through a queue without mic, speaker, robot, or playback."""
    queue = CancellableTtsChunkQueue(tts_client, latency_recorder=latency_recorder, turn_id=turn_id)
    worker = asyncio.create_task(queue.run())
    for chunk in phrase_chunks:
        await queue.enqueue(chunk)
    await queue.join()
    await queue.stop()
    await worker
    completed = queue.completed_chunks
    return TtsNoSpeakerSmokeResult(
        chunk_count=len(completed),
        total_audio_bytes=sum(len(chunk.audio_bytes) for chunk in completed),
        playback_side_effects=[],
        chunks=completed,
    )


async def run_tts_to_playback_smoke(
    *,
    phrase_chunks: list[str],
    tts_client: SpeechSynthesizer,
    playback_adapter: PlaybackAdapter,
    latency_recorder: LatencyEventRecorder,
    turn_id: str = "tts-playback-smoke",
) -> NoSpeakerPlaybackSmokeResult:
    """Synthesize phrase chunks and route encoded audio through the provided playback adapter."""
    tts_result = await run_tts_no_speaker_smoke(
        phrase_chunks=phrase_chunks,
        tts_client=tts_client,
        latency_recorder=latency_recorder,
        turn_id=turn_id,
    )
    playback_queue = CancellablePlaybackQueue(
        playback_adapter,
        latency_recorder=latency_recorder,
        turn_id=turn_id,
    )
    worker = asyncio.create_task(playback_queue.run())
    for chunk in tts_result.chunks:
        await playback_queue.enqueue(
            PlaybackChunk(
                audio_bytes=chunk.audio_bytes,
                sequence=chunk.sequence,
                text=chunk.text,
            )
        )
    await playback_queue.join()
    await playback_queue.stop()
    await worker
    return NoSpeakerPlaybackSmokeResult(
        chunk_count=len(tts_result.chunks),
        total_audio_bytes=tts_result.total_audio_bytes,
        playback_side_effects=playback_queue.playback_side_effects,
        played_chunks=[
            PlaybackChunk(
                audio_bytes=chunk.audio_bytes,
                sequence=chunk.sequence,
                text=chunk.text,
            )
            for chunk in tts_result.chunks
        ],
    )


async def run_tts_to_playback_no_speaker_smoke(
    *,
    phrase_chunks: list[str],
    tts_client: SpeechSynthesizer,
    playback_adapter: PlaybackAdapter,
    latency_recorder: LatencyEventRecorder,
    turn_id: str = "tts-playback-no-speaker-smoke",
) -> NoSpeakerPlaybackSmokeResult:
    """Backward-compatible alias for null/no-speaker playback smokes."""
    return await run_tts_to_playback_smoke(
        phrase_chunks=phrase_chunks,
        tts_client=tts_client,
        playback_adapter=playback_adapter,
        latency_recorder=latency_recorder,
        turn_id=turn_id,
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float value for %s=%r, using default=%s", name, raw, default)
        return default
