from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import numpy as np

from reachy_agent.agent_voice_latency import LatencyEventRecorder

logger = logging.getLogger(__name__)

DEFAULT_REACHY_DAEMON_BASE_URL = "http://127.0.0.1:8000"
REACHY_DAEMON_BASE_URL_ENV = "REACHY_DAEMON_BASE_URL"


@dataclass(frozen=True)
class PlaybackChunk:
    """Encoded audio chunk ready for a playback backend."""

    audio_bytes: bytes
    sequence: int
    text: str = ""
    sample_rate_hz: int = 24_000


@dataclass(frozen=True)
class PlaybackResult:
    """Playback result metadata without exposing or requiring audio hardware."""

    sequence: int
    audio_bytes: int
    side_effects: list[str]


@dataclass(frozen=True)
class NoSpeakerPlaybackSmokeResult:
    """No-speaker playback smoke artifact."""

    chunk_count: int
    total_audio_bytes: int
    playback_side_effects: list[str]
    played_chunks: list[PlaybackChunk]


class PlaybackCancelled(Exception):
    """Raised by playback adapters when the in-flight chunk is cancelled."""

    def __init__(self, reason: str) -> None:
        """Initialize with the cancellation reason reported by the backend."""
        super().__init__(reason)
        self.reason = reason


class LiveSpeakerGateRequired(RuntimeError):
    """Raised when a live speaker adapter is used without explicit live approval."""


class PlaybackAdapter(Protocol):
    """Minimal async playback backend interface."""

    async def play(self, chunk: PlaybackChunk) -> PlaybackResult:
        """Play or simulate one encoded audio chunk."""

    async def cancel_current(self, reason: str) -> None:
        """Cancel the current in-flight playback chunk if possible."""


class HttpMediaClient(Protocol):
    """Async client for Reachy's daemon HTTP media sound endpoints."""

    async def upload_sound(self, *, filename: str, audio_bytes: bytes) -> str:
        """Upload encoded audio and return the daemon-side playable file path."""
        ...

    async def play_sound(self, *, file: str) -> dict[str, Any]:
        """Start playing a daemon-side sound file."""
        ...

    async def stop_sound(self) -> dict[str, Any]:
        """Stop current daemon-side sound playback."""
        ...

    async def delete_sound(self, *, filename: str) -> dict[str, Any]:
        """Delete a temporary uploaded daemon-side sound file by filename."""
        ...


SleepFn = Callable[[float], Awaitable[None]]


class NullPlaybackAdapter:
    """No-speaker playback adapter that records chunks without side effects."""

    def __init__(self) -> None:
        """Initialize an in-memory null playback sink."""
        self._played_chunks: list[PlaybackChunk] = []
        self.cancelled_reasons: list[str] = []

    @property
    def played_chunks(self) -> list[PlaybackChunk]:
        """Return chunks that would have been played."""
        return list(self._played_chunks)

    async def play(self, chunk: PlaybackChunk) -> PlaybackResult:
        """Record a chunk and report no external side effects."""
        self._played_chunks.append(chunk)
        return PlaybackResult(sequence=chunk.sequence, audio_bytes=len(chunk.audio_bytes), side_effects=[])

    async def cancel_current(self, reason: str) -> None:
        """Record cancellation intent without touching audio hardware."""
        self.cancelled_reasons.append(reason)


class ReachyHttpMediaClient:
    """Reachy daemon HTTP media client using stdlib requests via threads."""

    def __init__(self, *, base_url: str | None = None, timeout_seconds: float = 20.0) -> None:
        """Initialize with the Reachy daemon base URL.

        Defaults to the mDNS hostname so Reachy can move between networks
        without baking a site-specific IP into the code. Set
        ``REACHY_DAEMON_BASE_URL`` for an explicit override.
        """
        resolved_base_url = base_url or os.getenv(REACHY_DAEMON_BASE_URL_ENV) or DEFAULT_REACHY_DAEMON_BASE_URL
        self._base_url = resolved_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def upload_sound(self, *, filename: str, audio_bytes: bytes) -> str:
        """Upload a WAV file to the daemon temp sound directory."""
        boundary = "----AGENTBoundary7MA4YWxkTrZu0gW"
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        body = prefix + audio_bytes + f"\r\n--{boundary}--\r\n".encode()
        payload = await asyncio.to_thread(
            self._request_json,
            "/api/media/sounds/upload",
            method="POST",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        data = payload.get("data") or {}
        return str(data.get("path") or data.get("file") or filename)

    async def play_sound(self, *, file: str) -> dict[str, Any]:
        """Start daemon-side sound playback."""
        body = json.dumps({"file": file}).encode()
        return await asyncio.to_thread(
            self._request_json,
            "/api/media/play_sound",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )

    async def stop_sound(self) -> dict[str, Any]:
        """Stop daemon-side sound playback."""
        return await asyncio.to_thread(self._request_json, "/api/media/stop_sound", method="POST")

    async def delete_sound(self, *, filename: str) -> dict[str, Any]:
        """Delete an uploaded daemon temp sound by filename."""
        quoted = urllib.parse.quote(filename)
        return await asyncio.to_thread(self._request_json, f"/api/media/sounds/{quoted}", method="DELETE")

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            method=method,
            headers=headers or {},
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            raw = response.read().decode("utf-8", "replace")
            return {"status": response.status, "data": json.loads(raw) if raw else None}


class ReachyHttpMediaPlaybackAdapter:
    """Playback adapter using Reachy's daemon HTTP upload/play/delete sound endpoints."""

    def __init__(
        self,
        media_client: HttpMediaClient | None = None,
        *,
        live_speaker_enabled: bool = False,
        sleep: SleepFn = asyncio.sleep,
        filename_prefix: str = "agent_tts_chunk",
    ) -> None:
        """Initialize an HTTP media playback adapter behind an explicit live-speaker gate."""
        self._media_client = media_client or ReachyHttpMediaClient()
        self._live_speaker_enabled = live_speaker_enabled
        self._sleep = sleep
        self._filename_prefix = filename_prefix

    async def play(self, chunk: PlaybackChunk) -> PlaybackResult:
        """Upload, play, wait for duration, and delete one WAV chunk when gated."""
        if not self._live_speaker_enabled:
            raise LiveSpeakerGateRequired("Reachy HTTP speaker playback requires explicit live_speaker_enabled=True")
        duration_seconds = _wav_duration_seconds(chunk.audio_bytes)
        filename = f"{self._filename_prefix}_{chunk.sequence}_{int(time.time() * 1000)}.wav"
        remote_file = await self._media_client.upload_sound(filename=filename, audio_bytes=chunk.audio_bytes)
        await self._media_client.play_sound(file=remote_file)
        if duration_seconds > 0:
            await self._sleep(duration_seconds)
        await self._media_client.delete_sound(filename=filename)
        return PlaybackResult(
            sequence=chunk.sequence,
            audio_bytes=len(chunk.audio_bytes),
            side_effects=[
                "daemon_temp_sound_upload",
                "speaker_playback",
                "daemon_temp_sound_delete_cleanup",
            ],
        )

    async def cancel_current(self, reason: str) -> None:
        """Stop daemon-side playback for barge-in or abort."""
        await self._media_client.stop_sound()


class ReachyMediaPlaybackAdapter:
    """Playback adapter that pushes decoded WAV audio into Reachy media after an explicit gate."""

    def __init__(self, reachy: Any, *, live_speaker_enabled: bool = False) -> None:
        """Initialize with a Reachy-like object exposing `.media`."""
        self._reachy = reachy
        self._live_speaker_enabled = live_speaker_enabled

    async def play(self, chunk: PlaybackChunk) -> PlaybackResult:
        """Decode a WAV chunk and push it into Reachy's media player when gated."""
        if not self._live_speaker_enabled:
            raise LiveSpeakerGateRequired("Reachy live speaker playback requires explicit live_speaker_enabled=True")
        audio_frame, sample_rate_hz = _decode_wav_to_float32(chunk.audio_bytes)
        output_sample_rate = int(self._reachy.media.get_output_audio_samplerate())
        if output_sample_rate != sample_rate_hz:
            audio_frame = _resample_linear(audio_frame, sample_rate_hz, output_sample_rate)
        self._reachy.media.push_audio_sample(audio_frame)
        return PlaybackResult(
            sequence=chunk.sequence,
            audio_bytes=len(chunk.audio_bytes),
            side_effects=["reachy_media_push_audio_sample"],
        )

    async def cancel_current(self, reason: str) -> None:
        """Clear Reachy's player queue when available."""
        media = self._reachy.media
        audio = getattr(media, "audio", None)
        if audio is not None and hasattr(audio, "clear_player") and callable(audio.clear_player):
            audio.clear_player()
            return
        if hasattr(media, "clear_player") and callable(media.clear_player):
            media.clear_player()
            return
        if hasattr(media, "stop_playing") and callable(media.stop_playing):
            media.stop_playing()


class CancellablePlaybackQueue:
    """Serialize playback chunks and support pending/current cancellation."""

    def __init__(
        self,
        playback_adapter: PlaybackAdapter,
        *,
        latency_recorder: LatencyEventRecorder | None = None,
        turn_id: str = "playback-queue",
    ) -> None:
        """Initialize playback queue with an injectable backend."""
        self._adapter = playback_adapter
        self._latency_recorder = latency_recorder
        self._turn_id = turn_id
        self._queue: asyncio.Queue[PlaybackChunk | None] = asyncio.Queue()
        self._completed_results: list[PlaybackResult] = []
        self._first_playback_recorded = False
        self._pending_cancel_reason: str | None = None
        self._pending_cancelled_count = 0
        self._stop_requested = False

    @property
    def completed_results(self) -> list[PlaybackResult]:
        """Return completed playback results."""
        return list(self._completed_results)

    @property
    def playback_side_effects(self) -> list[str]:
        """Return flattened side-effect labels reported by the playback backend."""
        effects: list[str] = []
        for result in self._completed_results:
            effects.extend(result.side_effects)
        return effects

    async def enqueue(self, chunk: PlaybackChunk) -> None:
        """Queue one playback chunk."""
        if self._stop_requested:
            return
        await self._queue.put(chunk)

    def cancel_pending(self, *, reason: str = "cancelled") -> int:
        """Drop queued chunks that have not started playback."""
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
        self._pending_cancelled_count += cancelled
        return cancelled

    async def cancel_current(self, *, reason: str = "cancelled") -> None:
        """Ask the backend to cancel the in-flight playback chunk."""
        await self._adapter.cancel_current(reason)

    async def run(self) -> None:
        """Process playback chunks until stopped."""
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                await self._play_one(item)
                self._flush_pending_cancel_event()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A playback/adapter error must not kill the worker: a dead worker leaves later
                # items unprocessed and join() hangs forever. Log, record, keep going.
                logger.warning("Playback chunk failed: %r", exc)
                self._record("playback_chunk_failed", error=type(exc).__name__)
            finally:
                self._queue.task_done()

    async def join(self) -> None:
        """Wait until current queue work has finished."""
        await self._queue.join()

    async def stop(self) -> None:
        """Request worker shutdown after current item."""
        self._stop_requested = True
        await self._queue.put(None)

    async def _play_one(self, chunk: PlaybackChunk) -> None:
        if not self._first_playback_recorded:
            self._record("first_playback_request", audio_bytes=len(chunk.audio_bytes), sequence=chunk.sequence)
            self._first_playback_recorded = True
        try:
            result = await self._adapter.play(chunk)
        except PlaybackCancelled as exc:
            self._record("playback_current_cancelled", reason=exc.reason, sequence=chunk.sequence)
            return
        self._completed_results.append(result)
        self._record("playback_chunk_done", audio_bytes=result.audio_bytes, sequence=result.sequence)

    def _flush_pending_cancel_event(self) -> None:
        if self._pending_cancel_reason is None:
            return
        self._record(
            "playback_queue_cancelled",
            reason=self._pending_cancel_reason,
            cancelled_chunks=self._pending_cancelled_count,
        )
        self._pending_cancel_reason = None
        self._pending_cancelled_count = 0

    def _record(self, event_type: str, **payload) -> None:  # type: ignore[no-untyped-def]
        if self._latency_recorder is None:
            return
        self._latency_recorder.record(event_type, turn_id=self._turn_id, **payload)


async def run_no_speaker_playback_smoke(*, chunks: list[PlaybackChunk]) -> NoSpeakerPlaybackSmokeResult:
    """Run encoded chunks through null playback without speaker side effects."""
    adapter = NullPlaybackAdapter()
    queue = CancellablePlaybackQueue(adapter)
    worker = asyncio.create_task(queue.run())
    for chunk in chunks:
        await queue.enqueue(chunk)
    await queue.join()
    await queue.stop()
    await worker
    return NoSpeakerPlaybackSmokeResult(
        chunk_count=len(adapter.played_chunks),
        total_audio_bytes=sum(len(chunk.audio_bytes) for chunk in adapter.played_chunks),
        playback_side_effects=queue.playback_side_effects,
        played_chunks=adapter.played_chunks,
    )


def _wav_duration_seconds(audio_bytes: bytes) -> float:
    """Return WAV duration in seconds for bounded HTTP playback waits."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        sample_rate_hz = wav.getframerate()
        frames = wav.getnframes()
    if sample_rate_hz <= 0:
        return 0.0
    return frames / sample_rate_hz


def _decode_wav_to_float32(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode mono/stereo PCM WAV bytes to mono float32 samples."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate_hz = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV playback is supported, got sample_width={sample_width}")
    pcm = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    audio_frame = pcm.astype(np.float32) / 32768.0
    return audio_frame, sample_rate_hz


def _resample_linear(audio_frame: np.ndarray, input_rate_hz: int, output_rate_hz: int) -> np.ndarray:
    """Resample a mono float32 frame with deterministic linear interpolation."""
    if len(audio_frame) == 0:
        return audio_frame
    target_len = max(1, int(round(len(audio_frame) * output_rate_hz / input_rate_hz)))
    if target_len == len(audio_frame):
        return audio_frame
    src_positions = np.linspace(0.0, 1.0, num=len(audio_frame), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(target_positions, src_positions, audio_frame).astype(np.float32)
