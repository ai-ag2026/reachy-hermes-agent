"""Synchronous WebSocket client for the Nemotron streaming-ASR server (:5094 /v1/stream).

Runs inside AgentSttFrontend.feed() (which the handler calls via asyncio.to_thread), so it is
deliberately SYNC (websocket-client, not the async websockets lib). One session per utterance:
open at speech onset, push 16 kHz float32 PCM chunks as they are voiced (transcription happens
DURING speech), then finish() pads a little trailing silence and flushes — so the FINAL transcript
is ready ~0.15 s after speech ends instead of a full batch transcription (measured 2026-07-02).

The running ``partial`` is kept for a future preemptive turn-start (Stage 2); Stage 1 only uses the
final. Any transport error degrades to None — the frontend then falls back to batch STT with the
locally buffered PCM (mid-utterance failures land there via ``active=False``, flush failures via
``finish() is None``).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NemotronStreamClient:
    """One streaming ASR session over a sync WebSocket. Reusable across utterances (start/finish)."""

    def __init__(
        self,
        url: str,
        *,
        tail_pad_s: float = 0.6,
        connect_timeout: float = 3.0,
        retry_cooldown_s: float = 60.0,
    ) -> None:
        self.url = url
        self.tail_pad_s = tail_pad_s
        self.connect_timeout = connect_timeout
        # Connect backoff: with :5094 down, EVERY utterance paid the full connect timeout (~3s)
        # at speech onset, stalling the feed thread forever (review 2026-07-02 round 2, P2).
        # After 2 consecutive failures, don't even try for a cooldown window (batch STT covers).
        self.retry_cooldown_s = retry_cooldown_s
        self._connect_failures = 0
        self._retry_at = 0.0
        self._ws = None
        self.partial = ""

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> bool:
        """Open a fresh session. Returns False on failure (caller falls back / drops)."""
        import time

        self._close()
        self.partial = ""
        if time.monotonic() < self._retry_at:
            return False  # cooling down after consecutive connect failures — don't stall the mic
        try:
            from websocket import create_connection  # websocket-client (sync)

            self._ws = create_connection(self.url, timeout=self.connect_timeout)
            self._connect_failures = 0
            return True
        except Exception as exc:
            self._connect_failures += 1
            if self._connect_failures >= 2:
                self._retry_at = time.monotonic() + self.retry_cooldown_s
                logger.warning(
                    "nemotron stream: connect failed %d× (%s): %r — cooling down %.0fs",
                    self._connect_failures,
                    self.url,
                    exc,
                    self.retry_cooldown_s,
                )
            else:
                logger.warning("nemotron stream: connect failed (%s): %r", self.url, exc)
            self._ws = None
            return False

    @property
    def active(self) -> bool:
        return self._ws is not None

    def push(self, pcm16k: np.ndarray) -> None:
        """Send one 16 kHz float32 PCM chunk; drain the server's partial reply (keeps it current)."""
        if self._ws is None:
            return
        try:
            import json

            self._ws.send_binary(np.ascontiguousarray(pcm16k, dtype=np.float32).tobytes())
            msg = self._ws.recv()  # server replies {"partial": ...} per chunk (backpressure drain)
            if msg:
                d = json.loads(msg)
                if "partial" in d:
                    self.partial = d["partial"]
        except Exception as exc:
            logger.warning("nemotron stream: push failed: %r", exc)
            self._close()

    def abort(self) -> None:
        """Drop the session immediately — no tail pad, no flush, no waiting on the server.

        Use whenever the transcript is not wanted (reset between turns, noise discard,
        barge-routing takeover): ``finish()`` pays ~0.3-1s of pad-chunk round-trips (up to the
        socket timeout on a wedged server), which froze the event loop when called from the
        handler's async path (review 2026-07-02 round 2, P1-3)."""
        self.partial = ""
        if self._ws is not None:
            try:
                # shutdown() closes the socket without the close-handshake wait of close().
                self._ws.shutdown()
            except Exception:
                pass
            self._ws = None

    def finish(self, sample_rate: int = 16000) -> str | None:
        """Pad trailing silence (so the last word decodes), flush, return the final transcript.

        Returns the transcript string on success (may be "" for silence) and None ONLY on a
        transport error — the caller uses that distinction to fall back to batch STT with the
        locally buffered PCM instead of dropping the utterance (review 2026-07-02 round 2, P2:
        a server restart exactly at the flush lost the sentence although batch was ready)."""
        if self._ws is None:
            return None
        try:
            import json

            pad = np.zeros(int(sample_rate * self.tail_pad_s), dtype=np.float32)
            step = int(sample_rate * 0.1) or 1
            for i in range(0, len(pad), step):
                self._ws.send_binary(pad[i : i + step].tobytes())
                self._ws.recv()
            self._ws.send("flush")
            msg = self._ws.recv()
            final = json.loads(msg).get("final", "") if msg else ""
            return (final or "").strip()
        except Exception as exc:
            logger.warning("nemotron stream: finish failed: %r", exc)
            return None
        finally:
            self._close()

    def _close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
