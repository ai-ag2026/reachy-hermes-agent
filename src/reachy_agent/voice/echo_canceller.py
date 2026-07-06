"""Acoustic Echo Cancellation (AEC) for barge-in — ported from sparky (NLMS, numpy-only).

Removes the robot's own TTS playback from the mic so Silero VAD can detect real human speech while
AGENT is talking (prerequisite for barge-in, Phase A). NLMS adaptive filter, pure numpy (no scipy
needed if you feed 16kHz PCM via feed_speaker_pcm). If the pure-Python per-sample loop is not
real-time on the CM4, fall back to the WebRTC AEC (libwebrtc-audio-processing is installed on the
robot) — see docs/findings/2026-06-24_phaseA-aec.md.

Usage:
    aec = AcousticEchoCanceller()
    aec.feed_speaker_pcm(ref_16k_int16_bytes)   # what we send to the speaker
    cleaned = aec.process_mic_chunk(mic_16k_int16_bytes)
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class AcousticEchoCanceller:
    """Thread-safe NLMS AEC. Speaker reference is buffered and consumed frame-by-frame as the mic is
    processed; when no reference is available (robot silent) the mic passes through unchanged."""

    def __init__(
        self, frame_size: int = 160, filter_length: int = 768, sample_rate: int = 16000, mu: float = 0.3
    ) -> None:
        # filter_length 768 = 48ms echo tail: real-time on the CM4 (0.65x) with strong suppression;
        # the Reachy mic+speaker are cm apart (short echo path). 3200 (200ms) is NOT real-time in
        # pure Python on the CM4. See docs/findings/2026-06-24_phaseA-aec.md.
        self._frame_size = frame_size  # 160 = 10ms @ 16kHz
        self._filter_length = filter_length
        self._sample_rate = sample_rate
        self._mu = mu
        self._w = np.zeros(filter_length, dtype=np.float64)
        self._x_hist = np.zeros(filter_length, dtype=np.float64)
        self._speaker_buf = bytearray()
        self._lock = threading.Lock()
        self._frames_processed = 0
        self._frames_with_ref = 0

    def feed_speaker_pcm(self, pcm_int16: bytes) -> None:
        """Feed raw 16kHz int16 mono PCM that is being sent to the speaker."""
        with self._lock:
            self._speaker_buf.extend(pcm_int16)

    def process_mic_chunk(self, mic_pcm: bytes) -> bytes:
        """Clean an arbitrary-length mic chunk (16kHz int16 mono). Returns same-length PCM."""
        frame_bytes = self._frame_size * 2
        if len(mic_pcm) < frame_bytes:
            return mic_pcm
        result = bytearray()
        offset = 0
        while offset + frame_bytes <= len(mic_pcm):
            result.extend(self._process_frame(mic_pcm[offset : offset + frame_bytes]))
            offset += frame_bytes
        if offset < len(mic_pcm):
            result.extend(mic_pcm[offset:])
        return bytes(result)

    def _process_frame(self, mic_frame: bytes) -> bytes:
        frame_bytes = self._frame_size * 2
        self._frames_processed += 1
        with self._lock:
            if len(self._speaker_buf) >= frame_bytes:
                speaker_data = bytes(self._speaker_buf[:frame_bytes])
                del self._speaker_buf[:frame_bytes]
                has_ref = True
            else:
                speaker_data = None
                has_ref = False
        mic = np.frombuffer(mic_frame, dtype=np.int16).astype(np.float64)
        if not has_ref:
            return mic_frame  # robot silent -> passthrough
        self._frames_with_ref += 1
        speaker = np.frombuffer(speaker_data, dtype=np.int16).astype(np.float64)
        output = np.zeros(self._frame_size, dtype=np.float64)
        x_hist = self._x_hist
        w = self._w
        for i in range(self._frame_size):
            x_hist[1:] = x_hist[:-1]  # in-place shift (no per-sample allocation, unlike np.roll)
            x_hist[0] = speaker[i]
            echo_est = np.dot(w, x_hist)
            error = mic[i] - echo_est
            output[i] = error
            norm = np.dot(x_hist, x_hist) + 1e-6
            w += (self._mu * error / norm) * x_hist  # in-place -> updates self._w
        return np.clip(output, -32768, 32767).astype(np.int16).tobytes()

    def clear(self) -> None:
        """Reset on playback cancel (clears reference + filter to avoid stale adaptation)."""
        with self._lock:
            self._speaker_buf.clear()
        self._w[:] = 0.0
        self._x_hist[:] = 0.0

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frames_processed,
            "frames_with_ref": self._frames_with_ref,
            "speaker_buf_bytes": len(self._speaker_buf),
            "filter_energy": float(np.dot(self._w, self._w)),
        }
