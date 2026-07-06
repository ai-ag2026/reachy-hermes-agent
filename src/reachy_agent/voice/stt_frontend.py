"""In-receive() streaming STT front-end for the AGENT voice backend (sparky backbone).

Push interface for the app's ``ConversationHandler.receive(frame)``: feed mic audio frames;
Silero VAD endpoints the utterance (speech start + trailing-silence end); the captured WAV is
transcribed by Parakeet (remote_http :5093). `feed()` returns the transcript when an utterance ends,
else None — the caller (AgentVoiceHandler.receive) then calls handle_final_transcript(transcript).

Reuses the ported sparky modules (`vad_capture.VAD` Silero v5, `stt_engine.STTEngine` remote_http →
our AI-VM Parakeet) per the 2026-06-23 backbone decision. AEC (sparky `webrtc_aecm`) is the
hardware add-on for the open-mic-next-to-speaker case (M5). s2s's progressive-partials is a future
low-latency enhancement; v0.1 emits a final transcript on endpoint.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .lexicon import correct_transcript
from .stt_engine import STTEngine
from .vad_capture import VAD

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = str(Path(__file__).parent / "models" / "silero_vad_v5.onnx")
_CHUNK = 512  # Silero VAD chunk @ 16kHz = 32ms


class AgentSttFrontend:
    """Push-driven VAD-endpointed STT front-end (Silero VAD + remote Parakeet)."""

    def __init__(
        self,
        *,
        stt_remote_url: str | None = None,  # default from $AGENT_STT_BASE_URL (preset-controlled)
        stt_model: str = "parakeet",
        model_path: str = _DEFAULT_MODEL,
        vad_threshold: float = 0.5,
        pause_ms: int = 700,
        target_sr: int = 16000,
        min_speech_chunks: int = 4,
        language: str | None = None,  # default from $AGENT_STT_LANGUAGE; "auto" to let the model detect
        vad_gain: float | None = None,  # gain applied ONLY to the VAD decision (not the STT audio)
        max_utterance_s: float = 30.0,  # hard cap so a never-ending speech run can't grow unbounded
    ) -> None:
        import os

        stt_remote_url = stt_remote_url or os.getenv("AGENT_STT_BASE_URL", "http://127.0.0.1:5093/v1")
        # Gain helps Silero detect quiet speech, but amplifying/clipping the audio sent to Parakeet
        # degrades transcription (noisy German gets mis-detected as phonetic English). So boost ONLY
        # the VAD's view; keep the captured speech at its original level for the STT.
        self.vad_gain = vad_gain if vad_gain is not None else float(os.getenv("AGENT_MIC_GAIN", "2.0"))
        # STTEngine defaults language to "en" -> Parakeet was told the speech is English and garbled
        # German into phonetic English ("We start to get single at last."). Force German by default;
        # AGENT_STT_LANGUAGE=auto restores model auto-detect, or set another code.
        language = language if language is not None else os.getenv("AGENT_STT_LANGUAGE", "de")
        self.vad = VAD(model_path)
        # Short remote timeout (default 4s, was the engine's 10s): while transcribe runs, the
        # record_loop consumes no mic frames — a hung Parakeet meant up to 10s of mic blackout
        # (audit 2026-07-02). Normal transcriptions finish well under 1s.
        stt_timeout = float(os.getenv("AGENT_STT_TIMEOUT_S", "4"))
        self.stt = STTEngine(
            engine="remote_http",
            remote_url=stt_remote_url,
            remote_model=stt_model,
            language=language,
            remote_timeout_s=stt_timeout,
        )
        # Streaming mode (Nemotron): instead of buffering the whole utterance and batch-transcribing
        # at endpoint, open a WS session at speech onset and push chunks as they are voiced, so the
        # final transcript is ready ~0.15s after speech ends (vs a full batch pass). AGENT_STT_MODE=stream.
        # Batch (Parakeet) stays the default. The VAD endpointing logic is identical in both modes.
        self._mode = os.getenv("AGENT_STT_MODE", "batch").strip().lower()
        self._stream = None
        if self._mode == "stream":
            from .nemotron_stream import NemotronStreamClient

            stream_url = os.getenv("AGENT_STT_STREAM_URL", "ws://127.0.0.1:5094/v1/stream")
            self._stream = NemotronStreamClient(stream_url)
        self.target_sr = target_sr
        self.threshold = vad_threshold
        self.pause_chunks = max(1, int(pause_ms / 32))
        self.min_speech_chunks = min_speech_chunks
        # Cap the captured-speech buffer: without a trailing pause (open-mic echo/noise) the int list
        # would grow for the whole session, then materialize a giant STT payload. Force-endpoint at this.
        self.max_speech_samples = max(target_sr, int(max_utterance_s * target_sr))
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech: list[int] = []
        self._pre: list[int] = []
        self._in_speech = False
        self._silence = 0
        self._speech_chunks = 0
        self._last_partial = ""  # stream mode: track partial stability for a preemptive turn-start
        self._partial_stable = 0  # consecutive feed() frames the partial has been unchanged

    @property
    def in_speech(self) -> bool:
        """True while a voiced utterance is currently in progress — for reopen/turn-continuation logic."""
        return self._in_speech

    @property
    def partial(self) -> str:
        """Latest streaming partial transcript (stream mode only; '' otherwise)."""
        return self._stream.partial if self._stream is not None else ""

    def stable_partial(self, *, min_chars: int = 15, stable_frames: int = 6) -> Optional[str]:
        """Return the partial for a preemptive turn-start, else None (stream mode only).

        Gated on the user having ALREADY PAUSED (``_silence`` past half the endpoint window): during
        speech the partial settles one word behind the final between words, so launching on the first
        stable partial always mismatched the final and wasted the turn (live test 2026-07-02). Only
        once trailing silence is under way is the last word emitted and the partial == the final —
        so the speculative turn is adopted. Head start ≈ the remaining half of the endpoint pause."""
        if not self._in_speech or self._stream is None:
            return None
        if self._silence < max(2, self.pause_chunks // 2):
            return None  # user hasn't paused yet -> partial still trails the final by the last word
        p = self._last_partial.strip()
        if self._partial_stable >= stable_frames and len(p) >= min_chars:
            # Lexicon-correct like the final: the adopt-compare normalizes partial vs final, and
            # an uncorrected "hallo thars" NEVER matched the corrected "hallo AGENT" final — the
            # speculation was discarded on every alias-bearing utterance, i.e. exactly the
            # address phrases the feature exists for (review 2026-07-02 round 2, P2).
            return correct_transcript(p)
        return None

    def reset(self) -> None:
        """Clear all buffered/endpointing state (use when reusing a front-end across turns, e.g. a
        barge-in monitor that may not have endpointed last turn)."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech = []
        self._pre = []
        self._in_speech = False
        self._silence = 0
        self._speech_chunks = 0
        self._last_partial = ""
        self._partial_stable = 0
        if self._stream is not None and self._stream.active:
            # abort, not finish: reset() is called from the handler's async path (barge takeover,
            # proactive playback, mute-lift) — finish() does blocking pad+flush round-trips and
            # froze the event loop for up to the socket timeout (review 2026-07-02 round 2, P1-3).
            self._stream.abort()
        try:
            self.vad.reset_states()
        except Exception:
            pass

    def _to16k(self, sr: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).astype(np.int16)
        if sr == self.target_sr or len(x) == 0:
            return x
        n = int(len(x) * self.target_sr / sr)
        idx = np.linspace(0, len(x), n, endpoint=False)
        return np.interp(idx, np.arange(len(x)), x.astype(np.float32)).astype(np.int16)

    def feed(self, sample_rate: int, samples_int16: np.ndarray) -> Optional[str]:
        """Feed one audio frame. Returns a transcript when an utterance ends, else None."""
        x = self._to16k(sample_rate, samples_int16)
        self._buf = np.concatenate([self._buf, x.astype(np.float32) / 32768.0])
        transcript: Optional[str] = None
        while len(self._buf) >= _CHUNK:
            chunk = self._buf[:_CHUNK]
            self._buf = self._buf[_CHUNK:]
            vad_chunk = np.clip(chunk * self.vad_gain, -1.0, 1.0) if self.vad_gain != 1.0 else chunk
            conf = float(self.vad(vad_chunk.reshape(1, -1), self.target_sr))  # VAD expects (batch, 512)
            self._dbg_n = getattr(self, "_dbg_n", 0) + 1
            self._dbg_cmax = max(getattr(self, "_dbg_cmax", 0.0), conf)
            if self._dbg_n % 100 == 0:
                if os.environ.get("AGENT_VOICE_DEBUG"):
                    logger.info(
                        "STT vad: chunk#%d conf_max(last100)=%.2f in_speech=%s",
                        self._dbg_n,
                        self._dbg_cmax,
                        self._in_speech,
                    )
                self._dbg_cmax = 0.0
            ci16 = (chunk * 32768.0).astype(np.int16)
            if conf >= self.threshold:
                if not self._in_speech:
                    self._in_speech = True
                    self._silence = 0
                    self._speech_chunks = 0
                    self._speech = list(self._pre)  # include pre-roll
                    logger.info("STT vad: SPEECH START (conf=%.2f)", conf)
                    if self._stream is not None and self._stream.start():
                        # stream the pre-roll (float32 @16k) so the utterance opening isn't lost
                        pre = np.array(self._pre, dtype=np.int16).astype(np.float32) / 32768.0
                        if len(pre):
                            self._stream.push(pre)
                self._speech.extend(ci16.tolist())
                self._speech_chunks += 1
                self._silence = 0
                if self._stream is not None and self._stream.active:
                    self._stream.push(chunk)
            elif self._in_speech:
                self._speech.extend(ci16.tolist())
                self._silence += 1
                if self._stream is not None and self._stream.active:
                    self._stream.push(chunk)  # push trailing frames too (context for the last words)
                if self._silence >= self.pause_chunks:
                    logger.info("STT vad: SPEECH END (spoke=%d chunks) -> transcribe", self._speech_chunks)
                    transcript = self._finish_utterance()
                    logger.info("STT transcript=%r", transcript)
            else:
                self._pre.extend(ci16.tolist())
                self._pre = self._pre[-_CHUNK * 12 :]  # ~0.4s pre-roll
            # Hard cap (after the if/elif/else): force-endpoint a runaway utterance that never pauses
            # (open-mic echo/noise) so the captured buffer can't grow for the whole session.
            if self._in_speech and len(self._speech) >= self.max_speech_samples:
                logger.warning(
                    "STT vad: max utterance length reached (%d samples) -> force transcribe", len(self._speech)
                )
                transcript = self._finish_utterance()
                logger.info("STT transcript=%r (forced cap)", transcript)
        # Track partial stability once per feed() frame (stream mode) for the preemptive turn-start.
        if self._stream is not None and self._in_speech:
            cur = self._stream.partial or ""
            if cur and cur == self._last_partial:
                self._partial_stable += 1
            else:
                self._last_partial = cur
                self._partial_stable = 0
        return transcript

    def _finish_utterance(self) -> Optional[str]:
        spoke = self._speech_chunks
        pcm = np.array(self._speech, dtype=np.int16)
        streaming = self._stream is not None and self._stream.active
        # reset state
        self._in_speech = False
        self._speech = []
        self._pre = []
        self._silence = 0
        self._speech_chunks = 0
        self._last_partial = ""
        self._partial_stable = 0
        try:
            self.vad.reset_states()
        except Exception:
            pass
        if spoke < self.min_speech_chunks or len(pcm) == 0:
            if streaming:
                self._stream.abort()  # discard short/noise — don't pay the pad+flush round-trips
            return None  # too short / noise — ignore
        # Streaming mode: the chunks were already sent during speech; finish() pads + flushes and the
        # final transcript comes back in ~0.15s (vs a full batch transcription). Lexicon-correct as usual.
        # finish() returns None ONLY on a transport error — then fall through to the batch path
        # below: the PCM is still local and Parakeet is ready (review 2026-07-02 round 2, P2).
        if streaming:
            text = self._stream.finish(sample_rate=self.target_sr)
            if text is not None:
                text = text.strip()
                return correct_transcript(text) if text else None
            logger.warning("STT stream flush failed — falling back to batch STT for this utterance")
        # feed() must never raise: the utterance state is already reset, so an escaping network
        # error silently swallowed the user's sentence. The PCM is still local — retry ONCE on a
        # transient failure, then give up with a warning (audit 2026-07-02).
        wav = self._to_wav(pcm)
        try:
            res = self.stt.transcribe(wav)
        except Exception as exc:
            logger.warning("STT transcribe failed (%r) — retrying once", exc)
            try:
                res = self.stt.transcribe(wav)
            except Exception as exc2:
                logger.warning("STT retry failed (%r) — utterance dropped", exc2)
                return None
        # Use res.text directly. The previous `getattr(res,"text") or str(res)` returned the STTResult
        # *repr* as a fake transcript when text was empty (noise) -> garbage committed on barge-in.
        # (Do NOT gate on is_speech: the remote_http result reports is_speech=False even for real
        # speech, so gating there suppressed all replies. Empty text -> None is the correct gate.)
        text = getattr(res, "text", None)
        if text is None:
            text = res if isinstance(res, str) else ""
        text = (text or "").strip()
        if not text:
            return None
        # Fix known proper-noun mishearings (AGENT/Reachy/Amalun/...) at the one STT
        # chokepoint — covers the main turn and the barge-in monitor alike.
        return correct_transcript(text)

    def _to_wav(self, pcm16: np.ndarray) -> bytes:
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.target_sr)
            w.writeframes(pcm16.tobytes())
        return b.getvalue()
