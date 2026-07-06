"""Streaming STT integration (Nemotron WS) — client flow + frontend stream-mode routing."""

import sys
import types

import numpy as np

from reachy_agent.voice.nemotron_stream import NemotronStreamClient


class _FakeWS:
    """Records sends; replies with a partial per binary chunk and a final on 'flush'."""

    def __init__(self):
        self.sent = []
        self._final = "hallo agent alles klar"

    def send_binary(self, b):
        self.sent.append(("bin", len(b)))

    def send(self, s):
        self.sent.append(("txt", s))

    def recv(self):
        # last send decides the reply
        kind, val = self.sent[-1]
        if kind == "txt" and val == "flush":
            return '{"final": "%s"}' % self._final
        return '{"partial": "hallo agent"}'

    def close(self):
        self.sent.append(("close", None))


def _install_fake_websocket(monkeypatch, fake):
    mod = types.ModuleType("websocket")
    mod.create_connection = lambda url, timeout=None: fake
    monkeypatch.setitem(sys.modules, "websocket", mod)


def test_stream_client_push_and_finish(monkeypatch):
    fake = _FakeWS()
    _install_fake_websocket(monkeypatch, fake)
    c = NemotronStreamClient("ws://x/stream", tail_pad_s=0.2)
    assert c.start() is True and c.active
    c.push(np.zeros(512, dtype=np.float32))
    assert c.partial == "hallo agent"  # partial drained from server reply
    final = c.finish(sample_rate=16000)
    assert final == "hallo agent alles klar"
    assert ("txt", "flush") in fake.sent  # flush was sent
    assert not c.active  # closed after finish


def test_stream_client_connect_failure_returns_false(monkeypatch):
    mod = types.ModuleType("websocket")

    def _boom(url, timeout=None):
        raise OSError("refused")

    mod.create_connection = _boom
    monkeypatch.setitem(sys.modules, "websocket", mod)
    c = NemotronStreamClient("ws://x/stream")
    assert c.start() is False and not c.active


def test_frontend_stream_mode_finish_uses_stream(monkeypatch):
    import pytest

    pytest.importorskip("onnxruntime")  # AgentSttFrontend builds the Silero VAD (app venv only)
    # Inject a fake stream client class so the frontend (AGENT_STT_MODE=stream) routes finish to it.
    import reachy_agent.voice.nemotron_stream as ns

    class _FakeStream:
        def __init__(self, *a, **k):
            self.active = False

        def start(self):
            self.active = True
            return True

        def finish(self, sample_rate=16000):
            return "hallo thars"  # mishearing -> lexicon should fix to AGENT

    monkeypatch.setattr(ns, "NemotronStreamClient", _FakeStream)
    monkeypatch.setenv("AGENT_STT_MODE", "stream")

    from reachy_agent.voice.stt_frontend import AgentSttFrontend

    fe = AgentSttFrontend()
    assert fe._stream is not None
    # simulate an endpointed utterance in stream mode
    fe._stream.start()
    fe._in_speech = True
    fe._speech = [1] * 2000
    fe._speech_chunks = fe.min_speech_chunks + 1
    out = fe._finish_utterance()
    assert out == "hallo AGENT"  # stream final, lexicon-corrected; batch engine not consulted


def test_frontend_batch_mode_default_has_no_stream(monkeypatch):
    import pytest

    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("AGENT_STT_MODE", raising=False)
    from reachy_agent.voice.stt_frontend import AgentSttFrontend

    fe = AgentSttFrontend()
    assert fe._stream is None and fe._mode == "batch"


def test_stream_client_abort_closes_without_flush(monkeypatch):
    fake = _FakeWS()
    fake.shutdown = lambda: fake.sent.append(("shutdown", None))
    _install_fake_websocket(monkeypatch, fake)
    c = NemotronStreamClient("ws://x/stream")
    assert c.start() is True
    c.push(np.zeros(512, dtype=np.float32))
    c.abort()
    assert not c.active and c.partial == ""
    kinds = [k for k, _ in fake.sent]
    assert "shutdown" in kinds
    assert ("txt", "flush") not in fake.sent  # no pad/flush round-trips on abort


def test_frontend_reset_and_noise_discard_abort_not_finish(monkeypatch):
    import pytest

    pytest.importorskip("onnxruntime")
    import reachy_agent.voice.nemotron_stream as ns

    class _FakeStream:
        def __init__(self, *a, **k):
            self.active = False
            self.aborted = 0
            self.finished = 0
            self.partial = ""

        def start(self):
            self.active = True
            return True

        def abort(self):
            self.aborted += 1
            self.active = False

        def finish(self, sample_rate=16000):
            self.finished += 1
            self.active = False
            return "x"

    monkeypatch.setattr(ns, "NemotronStreamClient", _FakeStream)
    monkeypatch.setenv("AGENT_STT_MODE", "stream")
    from reachy_agent.voice.stt_frontend import AgentSttFrontend

    # reset() mid-utterance must abort (cheap, called from the handler's async path), never
    # pay finish()'s pad+flush round-trips (review 2026-07-02 round 2, P1-3).
    fe = AgentSttFrontend()
    fe._stream.start()
    fe._in_speech = True
    fe.reset()
    assert fe._stream.aborted == 1 and fe._stream.finished == 0

    # noise/short discard in _finish_utterance must abort too
    fe2 = AgentSttFrontend()
    fe2._stream.start()
    fe2._in_speech = True
    fe2._speech = [1] * 100
    fe2._speech_chunks = 1  # below min_speech_chunks -> noise discard
    assert fe2._finish_utterance() is None
    assert fe2._stream.aborted == 1 and fe2._stream.finished == 0


def test_finish_distinguishes_silence_from_transport_error(monkeypatch):
    # silence: server returns empty final -> "" (not None)
    fake = _FakeWS()
    fake._final = ""
    _install_fake_websocket(monkeypatch, fake)
    c = NemotronStreamClient("ws://x/stream", tail_pad_s=0.1)
    assert c.start()
    assert c.finish(sample_rate=16000) == ""

    # transport error at flush -> None (caller falls back to batch)
    fake2 = _FakeWS()

    def _boom(*a, **k):
        raise OSError("connection reset")

    fake2.send = _boom
    _install_fake_websocket(monkeypatch, fake2)
    c2 = NemotronStreamClient("ws://x/stream", tail_pad_s=0.1)
    assert c2.start()
    assert c2.finish(sample_rate=16000) is None


def test_connect_backoff_after_consecutive_failures(monkeypatch):
    mod = types.ModuleType("websocket")
    attempts = {"n": 0}

    def _refuse(url, timeout=None):
        attempts["n"] += 1
        raise OSError("refused")

    mod.create_connection = _refuse
    monkeypatch.setitem(sys.modules, "websocket", mod)
    c = NemotronStreamClient("ws://x/stream", retry_cooldown_s=60.0)
    assert c.start() is False and attempts["n"] == 1
    assert c.start() is False and attempts["n"] == 2  # second real attempt -> arms cooldown
    assert c.start() is False and attempts["n"] == 2  # cooling down: NO connect attempt
    c._retry_at = 0.0  # cooldown elapsed
    assert c.start() is False and attempts["n"] == 3


def test_frontend_flush_error_falls_back_to_batch(monkeypatch):
    import pytest

    pytest.importorskip("onnxruntime")
    import reachy_agent.voice.nemotron_stream as ns

    class _FlushFailStream:
        def __init__(self, *a, **k):
            self.active = False
            self.partial = ""

        def start(self):
            self.active = True
            return True

        def finish(self, sample_rate=16000):
            self.active = False
            return None  # transport error at flush

        def abort(self):
            self.active = False

    monkeypatch.setattr(ns, "NemotronStreamClient", _FlushFailStream)
    monkeypatch.setenv("AGENT_STT_MODE", "stream")
    from reachy_agent.voice.stt_frontend import AgentSttFrontend

    fe = AgentSttFrontend()

    class _BatchResult:
        text = "hallo thars"

    fe.stt = type("_E", (), {"transcribe": staticmethod(lambda wav: _BatchResult())})()
    fe._stream.start()
    fe._in_speech = True
    fe._speech = [1] * 2000
    fe._speech_chunks = fe.min_speech_chunks + 1
    # flush fails -> batch path transcribes the buffered PCM (lexicon-corrected as usual)
    assert fe._finish_utterance() == "hallo AGENT"


def test_chirp_gain_applied_per_call():
    from reachy_agent.voice.astromech import _CACHE, chirp

    _CACHE.clear()
    loud = chirp("acknowledge", gain=0.8)
    quiet = chirp("acknowledge", gain=0.2)  # same cache entry, different gain
    assert abs(int(np.abs(loud).max()) - 0.8 * 32767) <= 1
    assert abs(int(np.abs(quiet).max()) - 0.2 * 32767) <= 1
