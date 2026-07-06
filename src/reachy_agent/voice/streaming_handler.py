"""AGENT streaming voice handler = app's AgentVoiceHandler + in-receive() STT front-end.

Lives at the SSoT↔app boundary. The app's `AgentVoiceHandler` is imported lazily (inside the factory)
so this SSoT package does not hard-require the conversation app to be importable. In the app/overlay,
build the handler via ``make_streaming_handler_class()`` and select it for BACKEND_PROVIDER=agent.

receive(mic_frame) feeds frames to `AgentSttFrontend` (Silero VAD endpoint + remote Parakeet :5093);
on an endpointed utterance it runs `handle_final_transcript` (AGENT brain -> Qwen TTS), proven in sim
(scripts/m2_b3_full_receive_chain.py). AEC (sparky webrtc_aecm) + s2s progressive-partials = M5/latency.
"""

from __future__ import annotations

from .stt_frontend import AgentSttFrontend


def make_streaming_handler_class():
    """Return a AgentVoiceHandler subclass that lets you INJECT a custom STT front-end.

    Lazy import keeps the SSoT package independent of the conversation app at import time.

    The subclass no longer overrides ``receive()``: the parent ``AgentVoiceHandler.receive``
    is the canonical live path (stereo→mono, float32→int16 scaling, ``to_thread`` off the
    event loop, turn/barge gating). An earlier standalone override re-implemented a naive
    ``feed(reshape(-1))`` that fed scrambled, truncated audio to the VAD (deaf) and blocked the
    loop on the remote STT round-trip. We instead seed the parent's lazily-built ``_stt`` with
    the injected front-end so there is a single source of truth for receive().
    """
    from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler

    class AgentStreamingVoiceHandler(AgentVoiceHandler):
        def __init__(self, deps, *, agent_client, tts_client, stt_frontend=None, **kw):
            super().__init__(deps, agent_client=agent_client, tts_client=tts_client, **kw)
            # Seed the parent's STT slot; _ensure_stt() (called in receive) returns it as-is.
            self._stt = stt_frontend or AgentSttFrontend()
            self.stt_frontend = self._stt  # back-compat alias for callers/tests

    return AgentStreamingVoiceHandler
