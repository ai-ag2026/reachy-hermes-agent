"""Local speech-to-speech event contract for the AGENT/Reachy MVP lane."""

from reachy_agent.local_s2s.events import (
    LocalS2SEvent,
    LocalS2SEventType,
    make_output_text_delta,
    make_response_cancelled,
    make_response_completed,
    make_transcription_completed,
    parse_local_s2s_event,
)
from reachy_agent.local_s2s.qwen_output import (
    QwenOutputBridge,
    TextChunkResult,
    collect_speakable_chunks,
)
from reachy_agent.local_s2s.runtime_smoke import (
    LocalS2SMvpSmokeReport,
    run_fake_local_s2s_mvp_smoke,
    run_fake_local_s2s_mvp_smoke_sync,
    run_live_agent_qwen_nullplayback_smoke,
    run_live_agent_qwen_nullplayback_smoke_sync,
)
from reachy_agent.local_s2s.stt_source import (
    FakeTranscriptSource,
    ParakeetTranscriptSource,
    SttSourceResult,
    TranscriptSource,
    transcript_event_from_audio,
)
from reachy_agent.local_s2s.agent_adapter import (
    AgentFastClient,
    AgentS2SAdapter,
    AgentStreamingClient,
)

__all__ = [
    "FakeTranscriptSource",
    "LocalS2SEvent",
    "LocalS2SEventType",
    "LocalS2SMvpSmokeReport",
    "ParakeetTranscriptSource",
    "QwenOutputBridge",
    "SttSourceResult",
    "AgentFastClient",
    "AgentS2SAdapter",
    "AgentStreamingClient",
    "TextChunkResult",
    "TranscriptSource",
    "collect_speakable_chunks",
    "make_response_cancelled",
    "make_response_completed",
    "make_transcription_completed",
    "make_output_text_delta",
    "parse_local_s2s_event",
    "run_fake_local_s2s_mvp_smoke",
    "run_fake_local_s2s_mvp_smoke_sync",
    "run_live_agent_qwen_nullplayback_smoke",
    "run_live_agent_qwen_nullplayback_smoke_sync",
    "transcript_event_from_audio",
]
