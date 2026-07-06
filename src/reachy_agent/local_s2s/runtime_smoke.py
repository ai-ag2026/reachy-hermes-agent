"""Local S2S MVP smokes for the Reachy/AGENT path."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from reachy_agent.local_s2s.events import make_transcription_completed
from reachy_agent.local_s2s.qwen_output import QwenOutputBridge
from reachy_agent.local_s2s.agent_adapter import AgentFastClient, AgentS2SAdapter, AgentStreamingClient
from reachy_agent.playback_queue import NullPlaybackAdapter
from reachy_agent.agent_bridge import HermesAgentClient, AgentBridgeConfig
from reachy_agent.agent_voice_latency import LatencyEventRecorder
from reachy_agent.tts_chunk_queue import QwenTtsClient, SpeechSynthesizer

DEFAULT_FAKE_AGENT_RESPONSE = "Ich bin ein lokaler Agent im Reachy-Körper."

DEFAULT_LATENCY_BUDGETS_MS: dict[str, dict[str, int]] = {
    "live-agent-qwen-nullplayback": {"warn_total_ms": 5000, "fail_total_ms": 8000},
    "fixture-stt-live-agent-qwen-nullplayback": {"warn_total_ms": 5500, "fail_total_ms": 9000},
    "live-stt-live-agent-qwen-nullplayback": {"warn_total_ms": 5500, "fail_total_ms": 9000},
}


@dataclass(frozen=True)
class LocalS2SMvpSmokeReport:
    """Metadata-only report for a local S2S MVP smoke."""

    ok: bool
    mode: str
    transcript_chars: int
    agent_delta_count: int
    qwen_call_count: int
    playback_chunk_count: int
    total_audio_bytes: int
    playback_side_effects: list[str]
    timings_ms: dict[str, int]
    latency_budget: dict[str, object] | None
    agent_response_valid: bool
    report_dir: str
    events_path: str
    latency_path: str


class FakeAgentStreamClient:
    """Fake AGENT stream client for deterministic no-network smokes."""

    def __init__(self, chunks: list[str]) -> None:
        """Initialize fake response chunks."""
        self.chunks = chunks
        self.calls: list[str] = []

    async def stream_fast(self, transcript: str) -> AsyncIterator[str]:
        """Yield configured chunks and record the transcript."""
        self.calls.append(transcript)
        for chunk in self.chunks:
            yield chunk


class FakeQwenTtsClient:
    """Fake Qwen TTS client for deterministic no-audio smokes."""

    def __init__(self) -> None:
        """Initialize fake call log."""
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        """Return deterministic fake WAV-like bytes."""
        self.calls.append(text)
        return f"FAKE_WAV:{text}".encode()


async def run_fake_local_s2s_mvp_smoke(
    *,
    transcript: str,
    report_root: str | Path = "reports",
    agent_response: str = DEFAULT_FAKE_AGENT_RESPONSE,
    latency_budget_ms: int | None = None,
) -> LocalS2SMvpSmokeReport:
    """Run the full synthetic local S2S MVP chain without live side effects."""
    agent_client = FakeAgentStreamClient([agent_response])
    qwen_client = FakeQwenTtsClient()
    return await _run_local_s2s_mvp_smoke(
        transcript=transcript,
        report_root=report_root,
        mode="fake",
        agent_client=agent_client,
        tts_client=qwen_client,
        qwen_call_counter=lambda: len(qwen_client.calls),
        live_agent_request=False,
        live_qwen_request=False,
        latency_budget_ms=latency_budget_ms,
    )


async def run_live_agent_qwen_nullplayback_smoke(
    *,
    transcript: str,
    report_root: str | Path = "reports",
    agent_client: AgentFastClient | AgentStreamingClient | None = None,
    tts_client: SpeechSynthesizer | None = None,
    mode: str = "live-agent-qwen-nullplayback",
    pre_timings_ms: dict[str, int] | None = None,
    latency_budget_ms: int | None = None,
) -> LocalS2SMvpSmokeReport:
    """Run live Hermes/AGENT plus live Qwen through NullPlayback only."""
    live_agent = agent_client or HermesAgentClient(AgentBridgeConfig.from_env().for_new_voice_session())
    live_tts = tts_client or QwenTtsClient()
    return await _run_local_s2s_mvp_smoke(
        transcript=transcript,
        report_root=report_root,
        mode=mode,
        agent_client=live_agent,
        tts_client=live_tts,
        qwen_call_counter=None,
        live_agent_request=agent_client is None,
        live_qwen_request=tts_client is None,
        pre_timings_ms=pre_timings_ms,
        latency_budget_ms=latency_budget_ms,
    )


async def _run_local_s2s_mvp_smoke(
    *,
    transcript: str,
    report_root: str | Path,
    mode: str,
    agent_client: AgentFastClient | AgentStreamingClient,
    tts_client: SpeechSynthesizer,
    qwen_call_counter: Any,
    live_agent_request: bool,
    live_qwen_request: bool,
    pre_timings_ms: dict[str, int] | None = None,
    latency_budget_ms: int | None = None,
) -> LocalS2SMvpSmokeReport:
    started = datetime.now().astimezone()
    report_dir = Path(report_root) / f"reachy-local-s2s-mvp-{mode}-{started.strftime('%Y%m%dT%H%M%S%z')}"
    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "EVENTS.jsonl"
    latency_path = report_dir / "latency.jsonl"

    timings: dict[str, int] = dict(pre_timings_ms or {})
    wall_start = time.perf_counter()
    input_event = make_transcription_completed(transcript=transcript).to_dict()
    agent_adapter = AgentS2SAdapter(agent_client)

    agent_start = time.perf_counter()
    response_events = await agent_adapter.handle_event(input_event)
    timings["agent_response_ms"] = _elapsed_ms(agent_start)
    timings["first_delta_equivalent_ms"] = timings["agent_response_ms"]

    playback = NullPlaybackAdapter()
    recorder = LatencyEventRecorder(latency_path, session_id=mode)
    output_bridge = QwenOutputBridge(tts_client, playback_adapter=playback, latency_recorder=recorder, turn_id=mode)

    qwen_start = time.perf_counter()
    playback_result = await output_bridge.handle_events(response_events)
    timings["qwen_nullplayback_ms"] = _elapsed_ms(qwen_start)
    timings["first_audio_equivalent_ms"] = timings["agent_response_ms"] + timings["qwen_nullplayback_ms"]
    timings["runtime_total_ms"] = _elapsed_ms(wall_start)
    timings["total_from_audio_available_ms"] = _total_from_audio_available_ms(timings)
    timings["total_including_generated_input_ms"] = timings["total_from_audio_available_ms"] + timings.get(
        "input_audio_generation_ms", 0
    )
    timings["end_to_end_total_ms"] = timings["total_from_audio_available_ms"]
    timings["total_ms"] = timings["end_to_end_total_ms"]

    _write_sanitized_events(events_path, [input_event, *response_events])
    qwen_call_count = qwen_call_counter() if qwen_call_counter is not None else playback_result.chunk_count
    agent_response_valid = _agent_response_valid(response_events)
    report = LocalS2SMvpSmokeReport(
        ok=bool(agent_response_valid and playback_result.chunk_count >= 1 and playback_result.total_audio_bytes > 0),
        mode=mode,
        transcript_chars=len(transcript.strip()),
        agent_delta_count=sum(1 for event in response_events if event.get("type") == "response.output_text.delta"),
        qwen_call_count=qwen_call_count,
        playback_chunk_count=playback_result.chunk_count,
        total_audio_bytes=playback_result.total_audio_bytes,
        playback_side_effects=playback_result.playback_side_effects,
        timings_ms=timings,
        latency_budget=_latency_budget_result(mode, timings, latency_budget_ms),
        agent_response_valid=agent_response_valid,
        report_dir=str(report_dir),
        events_path=str(events_path),
        latency_path=str(latency_path),
    )
    (report_dir / "report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    (report_dir / "REPORT.md").write_text(
        _markdown_report(report, live_agent_request=live_agent_request, live_qwen_request=live_qwen_request)
    )
    return report


def run_fake_local_s2s_mvp_smoke_sync(
    *,
    transcript: str,
    report_root: str | Path = "reports",
    agent_response: str = DEFAULT_FAKE_AGENT_RESPONSE,
    latency_budget_ms: int | None = None,
) -> LocalS2SMvpSmokeReport:
    """Run the fake local S2S MVP smoke from synchronous callers."""
    return asyncio.run(
        run_fake_local_s2s_mvp_smoke(
            transcript=transcript,
            report_root=report_root,
            agent_response=agent_response,
            latency_budget_ms=latency_budget_ms,
        )
    )


def run_live_agent_qwen_nullplayback_smoke_sync(
    *,
    transcript: str,
    report_root: str | Path = "reports",
    mode: str = "live-agent-qwen-nullplayback",
    pre_timings_ms: dict[str, int] | None = None,
    latency_budget_ms: int | None = None,
) -> LocalS2SMvpSmokeReport:
    """Run the live AGENT/Qwen NullPlayback smoke from synchronous callers."""
    return asyncio.run(
        run_live_agent_qwen_nullplayback_smoke(
            transcript=transcript,
            report_root=report_root,
            mode=mode,
            pre_timings_ms=pre_timings_ms,
            latency_budget_ms=latency_budget_ms,
        )
    )


def _write_sanitized_events(path: Path, events: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sequence, event in enumerate(events):
            event_type = str(event.get("type") or "")
            sanitized: dict[str, object] = {"sequence": sequence, "type": event_type}
            if "transcript" in event:
                sanitized["transcript_chars"] = len(str(event.get("transcript") or ""))
            if "delta" in event:
                sanitized["delta_chars"] = len(str(event.get("delta") or ""))
            if "reason" in event:
                sanitized["reason"] = str(event.get("reason") or "")
            fh.write(json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n")


def _markdown_report(report: LocalS2SMvpSmokeReport, *, live_agent_request: bool, live_qwen_request: bool) -> str:
    return f"""# Reachy Local S2S MVP Smoke

Mode: `{report.mode}`
OK: `{report.ok}`

## Metadata

- transcript chars: `{report.transcript_chars}`
- AGENT delta events: `{report.agent_delta_count}`
- Qwen calls/equivalent chunks: `{report.qwen_call_count}`
- NullPlayback chunks: `{report.playback_chunk_count}`
- total audio bytes: `{report.total_audio_bytes}`
- playback side effects: `{report.playback_side_effects}`
- timings ms: `{report.timings_ms}`
- latency budget: `{report.latency_budget}`
- AGENT response valid: `{report.agent_response_valid}`

## Live dependency use

- live AGENT request: `{live_agent_request}`
- live Qwen request: `{live_qwen_request}`

## Boundaries

```text
No microphone.
No speaker.
No camera.
No robot movement.
No Reachy app/daemon mutation.
No container mutation.
NullPlayback only.
```

## Artifacts

- `report.json`
- `EVENTS.jsonl`
- `latency.jsonl`
- `REPORT.md`
"""


def _latency_budget_result(
    mode: str, timings: dict[str, int], override_fail_total_ms: int | None
) -> dict[str, object] | None:
    defaults = DEFAULT_LATENCY_BUDGETS_MS.get(mode)
    if defaults is None and override_fail_total_ms is None:
        return None
    fail_total_ms = override_fail_total_ms if override_fail_total_ms is not None else int(defaults["fail_total_ms"])  # type: ignore[index]
    warn_total_ms = int(defaults.get("warn_total_ms", fail_total_ms)) if defaults else fail_total_ms
    observed_total_ms = int(timings.get("end_to_end_total_ms", timings.get("total_ms", 0)))
    violations: list[str] = []
    if observed_total_ms > fail_total_ms:
        status = "fail"
        violations.append("end_to_end_total_ms_exceeds_fail_threshold")
    elif observed_total_ms > warn_total_ms:
        status = "warn"
        violations.append("end_to_end_total_ms_exceeds_warn_threshold")
    else:
        status = "pass"
    return {
        "status": status,
        "observed_ms": {"end_to_end_total_ms": observed_total_ms},
        "thresholds_ms": {"warn_total_ms": warn_total_ms, "fail_total_ms": fail_total_ms},
        "violations": violations,
    }


def _total_from_audio_available_ms(timings: dict[str, int]) -> int:
    if "stt_total_ms" in timings:
        return timings["stt_total_ms"] + timings["runtime_total_ms"]
    return timings["runtime_total_ms"]


def _agent_response_valid(events: list[dict[str, object]]) -> bool:
    deltas = [
        str(event.get("delta") or "").strip() for event in events if event.get("type") == "response.output_text.delta"
    ]
    non_empty = [delta for delta in deltas if delta]
    if not non_empty:
        return False
    fallback_texts = {
        "Da hakt gerade die Verbindung zu AGENT.",
        "Ich prüfe das weiter im Hintergrund.",
        "Das habe ich akustisch nicht erwischt.",
    }
    return not all(delta in fallback_texts for delta in non_empty)


def _elapsed_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))
