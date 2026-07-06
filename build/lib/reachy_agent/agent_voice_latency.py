from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable


class LatencyEventRecorder:
    """Append compact per-session voice latency events as JSONL."""

    def __init__(self, path: str | Path, *, session_id: str, clock: Callable[[], float] | None = None) -> None:
        """Initialize a recorder for one voice session JSONL file."""
        self.path = Path(path)
        self.session_id = session_id
        self._clock = clock or time.perf_counter
        self._started_at: float | None = None

    def record(self, event_type: str, **payload: Any) -> dict[str, Any]:
        """Append one latency event and return the serialized payload."""
        timestamp = self._clock()
        if self._started_at is None:
            self._started_at = timestamp
        event = {
            "type": event_type,
            "timestamp": timestamp,
            "relative_ms": int(round((timestamp - self._started_at) * 1000)),
            "session_id": self.session_id,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event


def _read_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event, dict):
            events.append(event)
    return events


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    if percentile == 50:
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return int(round((ordered[midpoint - 1] + ordered[midpoint]) / 2))
    rank = math.ceil((percentile / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _stats(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "max": max(values) if values else None,
    }


def summarize_latency_events(path: str | Path) -> dict[str, Any]:
    """Summarize compact latency JSONL without exposing raw transcripts."""
    events = _read_events(path)
    by_turn: dict[str, dict[str, Any]] = {}
    for event in events:
        turn_id = str(event.get("turn_id") or "default")
        by_turn.setdefault(turn_id, {})[str(event.get("type"))] = event

    speech_to_transcript: list[int] = []
    first_audio_after_transcript: list[int] = []
    response_done_after_transcript: list[int] = []
    first_token_after_request: list[int] = []
    first_tts_after_request: list[int] = []
    first_tts_after_vad: list[int] = []
    first_audio_after_tts_request: list[int] = []
    first_audio_after_vad: list[int] = []
    tool_result_after_started: list[int] = []
    ask_agent_tool_result_after_started: list[int] = []
    ask_agent_wait_ack_after_started: list[int] = []

    tool_events: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in {"tool_started", "tool_result_ready", "tool_wait_ack_queued"}:
            continue
        turn_id = str(event.get("turn_id") or "default")
        tool_id = str(event.get("call_id") or event.get("tool_id") or event.get("tool_name") or "default")
        tool_events.setdefault((turn_id, tool_id), {})[event_type] = event

    for per_tool in tool_events.values():
        started = per_tool.get("tool_started")
        result = per_tool.get("tool_result_ready")
        wait_ack = per_tool.get("tool_wait_ack_queued")
        tool_name = str((started or result or wait_ack or {}).get("tool_name") or "")
        if started and result:
            elapsed = int(round((result["timestamp"] - started["timestamp"]) * 1000))
            tool_result_after_started.append(elapsed)
            if tool_name == "ask_agent":
                ask_agent_tool_result_after_started.append(elapsed)
        if started and wait_ack and tool_name == "ask_agent":
            ask_agent_wait_ack_after_started.append(int(round((wait_ack["timestamp"] - started["timestamp"]) * 1000)))

    for turn_events in by_turn.values():
        speech_started = turn_events.get("speech_started")
        transcript_completed = turn_events.get("transcript_completed") or turn_events.get("speech_partial_ready")
        first_audio = turn_events.get("first_audio_delta") or turn_events.get("first_audio_bytes")
        response_done = turn_events.get("response_done")
        vad_end = turn_events.get("vad_end") or turn_events.get("speech_stopped")
        first_responder_request = turn_events.get("first_responder_request")
        first_token = turn_events.get("first_token")
        first_tts_request = turn_events.get("first_tts_request")
        if speech_started and transcript_completed:
            speech_to_transcript.append(
                int(round((transcript_completed["timestamp"] - speech_started["timestamp"]) * 1000))
            )
        if transcript_completed and first_audio:
            first_audio_after_transcript.append(
                int(round((first_audio["timestamp"] - transcript_completed["timestamp"]) * 1000))
            )
        if transcript_completed and response_done:
            response_done_after_transcript.append(
                int(round((response_done["timestamp"] - transcript_completed["timestamp"]) * 1000))
            )
        if first_responder_request and first_token:
            first_token_after_request.append(
                int(round((first_token["timestamp"] - first_responder_request["timestamp"]) * 1000))
            )
        if first_responder_request and first_tts_request:
            first_tts_after_request.append(
                int(round((first_tts_request["timestamp"] - first_responder_request["timestamp"]) * 1000))
            )
        if vad_end and first_tts_request:
            first_tts_after_vad.append(int(round((first_tts_request["timestamp"] - vad_end["timestamp"]) * 1000)))
        if first_tts_request and first_audio:
            first_audio_after_tts_request.append(
                int(round((first_audio["timestamp"] - first_tts_request["timestamp"]) * 1000))
            )
        if vad_end and first_audio:
            first_audio_after_vad.append(int(round((first_audio["timestamp"] - vad_end["timestamp"]) * 1000)))

    return {
        "event_count": len(events),
        "turn_count": len(by_turn),
        "speech_to_transcript_ms": _stats(speech_to_transcript),
        "first_audio_after_transcript_ms": _stats(first_audio_after_transcript),
        "response_done_after_transcript_ms": _stats(response_done_after_transcript),
        "first_token_after_request_ms": _stats(first_token_after_request),
        "first_tts_after_request_ms": _stats(first_tts_after_request),
        "first_tts_after_vad_ms": _stats(first_tts_after_vad),
        "first_audio_after_tts_request_ms": _stats(first_audio_after_tts_request),
        "first_audio_after_vad_ms": _stats(first_audio_after_vad),
        "tool_result_after_started_ms": _stats(tool_result_after_started),
        "ask_agent_tool_result_after_started_ms": _stats(ask_agent_tool_result_after_started),
        "ask_agent_wait_ack_after_started_ms": _stats(ask_agent_wait_ack_after_started),
    }
