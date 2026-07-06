from __future__ import annotations

import json
from pathlib import Path

from reachy_agent.local_s2s.runtime_smoke import (
    FakeQwenTtsClient,
    FakeAgentStreamClient,
    run_fake_local_s2s_mvp_smoke_sync,
)


def test_fake_local_s2s_mvp_smoke_writes_metadata_reports(tmp_path: Path) -> None:
    """Run the full fake MVP chain and write metadata-safe artifacts."""
    report = run_fake_local_s2s_mvp_smoke_sync(transcript="Was bist du?", report_root=tmp_path)

    report_dir = Path(report.report_dir)
    assert report.ok is True
    assert report.mode == "fake"
    assert report.transcript_chars == len("Was bist du?")
    assert report.agent_delta_count == 1
    assert report.qwen_call_count == 1
    assert report.playback_chunk_count == 1
    assert report.total_audio_bytes > 0
    assert report.playback_side_effects == []
    assert report.timings_ms["runtime_total_ms"] >= 0
    assert report.timings_ms["end_to_end_total_ms"] == report.timings_ms["runtime_total_ms"]
    assert report.timings_ms["total_ms"] == report.timings_ms["end_to_end_total_ms"]
    assert report.latency_budget is None
    assert report.agent_response_valid is True
    assert report_dir.exists()
    assert (report_dir / "report.json").exists()
    assert (report_dir / "EVENTS.jsonl").exists()
    assert (report_dir / "latency.jsonl").exists()
    assert (report_dir / "REPORT.md").exists()

    report_payload = json.loads((report_dir / "report.json").read_text())
    assert report_payload["ok"] is True
    assert report_payload["playback_side_effects"] == []
    assert "timings_ms" in report_payload
    assert "latency_budget" in report_payload


def test_fake_smoke_latency_budget_can_pass_or_fail(tmp_path: Path) -> None:
    """Expose latency budget verdict separately from functional success."""
    pass_report = run_fake_local_s2s_mvp_smoke_sync(transcript="Kurz", report_root=tmp_path, latency_budget_ms=60_000)
    assert pass_report.ok is True
    assert pass_report.latency_budget is not None
    assert pass_report.latency_budget["status"] == "pass"
    assert pass_report.latency_budget["violations"] == []

    fail_report = run_fake_local_s2s_mvp_smoke_sync(transcript="Kurz", report_root=tmp_path, latency_budget_ms=-1)
    assert fail_report.ok is True
    assert fail_report.latency_budget is not None
    assert fail_report.latency_budget["status"] in {"warn", "fail"}


def test_fake_smoke_events_are_sanitized(tmp_path: Path) -> None:
    """Write event metadata without raw transcript or response text."""
    report = run_fake_local_s2s_mvp_smoke_sync(transcript="Geheimer Test", report_root=tmp_path)

    events_text = Path(report.events_path).read_text()
    assert "Geheimer Test" not in events_text
    assert "Ich bin AGENT" not in events_text

    events = [json.loads(line) for line in events_text.splitlines() if line.strip()]
    assert events[0]["type"] == "conversation.item.input_audio_transcription.completed"
    assert events[0]["transcript_chars"] == len("Geheimer Test")
    assert any(event["type"] == "response.output_text.delta" for event in events)
    assert any(event["type"] == "response.completed" for event in events)


def test_fake_smoke_cli_report_json_shape(tmp_path: Path) -> None:
    """Keep report JSON small and machine-readable for later pack chaining."""
    report = run_fake_local_s2s_mvp_smoke_sync(transcript="Was bist du?", report_root=tmp_path)

    payload = json.loads((Path(report.report_dir) / "report.json").read_text())
    assert sorted(payload) == [
        "agent_delta_count",
        "agent_response_valid",
        "events_path",
        "latency_budget",
        "latency_path",
        "mode",
        "ok",
        "playback_chunk_count",
        "playback_side_effects",
        "qwen_call_count",
        "report_dir",
        "timings_ms",
        "total_audio_bytes",
        "transcript_chars",
    ]


def test_live_agent_qwen_nullplayback_mode_rejects_agent_error_fallback(tmp_path: Path) -> None:
    """Do not mark live smokes green when AGENT only returned a bridge fallback."""
    agent = FakeAgentStreamClient(["Da hakt gerade die Verbindung zu AGENT."])
    qwen = FakeQwenTtsClient()

    import asyncio

    from reachy_agent.local_s2s.runtime_smoke import run_live_agent_qwen_nullplayback_smoke

    report = asyncio.run(
        run_live_agent_qwen_nullplayback_smoke(
            transcript="Was bist du?",
            report_root=tmp_path,
            agent_client=agent,
            tts_client=qwen,
        )
    )

    assert report.ok is False
    assert report.agent_response_valid is False
    assert report.playback_chunk_count == 1
    assert report.playback_side_effects == []


def test_live_agent_qwen_nullplayback_mode_accepts_injected_clients(tmp_path: Path) -> None:
    """Exercise the live-mode report path without network by injecting fake clients."""
    agent = FakeAgentStreamClient(["Live Modus. ", "NullPlayback bleibt sicher."])
    qwen = FakeQwenTtsClient()

    import asyncio

    from reachy_agent.local_s2s.runtime_smoke import run_live_agent_qwen_nullplayback_smoke

    report = asyncio.run(
        run_live_agent_qwen_nullplayback_smoke(
            transcript="Was bist du?",
            report_root=tmp_path,
            agent_client=agent,
            tts_client=qwen,
            pre_timings_ms={"stt_total_ms": 123},
        )
    )

    assert report.ok is True
    assert report.mode == "live-agent-qwen-nullplayback"
    assert agent.calls == ["Was bist du?"]
    assert qwen.calls == ["Live Modus.", "NullPlayback bleibt sicher."]
    assert report.timings_ms["stt_total_ms"] == 123
    assert report.timings_ms["end_to_end_total_ms"] == report.timings_ms["runtime_total_ms"] + 123
    assert report.playback_side_effects == []
    assert Path(report.report_dir, "REPORT.md").read_text().count("live AGENT request: `False`") == 1
