import asyncio

import pytest

from reachy_agent.runtime.adapters.hermes_agent import HermesAgentBodyController
from reachy_agent.runtime.core.body import BodyAction, BodyActionError


def test_hermes_agent_adapter_maps_chirp_to_reachy_body_payload():
    calls: list[dict[str, object]] = []

    async def fake_tool(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"ok": True, "message": "accepted", "readback": {"source": "fake"}}

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        result = await controller.execute(BodyAction(action="chirp", name="affirm", turn_id="turn-1"))
        assert result.ok is True
        assert result.action == "chirp"
        assert result.turn_id == "turn-1"
        assert result.message == "accepted"
        assert result.readback == {"source": "fake"}

    asyncio.run(run())
    assert calls == [{"action": "chirp", "name": "affirm", "turn_id": "turn-1"}]


def test_hermes_agent_adapter_preserves_look_fields_and_metadata():
    calls: list[dict[str, object]] = []

    async def fake_tool(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"ok": True}

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        await controller.execute(
            BodyAction(
                action="look",
                direction="left",
                intensity=0.5,
                duration_s=0.25,
                metadata={"reason": "unit-test"},
            )
        )

    asyncio.run(run())
    assert calls == [
        {
            "action": "look",
            "direction": "left",
            "intensity": 0.5,
            "duration_s": 0.25,
            "metadata": {"reason": "unit-test"},
        }
    ]


def test_hermes_agent_adapter_maps_center_direction_to_private_front():
    calls: list[dict[str, object]] = []

    async def fake_tool(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"ok": True}

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        await controller.execute(BodyAction(action="look", direction="center"))

    asyncio.run(run())
    assert calls == [{"action": "look", "direction": "front"}]


def test_hermes_agent_adapter_forwards_emote_emotion():
    calls: list[dict[str, object]] = []

    async def fake_tool(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"ok": True, "readback": {"emotion": "happy"}}

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        result = await controller.execute(BodyAction(action="emote", emotion="happy", turn_id="turn-2"))
        assert result.ok is True
        assert result.readback == {"emotion": "happy"}

    asyncio.run(run())
    assert calls == [{"action": "emote", "turn_id": "turn-2", "emotion": "happy"}]


def test_hermes_agent_adapter_rejects_empty_chirp_name_before_tool_call():
    called = False

    async def fake_tool(payload: dict[str, object]) -> dict[str, object]:
        nonlocal called
        called = True
        return {"ok": True}

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        with pytest.raises(BodyActionError, match="chirp action requires name"):
            await controller.execute(BodyAction(action="chirp"))

    asyncio.run(run())
    assert called is False


def test_hermes_agent_adapter_stop_uses_stop_action_and_result_flag():
    calls: list[dict[str, object]] = []

    async def fake_tool(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"ok": True, "stopped": True, "message": "stopped"}

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        result = await controller.stop("cleanup")
        assert result.ok is True
        assert result.stopped is True
        assert result.action == "stop"
        assert result.message == "stopped"

    asyncio.run(run())
    assert calls == [{"action": "stop", "reason": "cleanup"}]


def test_hermes_agent_adapter_accepts_json_string_tool_response():
    async def fake_tool(payload: dict[str, object]) -> str:
        assert payload == {"action": "chirp", "name": "affirm", "turn_id": "turn-json"}
        return '{"ok": true, "message": "json accepted", "readback": {"source": "plugin-json"}}'

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        result = await controller.execute(BodyAction(action="chirp", name="affirm", turn_id="turn-json"))
        assert result.ok is True
        assert result.action == "chirp"
        assert result.turn_id == "turn-json"
        assert result.message == "json accepted"
        assert result.readback == {"source": "plugin-json"}

    asyncio.run(run())


def test_hermes_agent_adapter_reports_json_error_response_as_not_ok():
    async def fake_tool(payload: dict[str, object]) -> str:
        return '{"error": "reachy platform not running"}'

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        result = await controller.execute(BodyAction(action="chirp", name="affirm"))
        assert result.ok is False
        assert result.action == "chirp"
        assert result.message == "reachy platform not running"
        assert result.readback == {}

    asyncio.run(run())


def test_hermes_agent_adapter_rejects_malformed_json_string_response():
    async def fake_tool(payload: dict[str, object]) -> str:
        return "not-json"

    async def run():
        controller = HermesAgentBodyController(call_reachy_body=fake_tool)
        with pytest.raises(BodyActionError, match="invalid JSON"):
            await controller.execute(BodyAction(action="chirp", name="affirm"))

    asyncio.run(run())
