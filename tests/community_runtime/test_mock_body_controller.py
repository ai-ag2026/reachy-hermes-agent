import asyncio

from reachy_agent.runtime.adapters.mock_body import MockBodyController
from reachy_agent.runtime.core.body import BodyAction


def test_mock_body_controller_records_accepted_actions():
    async def run():
        controller = MockBodyController()
        result = await controller.execute(BodyAction(action="chirp", name="affirm", turn_id="turn-1"))
        assert result.ok is True
        assert result.action == "chirp"
        assert result.turn_id == "turn-1"
        assert result.readback == {"mock": True, "recorded_actions": 1}
        status = await controller.status()
        assert status.available is False
        assert status.current_actions == ("chirp",)

    asyncio.run(run())


def test_mock_body_controller_stop_clears_actions():
    async def run():
        controller = MockBodyController()
        await controller.execute(BodyAction(action="chirp"))
        result = await controller.execute(BodyAction(action="stop"))
        assert result.ok is True
        assert result.stopped is True
        assert controller.actions == []
        assert controller.stopped is True

    asyncio.run(run())


def test_mock_body_controller_never_requires_env_or_hardware():
    async def run():
        controller = MockBodyController()
        result = await controller.execute(BodyAction(action="look", direction="left"))
        assert result.ok is True
        assert result.readback["mock"] is True

    asyncio.run(run())
