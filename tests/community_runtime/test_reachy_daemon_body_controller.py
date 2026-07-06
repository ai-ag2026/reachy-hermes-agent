import asyncio
import json

import httpx
import pytest

from reachy_agent.runtime.adapters.reachy_daemon import ReachyDaemonBodyController
from reachy_agent.runtime.core.body import BodyAction, BodyActionError
from reachy_agent.runtime.core.safety import SafetyConfig


def _json_response(data: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data)


def test_reachy_daemon_status_reads_small_safe_endpoints_without_state_full():
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/daemon/status":
            return _json_response({"status": "running", "error": None})
        if request.url.path == "/api/motors/status":
            return _json_response({"mode": "disabled"})
        if request.url.path == "/api/move/running":
            return _json_response([])
        if request.url.path == "/api/media/status":
            return _json_response({"available": True})
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            controller = ReachyDaemonBodyController(
                base_url="http://example.invalid",
                client=client,
                safety=SafetyConfig(body_enabled=True, live_movement_enabled=False),
            )
            status = await controller.status()
            assert status.available is True
            assert status.motors_enabled is False
            assert status.current_actions == ()
            assert status.media_available is True
            assert status.error is None

    asyncio.run(run())
    assert ("GET", "/api/state/full") not in calls
    assert calls == [
        ("GET", "/api/daemon/status"),
        ("GET", "/api/motors/status"),
        ("GET", "/api/move/running"),
        ("GET", "/api/media/status"),
    ]


def test_reachy_daemon_chirp_uses_configured_sound_map_and_returns_sanitized_readback():
    calls: list[tuple[str, str, dict]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}") if request.content else {}
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/api/daemon/status":
            return _json_response({"status": "running", "error": None})
        if request.url.path == "/api/motors/status":
            return _json_response({"mode": "disabled"})
        if request.url.path == "/api/move/running":
            return _json_response([])
        if request.url.path == "/api/media/status":
            return _json_response({"available": True})
        if request.url.path == "/api/media/play_sound":
            return _json_response({"status": "ok"})
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            controller = ReachyDaemonBodyController(
                base_url="http://example.invalid",
                client=client,
                safety=SafetyConfig(body_enabled=True),
                sound_map={"affirm": "affirm.wav"},
            )
            result = await controller.execute(BodyAction(action="chirp", name="affirm", turn_id="t-1"))
            assert result.ok is True
            assert result.action == "chirp"
            assert result.turn_id == "t-1"
            assert result.readback == {"daemon_status": "ok", "file": "affirm.wav"}

    asyncio.run(run())
    assert calls[-1] == ("POST", "/api/media/play_sound", {"file": "affirm.wav"})


def test_reachy_daemon_live_look_is_blocked_without_live_movement_before_post():
    post_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            post_paths.append(request.url.path)
        if request.url.path == "/api/daemon/status":
            return _json_response({"status": "running", "error": None})
        if request.url.path == "/api/motors/status":
            return _json_response({"mode": "enabled"})
        if request.url.path == "/api/move/running":
            return _json_response([])
        if request.url.path == "/api/media/status":
            return _json_response({"available": True})
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            controller = ReachyDaemonBodyController(
                base_url="http://example.invalid",
                client=client,
                safety=SafetyConfig(body_enabled=True, live_movement_enabled=False),
            )
            with pytest.raises(BodyActionError, match="live movement is disabled"):
                await controller.execute(BodyAction(action="look", direction="left", duration_s=0.5))

    asyncio.run(run())
    assert post_paths == []


def test_reachy_daemon_stop_posts_stop_and_clears_running_state():
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/move/stop":
            return _json_response({"status": "ok"})
        if request.url.path == "/api/move/running":
            return _json_response([])
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            controller = ReachyDaemonBodyController(base_url="http://example.invalid", client=client)
            result = await controller.stop("test cleanup")
            assert result.ok is True
            assert result.stopped is True
            assert result.action == "stop"
            assert result.readback == {"running_actions": 0}

    asyncio.run(run())
    assert calls == [("POST", "/api/move/stop"), ("GET", "/api/move/running")]


def test_reachy_daemon_head_tracking_is_explicitly_parked_without_posting():
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/daemon/status":
            return _json_response({"status": "running", "error": None})
        if request.url.path == "/api/motors/status":
            return _json_response({"mode": "disabled"})
        if request.url.path == "/api/move/running":
            return _json_response([])
        if request.url.path == "/api/media/status":
            return _json_response({"available": True})
        raise AssertionError(f"unexpected endpoint: {request.method} {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            controller = ReachyDaemonBodyController(
                base_url="http://example.invalid",
                client=client,
                safety=SafetyConfig(body_enabled=True),
            )
            with pytest.raises(BodyActionError, match="daemon head_tracking is explicitly parked"):
                await controller.execute(BodyAction(action="head_tracking", enabled=True))

    asyncio.run(run())
    assert all(method == "GET" for method, _path in calls)
    assert ("GET", "/api/state/full") not in calls
