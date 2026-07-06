from __future__ import annotations

from reachy_agent.playback_queue import (
    DEFAULT_REACHY_DAEMON_BASE_URL,
    REACHY_DAEMON_BASE_URL_ENV,
    ReachyHttpMediaClient,
)


def test_reachy_http_media_client_defaults_to_mdns_hostname(monkeypatch):
    monkeypatch.delenv(REACHY_DAEMON_BASE_URL_ENV, raising=False)

    client = ReachyHttpMediaClient()

    assert client._base_url == DEFAULT_REACHY_DAEMON_BASE_URL
    assert client._base_url == "http://127.0.0.1:8000"


def test_reachy_http_media_client_accepts_env_override(monkeypatch):
    monkeypatch.setenv(REACHY_DAEMON_BASE_URL_ENV, "http://custom-reachy.local:8000/")

    client = ReachyHttpMediaClient()

    assert client._base_url == "http://custom-reachy.local:8000"


def test_reachy_http_media_client_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv(REACHY_DAEMON_BASE_URL_ENV, "http://env-reachy.local:8000")

    client = ReachyHttpMediaClient(base_url="http://explicit-reachy.local:8000/")

    assert client._base_url == "http://explicit-reachy.local:8000"
