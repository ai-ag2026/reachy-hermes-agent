"""Unit tests for the cloud↔local preset toggle (pure; no app/network)."""

import pytest

from reachy_agent.config.presets import (
    PRESETS,
    apply_preset,
    check_secrets,
    preset_env,
    required_secrets,
)


def test_both_presets_exist():
    assert set(PRESETS) == {"local", "cloud"}


def test_local_preset_is_all_local_agent():
    env = preset_env("local")
    assert env["BACKEND_PROVIDER"] == "agent"
    assert "8642" in env["AGENT_BASE_URL"]
    assert env["AGENT_STT_BASE_URL"].endswith(":5093/v1")
    assert env["AGENT_QWEN_TTS_BASE_URL"].endswith(":7034/v1")
    assert env["AGENT_QWEN_TTS_VOICE"] == "default"
    assert env["AGENT_VLM_BASE_URL"].endswith(":3448/v1")
    assert env["AGENT_VLM_MODEL"] == "gemma-4-E4B"
    assert required_secrets("local") == ["API_SERVER_KEY"]


def test_cloud_preset_selects_openai():
    env = preset_env("cloud")
    assert env["BACKEND_PROVIDER"] == "openai"
    assert required_secrets("cloud") == ["OPENAI_API_KEY"]


def test_apply_preset_sets_selection_vars_in_target_env():
    fake = {}
    apply_preset("local", fake)
    assert fake["BACKEND_PROVIDER"] == "agent"
    assert fake["AGENT_QWEN_TTS_VOICE"] == "default"
    apply_preset("cloud", fake)
    assert fake["BACKEND_PROVIDER"] == "openai"  # toggled


def test_presets_never_embed_secrets():
    for name in PRESETS:
        for secret in required_secrets(name):
            assert secret not in preset_env(name), f"{name} must not embed {secret}"


def test_check_secrets_reports_missing():
    assert check_secrets("local", {}) == ["API_SERVER_KEY"]
    assert check_secrets("local", {"API_SERVER_KEY": "x"}) == []


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        preset_env("nope")
