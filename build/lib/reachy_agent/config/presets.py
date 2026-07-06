"""Cloud↔local preset toggle for the Reachy-AGENT voice stack (the operator req. 2026-06-23).

Flips the official app + our AGENT clients/front-end between:
- ``local`` — all-local: AGENT brain (Hermes :8642) + Parakeet STT (:5093) + Qwen3 TTS (:7034, default).
  The production path (and what every sim smoke proved).
- ``cloud`` — OpenAI realtime backend (brain+STT+TTS in the cloud). Dev de-risking / known-good reference.

Presets set only SELECTION env vars (backend/endpoints/model/voice). **Secrets are NEVER embedded** —
they live in ``~/.hermes/.env`` and are referenced by name via ``required_secrets()``.

API:
  apply_preset("local")          # set selection env in os.environ (call before building handler/front-end)
  preset_env("local")            # the dict of selection vars
  required_secrets("local")      # secret env var names that must be present
  check_secrets("local")         # list of MISSING required secrets (empty = ok)
CLI:
  python -m reachy_agent.config.presets local --export   # `export VAR=...` lines (eval/source)
  python -m reachy_agent.config.presets local --check    # verify required secrets present
"""

from __future__ import annotations

import os
from typing import Dict, List

PRESETS: Dict[str, dict] = {
    "local": {
        "desc": "All-local: AGENT brain (:8642) + Parakeet STT (:5093) + Qwen3 TTS (:7034, default) "
        "+ Gemma vision (:3448).",
        "env": {
            "BACKEND_PROVIDER": "agent",
            "AGENT_BASE_URL": "http://127.0.0.1:8642/v1",
            "AGENT_MODEL": "AGENT",
            "AGENT_STT_BASE_URL": "http://127.0.0.1:5093/v1",
            "AGENT_QWEN_TTS_BASE_URL": "http://127.0.0.1:7034/v1",
            "AGENT_QWEN_TTS_MODEL": "qwen3-tts",
            "AGENT_QWEN_TTS_VOICE": "default",
            # one-shot vision (M3/M5): served Gemma vision endpoint, OpenAI-compatible.
            "AGENT_VLM_BASE_URL": "http://127.0.0.1:3448/v1",
            "AGENT_VLM_MODEL": "gemma-4-E4B",
        },
        "required_secrets": ["API_SERVER_KEY"],
    },
    "cloud": {
        "desc": "Cloud reference: OpenAI realtime backend (brain+STT+TTS in cloud). Dev de-risking.",
        "env": {
            "BACKEND_PROVIDER": "openai",
        },
        "required_secrets": ["OPENAI_API_KEY"],
    },
}


def _get(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}")
    return PRESETS[name]


def preset_env(name: str) -> Dict[str, str]:
    return dict(_get(name)["env"])


def required_secrets(name: str) -> List[str]:
    return list(_get(name)["required_secrets"])


def apply_preset(name: str, environ: Dict[str, str] | None = None) -> Dict[str, str]:
    """Set the preset's selection env vars into ``environ`` (default os.environ). Returns what was set."""
    env = environ if environ is not None else os.environ
    vals = preset_env(name)
    env.update(vals)
    return vals


def check_secrets(name: str, environ: Dict[str, str] | None = None) -> List[str]:
    """Return the list of required secrets MISSING from environ (empty list = all present)."""
    env = environ if environ is not None else os.environ
    return [s for s in required_secrets(name) if not (env.get(s) or "").strip()]


def _main(argv: List[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Reachy-AGENT cloud↔local preset toggle")
    p.add_argument("preset", choices=sorted(PRESETS))
    p.add_argument("--export", action="store_true", help="print `export VAR=...` lines")
    p.add_argument("--check", action="store_true", help="verify required secrets are present")
    a = p.parse_args(argv)
    print(f"# preset {a.preset}: {_get(a.preset)['desc']}")
    if a.check:
        missing = check_secrets(a.preset)
        print(f"# required secrets: {required_secrets(a.preset)} | missing: {missing or 'none'}")
        return 1 if missing else 0
    for k, v in preset_env(a.preset).items():
        print(f"export {k}={v}" if a.export else f"{k}={v}")
    miss = check_secrets(a.preset)
    if miss:
        print(f"# NOTE: required secret(s) not set in env: {miss} (source ~/.hermes/.env)")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
