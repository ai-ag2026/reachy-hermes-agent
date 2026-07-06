# reachy-hermes-agent

Brain-side runtime for embodying a **[Hermes](https://github.com/NousResearch/hermes) agent** — or
any OpenAI-compatible backend — in the [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/)
robot. This is the counterpart to the conversation-app fork: the app is the body (mic, speaker,
camera, motors); this package is the reusable Python core that turns a text agent into a live,
embodied voice.

> Distribution name `reachy-hermes-agent`; import as `reachy_agent`.

**Generic core, Hermes-flavored integration.** The client talks plain OpenAI-compatible
`/chat/completions`, so it works with any backend. When pointed at a Hermes gateway it additionally
uses optional `X-Hermes-Session-*` headers for working-thread continuity and long-term (Hindsight)
memory — these are injected, off by default, and harmlessly ignored by non-Hermes endpoints. The
`reachy_agent.runtime` primitives are fully framework-neutral.

It is **no-hardware-first**: every network endpoint defaults to a `127.0.0.1` placeholder and every
physical surface is opt-in, so the test suite and examples run with no robot, no camera, and no
private services.

## What's inside

- `reachy_agent.runtime` — neutral, dependency-light primitives: `BodyAction`, `BodyStatus`,
  `SafeMovementPolicy`, and mock / daemon / callable body controllers. Fake-first, safe by default.
- `reachy_agent.voice` — the voice pipeline: VAD capture, STT front-end, semantic barge-in gate,
  streaming TTS chunk queue, astromech-style audio cues, and a proper-noun correction lexicon.
- `reachy_agent.body` — one-shot vision safety wrapper and bounded head-movement planning/execution
  (antennas and body yaw are always preserved; deltas are clamped).
- `reachy_agent.local_s2s` — a local speech-to-speech turn runner and event model.
- `reachy_agent.config` — a cloud↔local preset toggle and per-module `from_env` configuration.

## Install

```bash
python -m pip install -e ".[dev]"        # core + test tooling
python -m pip install -e ".[voice]"      # add the ONNX VAD + streaming-ASR client
python -m pip install -e ".[robot]"      # add the Reachy Mini SDK / simulator
```

## No-hardware quickstart

```bash
python -m pytest -q                       # full suite, no robot / network required
```

## Configuration

Everything is driven by `AGENT_*` environment variables, all defaulting to `127.0.0.1` placeholders.
Point them at your own services:

| Variable | Purpose |
|----------|---------|
| `AGENT_BASE_URL` | Your OpenAI-compatible agent brain (`{url}/chat/completions`). |
| `AGENT_MODEL` | Model name sent to the agent (default `local-agent`). |
| `AGENT_STT_BASE_URL` | Speech-to-text endpoint. |
| `AGENT_QWEN_TTS_BASE_URL` / `AGENT_QWEN_TTS_VOICE` | Text-to-speech endpoint and voice. |
| `AGENT_VLM_BASE_URL` | Optional vision model for the one-shot "look" tool. |
| `AGENT_DAEMON_BASE_URL` | Reachy daemon HTTP API (playback / movement / status). |
| `AGENT_STT_ALIASES` | Add your robot's / household's proper nouns for STT correction, no code. |

No real endpoints, hostnames, credentials, or persona ship in this repository.

## Safety defaults

Live movement and audio playback are disabled unless a caller opts in through explicit configuration
and the requested action still passes the movement policy checks. Vision is one-shot and never
persists raw frames.

## License

Dual-licensed under `Apache-2.0 OR MIT`.
