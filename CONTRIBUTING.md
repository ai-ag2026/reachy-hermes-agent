# Contributing to reachy-hermes-agent

Thanks for helping with the Reachy Mini agent runtime.

This package is no-hardware-first. Contributions should keep the core usable without a specific agent backend, private infrastructure, a live robot, camera, microphone, speaker, or movement.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m ruff check --isolated src tests
python -m pytest -q
python -m compileall -q src
python -m pip wheel --no-deps -w dist .
```

## Contribution rules

- Keep changes small and reviewable.
- Add or update fake/no-hardware tests for behavior changes.
- Keep hardware access off by default.
- Do not add hardcoded robot hosts, local network addresses, credentials, tokens, `.env` values, logs, transcripts, images, or audio captures.
- Keep specific-agent integrations optional and injected through adapter boundaries. The core package must not require a private runtime.
- Document any robot-side effect explicitly: movement, sound, camera, microphone, daemon calls, or persistent state changes.

## Safety expectations

A pull request that can affect hardware must state:

1. the exact side effect;
2. whether it is enabled by default;
3. which policy or config gate controls it;
4. which fake/no-hardware tests prove the default-safe path;
5. what manual live validation would be required later.

Live robot validation is not part of the default CI path.

## Pull request checklist

Before opening a pull request, run:

```bash
python -m ruff check --isolated src tests
python -m pytest -q
python -m compileall -q src
python -m pip wheel --no-deps -w dist .
```

Then confirm:

- [ ] No private infrastructure or secrets are included.
- [ ] No hardware behavior is enabled by default.
- [ ] Tests cover the changed public contract.
- [ ] Documentation explains any new safety boundary.
