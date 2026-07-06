# Pull Request

## Summary

Describe the change in one or two paragraphs.

## Scope

- [ ] Core no-hardware runtime
- [ ] Mock controller or tests
- [ ] Optional daemon adapter
- [ ] Optional agent adapter
- [ ] Documentation or examples
- [ ] Packaging or CI

## Safety and privacy

- [ ] No secrets, tokens, private hosts, local paths, raw logs, transcripts, images, or audio are included.
- [ ] No live robot, camera, microphone, speaker, movement, or daemon access is enabled by default.
- [ ] Any hardware side effect is explicitly described and gated by config or policy.
- [ ] The core package remains usable without private runtimes or optional adapters.

## Verification

Paste the commands you ran:

```bash
python -m ruff check --isolated src tests
python -m pytest -q tests/community_runtime
python -m compileall -q src tests/community_runtime
python -m pip wheel --no-deps -w dist .
```

## Notes for reviewers

Mention any follow-up live validation that is intentionally out of scope for this pull request.
