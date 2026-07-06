# Security Policy

`reachy-agent-runtime` is intended to be conservative by default: no live robot access, no camera, no microphone, no speaker, and no movement unless a caller opts in through explicit configuration and policy checks.

## Supported versions

This public-candidate template is not a release yet.

| Version | Supported |
|---|---|
| 0.1.x public candidate | No public support channel yet |

Before publication, replace this section with the chosen project support window.

## Reporting a vulnerability

Before publication, replace this placeholder with the chosen private security advisory or contact path for the public repository.

Do not include secrets, tokens, private URLs, raw logs, transcripts, images, audio recordings, or live robot credentials in public reports.

Useful reports include:

- affected version or commit;
- whether a live robot was involved;
- whether hardware side effects were possible;
- minimal reproduction steps using fake/no-hardware tests where possible;
- expected safe behavior;
- observed behavior.

## Hardware and privacy sensitive issues

Treat these as security-sensitive until triaged:

- movement enabled without explicit opt-in;
- camera, microphone, speaker, or robot daemon access enabled by default;
- hardcoded hosts, IP addresses, credentials, or `.env` values;
- logs or artifacts that contain transcripts, images, audio, or private runtime metadata;
- adapter behavior that makes a private runtime mandatory for the core package.
