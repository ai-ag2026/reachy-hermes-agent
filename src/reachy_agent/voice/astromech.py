"""Procedural astromech (R2D2-style) status sounds — pure numpy, no assets, no deps.

A small palette of non-verbal droid cues so AGENT can signal its STATE without words: it heard you
and is thinking, it's working on a tool, the answer is ready, something failed, yes/no, etc. Each
sound is synthesized (warbling pitch-swept bleeps + a square harmonic for the droid timbre) and
returned as int16 mono PCM ready for the app's output queue — nothing has to be bundled and it
plays through the existing PCM path. Character reference: ElGrorg/R3-MNE-Reachy-Astromech.

Play via ``chirp(name, sample_rate)``; names are in ``CHIRPS``.
"""

from __future__ import annotations

import numpy as np

_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _bleep(
    sr: int, f0: float, f1: float, dur: float, *, vib_hz: float = 18.0, vib_depth: float = 0.05, curve: float = 1.0
) -> np.ndarray:
    """One warbling bleep: pitch glide f0->f1 (curve>1 eases late) with vibrato + a square harmonic."""
    n = max(1, int(sr * dur))
    t = np.linspace(0.0, dur, n, endpoint=False)
    ramp = (np.linspace(0.0, 1.0, n)) ** curve
    freq = (f0 + (f1 - f0) * ramp) * (1.0 + vib_depth * np.sin(2.0 * np.pi * vib_hz * t))
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    wave = 0.8 * np.sin(phase) + 0.2 * np.sign(np.sin(phase))
    attack = max(1, int(sr * 0.008))
    env = np.minimum(1.0, np.arange(n) / attack) * np.exp(-3.2 * t / max(dur, 1e-6))
    return wave * env


def _gap(sr: int, s: float) -> np.ndarray:
    return np.zeros(max(0, int(sr * s)), dtype=np.float64)


def _seq(sr: int, name: str) -> np.ndarray:
    b = _bleep
    g = _gap(sr, 0.018)
    if name == "acknowledge":  # "bee-boo-beep" — heard you, thinking
        parts = [b(sr, 520, 880, 0.12), g, b(sr, 960, 620, 0.10), g, b(sr, 700, 1040, 0.09)]
    elif name == "working":  # busy mid warble, strong vibrato
        parts = [b(sr, 640, 700, 0.30, vib_hz=11, vib_depth=0.12)]
    elif name == "ready":  # bright confident ascent — here's the answer
        parts = [b(sr, 600, 900, 0.08, curve=1.4), b(sr, 900, 1320, 0.10, curve=1.6)]
    elif name == "done":  # short satisfied two-tone down
        parts = [b(sr, 1000, 1000, 0.07), g, b(sr, 760, 620, 0.09)]
    elif name == "error":  # detuned descending — uh-oh
        parts = [b(sr, 820, 300, 0.28, vib_hz=9, vib_depth=0.10, curve=0.7)]
    elif name == "affirm":  # two quick rising blips — yes
        parts = [b(sr, 700, 980, 0.06), g, b(sr, 900, 1200, 0.06)]
    elif name == "negative":  # low descending "bwomp" — no
        parts = [b(sr, 520, 300, 0.16, curve=0.8)]
    elif name == "curious":  # wavering up-down-up — hm?
        parts = [b(sr, 700, 950, 0.09), b(sr, 950, 640, 0.08), b(sr, 640, 900, 0.09)]
    elif name == "notify":  # bright two-tone ding — proactive delivery
        parts = [b(sr, 1046, 1046, 0.09), g, b(sr, 1318, 1318, 0.12)]
    elif name == "wake":  # boot sweep low->high, accelerating warble
        parts = [b(sr, 300, 1100, 0.45, vib_hz=6, vib_depth=0.06, curve=1.8)]
    elif name == "sleep":  # power-down high->low, slowing
        parts = [b(sr, 1000, 240, 0.55, vib_hz=7, vib_depth=0.08, curve=0.6)]
    else:
        parts = [b(sr, 700, 900, 0.10)]
    return np.concatenate(parts)


CHIRPS = (
    "acknowledge",
    "working",
    "ready",
    "done",
    "error",
    "affirm",
    "negative",
    "curious",
    "notify",
    "wake",
    "sleep",
)


def chirp(name: str, sample_rate: int = 24000, gain: float = 0.5) -> np.ndarray:
    """Return the named status sound as int16 mono PCM at ``sample_rate``.

    The cache holds the peak-normalized float signal; the gain is applied on every call —
    baking the first caller's gain into the cache made runtime AGENT_CHIRP_GAIN changes apply
    only to never-played chirps (review 2026-07-02 round 2, P2)."""
    key = (name, sample_rate)
    if key not in _CACHE:
        sig = _seq(sample_rate, name)
        peak = float(np.max(np.abs(sig))) or 1.0
        _CACHE[key] = sig / peak
    return (_CACHE[key] * float(np.clip(gain, 0.0, 1.0)) * 32767.0).astype(np.int16)
