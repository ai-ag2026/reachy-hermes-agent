"""Procedural astromech status chirps: every palette entry is valid, bounded, non-silent PCM."""

import numpy as np

from reachy_agent.voice.astromech import CHIRPS, chirp


def test_all_chirps_valid_pcm():
    for name in CHIRPS:
        c = chirp(name, 24000)
        assert c.dtype == np.int16
        assert 0.05 < len(c) / 24000 < 0.8  # short cue
        assert int(np.abs(c).max()) > 3000  # audibly non-silent
        assert int(np.abs(c).max()) <= 32767


def test_gain_scales_and_caches():
    a = chirp("acknowledge", 16000)
    b = chirp("acknowledge", 16000)
    # The cache holds the normalized float signal; gain is applied per call so runtime
    # AGENT_CHIRP_GAIN changes take effect (review 2026-07-02 round 2, P2). Same content,
    # fresh int16 arrays.
    assert np.array_equal(a, b)
    assert int(np.abs(a).max()) <= int(32767 * 0.5) + 1
    louder = chirp("acknowledge", 16000, gain=0.9)
    assert int(np.abs(louder).max()) > int(np.abs(a).max())


def test_unknown_name_falls_back_without_error():
    c = chirp("does-not-exist", 24000)
    assert c.dtype == np.int16 and len(c) > 0
