"""Pure unit tests for QwenTtsClient.stream_pcm (low-latency PCM streaming) — no network.

A fake `stream_source` feeds raw PCM bytes in arbitrary chunks (including an odd-byte split) and we
assert: every yield is whole-sample int16, the reconstructed samples equal the original, the first
chunk is yielded before all bytes are consumed (streaming, not buffer-all), and the request payload
asks for pcm+stream.
"""

import asyncio

import numpy as np

from reachy_agent.tts_chunk_queue import QwenTtsClient, QwenTtsConfig


def _collect(text, raw_chunks, capture=None):
    async def fake_source(payload, headers):
        if capture is not None:
            capture["payload"] = payload
            capture["headers"] = headers
        for c in raw_chunks:
            yield c

    client = QwenTtsClient(config=QwenTtsConfig(api_key_env="UNSET_KEY_ENV"), stream_source=fake_source)

    async def run():
        out = []
        async for sr, arr in client.stream_pcm(text):
            out.append((sr, arr))
        return out

    return asyncio.run(run())


def test_stream_reconstructs_samples_across_odd_boundaries():
    samples = np.array([0, 1, -1, 32767, -32768, 100, -100, 7], dtype="<i2")
    raw = samples.tobytes()  # 16 bytes
    # split with an ODD-length first chunk to exercise leftover carry
    chunks = [raw[:3], raw[3:9], raw[9:]]
    cap = {}
    out = _collect("hallo welt", chunks, cap)

    assert all(arr.dtype == np.int16 for _sr, arr in out)
    assert all(sr == 24_000 for sr, _arr in out)
    recon = np.concatenate([arr for _sr, arr in out])
    np.testing.assert_array_equal(recon, samples)
    # payload requested low-latency streaming PCM
    assert cap["payload"]["response_format"] == "pcm"
    assert cap["payload"]["stream"] is True
    assert cap["payload"]["input"] == "hallo welt"


def test_first_chunk_arrives_before_stream_completes():
    """stream_pcm must yield as bytes arrive, not after draining the whole source."""
    samples = np.arange(0, 1000, dtype="<i2")
    raw = samples.tobytes()
    third = len(raw) // 3

    drained = {"n": 0}

    async def fake_source(payload, headers):
        for c in (raw[:third], raw[third : 2 * third], raw[2 * third :]):
            drained["n"] += 1
            yield c

    client = QwenTtsClient(config=QwenTtsConfig(api_key_env="UNSET_KEY_ENV"), stream_source=fake_source)

    async def run():
        gen = client.stream_pcm("x")
        first_sr, first_arr = await gen.__anext__()
        # at the moment of the first yield, the source must NOT have been fully drained
        chunks_pulled_at_first_yield = drained["n"]
        rest = [(first_sr, first_arr)]
        async for sr, arr in gen:
            rest.append((sr, arr))
        return chunks_pulled_at_first_yield, rest

    pulled_at_first, all_chunks = asyncio.run(run())
    assert pulled_at_first == 1, "first audio chunk should be emitted after the first source chunk"
    recon = np.concatenate([a for _s, a in all_chunks])
    np.testing.assert_array_equal(recon, samples)


def test_empty_text_yields_nothing():
    out = _collect("   ", [b"\x01\x02"])
    assert out == []
