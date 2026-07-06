"""Audit 2026-07-02: a per-item synthesis/playback failure must not kill the queue worker.

Before the fix, an exception in _synthesize_one/_play_one escaped run(), the worker task died
silently, and any later item's task_done() was never called -> queue.join() hung forever.
"""

import asyncio

from reachy_agent.tts_chunk_queue import CancellableTtsChunkQueue


class _FlakyTts:
    """Raises on a designated phrase, returns bytes otherwise."""

    def __init__(self, boom_on: str) -> None:
        self.boom_on = boom_on
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if text == self.boom_on:
            raise RuntimeError("TTS 500")
        return b"\x00\x01" * 8


def test_tts_worker_survives_synthesis_error_and_join_returns():
    async def run():
        client = _FlakyTts(boom_on="bad")
        queue = CancellableTtsChunkQueue(client)
        worker = asyncio.create_task(queue.run())
        for phrase in ("good one", "bad", "good two"):
            await queue.enqueue(phrase)

        # The bug would make this hang; bound it so a regression fails fast instead of stalling.
        await asyncio.wait_for(queue.join(), timeout=2.0)
        await queue.stop()
        await asyncio.wait_for(worker, timeout=2.0)
        return client.calls, queue.completed_chunks

    calls, completed = asyncio.run(run())
    # All three were attempted — the worker did not die on "bad".
    assert calls == ["good one", "bad", "good two"]
    # The two good chunks were produced despite the failure in between.
    assert len(completed) == 2
