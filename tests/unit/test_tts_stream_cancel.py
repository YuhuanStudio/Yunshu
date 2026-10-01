"""R28: TTS stream cancel is cooperative at chunk granularity, and a stalled
client never pins the shared executor for long."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from yunshu_engine import audio_engine


class _Model:
    sample_rate = 24000

    def __init__(self, n=500):
        self.n = n
        self.computed = 0

    def generate(self, text, verbose=False):  # signature only
        raise AssertionError

    def stream_generate(self, text, verbose=False):
        for _ in range(self.n):
            self.computed += 1  # a "forward" happened
            yield SimpleNamespace(audio=np.zeros(16, dtype=np.float32), text="x")


def _engine(model):
    eng = audio_engine.TTSEngine.__new__(audio_engine.TTSEngine)
    eng._model = model
    eng._executor = ThreadPoolExecutor(max_workers=1)
    eng._stats_lock = threading.Lock()
    eng._stream_count = 0
    eng._total_stream_ms = 0.0
    return eng


@pytest.mark.asyncio
async def test_cancel_before_start_computes_no_chunk():
    model = _Model()
    eng = _engine(model)
    ev = asyncio.Event()
    ev.set()
    chunks = [c async for c in eng.synthesize_stream("hi", cancel_event=ev)]
    eng._executor.shutdown(wait=True)
    assert model.computed == 0
    assert all(not c.get("audio") for c in chunks)


@pytest.mark.asyncio
async def test_stalled_client_releases_executor_quickly():
    model = _Model(n=500)
    eng = _engine(model)
    agen = eng.synthesize_stream("hi")
    await agen.__anext__()  # first chunk, then the client stops reading
    await asyncio.sleep(0.5)
    t0 = time.monotonic()
    done = eng._executor.submit(lambda: 1)  # another request wants the thread
    assert await asyncio.wait_for(asyncio.wrap_future(done), 4.0) == 1
    assert time.monotonic() - t0 < 4.0
    await agen.aclose()
    eng._executor.shutdown(wait=True)
