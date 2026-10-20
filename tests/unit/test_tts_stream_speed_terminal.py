"""Streaming TTS: honor speed without native speed control, and never drop an audio
chunk to make room for the terminal marker."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from yunshu_engine import audio_engine


class _Model:
    sample_rate = 24000

    def __init__(self, n, samples=100):
        self.n = n
        self.samples = samples

    def generate(self, text, verbose=False):  # no speed parameter
        raise AssertionError

    def stream_generate(self, text, verbose=False):
        for i in range(self.n):
            yield SimpleNamespace(
                audio=np.full(self.samples, (i + 1) / 1000.0, dtype=np.float32),
                text="x",
            )


def _engine(model):
    eng = audio_engine.TTSEngine.__new__(audio_engine.TTSEngine)
    eng._model = model
    eng._executor = ThreadPoolExecutor(max_workers=1)
    eng._stats_lock = threading.Lock()
    eng._stream_count = 0
    eng._total_stream_ms = 0.0
    return eng


def _pcm_samples(chunks):
    total = 0
    for i, c in enumerate(chunks):
        n = len(c["audio"])
        if i == 0:
            n -= 44  # WAV header
        total += n // 2
    return total


@pytest.mark.asyncio
async def test_stream_resamples_when_model_has_no_speed():
    eng = _engine(_Model(n=3, samples=100))
    chunks = [c async for c in eng.synthesize_stream("hi", speed=2.0)]
    eng._executor.shutdown(wait=True)
    assert _pcm_samples(chunks) == 3 * 50


@pytest.mark.asyncio
async def test_terminal_marker_does_not_drop_audio():
    n = 70
    eng = _engine(_Model(n=n, samples=10))
    agen = eng.synthesize_stream("hi")
    chunks = [await agen.__anext__()]
    await asyncio.sleep(0.5)  # worker fills the queue and reaches the terminal put
    async for c in agen:
        chunks.append(c)
    eng._executor.shutdown(wait=True)
    assert chunks[-1]["is_final"] is True
    assert _pcm_samples(chunks) == n * 10
