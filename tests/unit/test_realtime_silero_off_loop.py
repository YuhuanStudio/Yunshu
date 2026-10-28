"""Silero VAD (model load + MLX feed) runs on the MLX executor thread, not the event loop.

The first append downloaded/initialised the model synchronously on the loop (stalling every
other session and request) and each feed ran MLX evals there while generation ran on the
dedicated MLX thread (the executor docstring: all MLX work must be serialized on one thread).
"""

from __future__ import annotations

import struct
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import yunshu_gateway.routers.realtime as rt


class _Cfg:
    chunk_size = 512


class _Branch:
    config = _Cfg()


class _Model:
    def __init__(self, seen):
        self.seen = seen

    def _branch(self, rate):
        return _Branch()

    def feed(self, samples, state, rate):
        self.seen["feed"] = threading.current_thread().name
        return np.array([0.9]), (state or 0) + 1


@pytest.mark.asyncio
async def test_silero_load_and_feed_run_on_the_mlx_thread(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_VAD", "silero")
    seen: dict = {}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake-mlx")
    fake = types.ModuleType("yunshu_engine.mlx_executor")
    fake.get_mlx_executor = lambda: pool
    monkeypatch.setitem(sys.modules, "yunshu_engine.mlx_executor", fake)

    def _load():
        seen["load"] = threading.current_thread().name
        return _Model(seen)

    monkeypatch.setattr(rt, "_get_silero_vad", _load)
    ws = MagicMock()
    ws.send_json = AsyncMock()
    s = rt.RealtimeSession(ws)
    s.session.turn_detection = {"type": "server_vad", "threshold": 0.5}
    pcm = struct.pack("<24000h", *([9000] * 24000))
    try:
        await s._run_vad(pcm)
    finally:
        pool.shutdown()
    assert seen["load"].startswith("fake-mlx")
    assert seen["feed"].startswith("fake-mlx")
    assert s._vad_speaking is True
