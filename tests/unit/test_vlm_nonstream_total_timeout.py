"""A non-streaming ``timeout`` bounds the whole call, also through the orphan-adopting core
(which reads the same field as a gap between tokens when streaming)."""

from __future__ import annotations

import asyncio
import time

from yunshu_engine.request import RequestOutput
from yunshu_engine.vlm_engine import VLMEngine


def _engine():
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = object()
    eng._model_path = "local-model"
    eng._num_requests_processed = 0
    eng._total_reasoning_tokens = 0
    return eng


def test_nonstream_timeout_is_a_total_deadline(monkeypatch):
    eng = _engine()

    async def impl(self, messages=None, cancel_event=None, **kw):
        # a reply that keeps producing tokens: never idle for longer than the timeout
        for i in range(100):
            if (
                cancel_event.is_set()
            ):  # the real engine stops its row on the cancel event
                return
            await asyncio.sleep(0.05)
            yield RequestOutput(request_id="r", new_text="x", completion_tokens=i + 1)
        yield RequestOutput(
            request_id="r", new_text="", finished=True, finish_reason="stop"
        )

    monkeypatch.setattr(VLMEngine, "_generate_stream_impl", impl)
    t0 = time.perf_counter()
    out = asyncio.run(
        eng.generate(
            messages=[{"role": "user", "content": "hello"}], timeout_seconds=0.4
        )
    )
    assert out["finish_reason"] == "timeout"
    assert time.perf_counter() - t0 < 2.0
