"""Non-streaming generate() through the orphan-adopting stream core: the request is hashed
once, the mismatch hashes are lazy, and ``timeout`` stays a deadline on the whole call."""

from __future__ import annotations

import asyncio

from yunshu_engine import orphan_adoption
from yunshu_engine.request import RequestOutput
from yunshu_engine.vlm_engine import VLMEngine


def _engine():
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = object()
    eng._model_path = "local-model"
    eng._num_requests_processed = 0
    eng._total_reasoning_tokens = 0
    return eng


def _count(monkeypatch, name):
    calls = []
    real = getattr(orphan_adoption, name)

    def wrapper(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(orphan_adoption, name, wrapper)
    return calls


def test_nonstream_request_is_hashed_once_and_parts_stay_lazy(monkeypatch):
    eng = _engine()
    keys = _count(monkeypatch, "request_key")
    parts = _count(monkeypatch, "key_parts")

    async def impl(self, messages=None, cancel_event=None, **kw):
        yield RequestOutput(
            request_id="r", new_text="hi", finished=True, finish_reason="stop"
        )

    monkeypatch.setattr(VLMEngine, "_generate_stream_impl", impl)

    out = asyncio.run(eng.generate(messages=[{"role": "user", "content": "hello"}]))
    assert out["text"] == "hi"
    assert len(keys) == 1, "generate -> collect -> stream must share one key"
    assert parts == [], "key_parts is only for mismatch logging"
