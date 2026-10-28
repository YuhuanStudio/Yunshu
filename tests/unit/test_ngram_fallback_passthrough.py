"""The n-gram spec fallbacks to _generate_fast must carry every request knob, including
kv_cache_breakpoints (prefix-cache entries at cache_control positions), priority, min_tokens
and ignore_eos; they used to be dropped silently."""

from __future__ import annotations

import asyncio
import types

from yunshu_engine.batched_engine import BatchedEngine

KW = dict(priority=7, kv_cache_breakpoints=[12, 40], min_tokens=3, ignore_eos=True)


def _engine(captured):
    e = BatchedEngine.__new__(BatchedEngine)
    e._model = object()  # make_prompt_cache fails -> cache "not trimmable"
    e._tokenizer = types.SimpleNamespace(encode=lambda t: [1, 2, 3])

    async def _fast(**kw):
        captured.update(kw)
        return "fast-result"

    e._generate_fast = _fast
    return e


def test_untrimmable_cache_fallback_passes_breakpoints_and_priority():
    got: dict = {}
    e = _engine(got)
    out = asyncio.run(e._generate_ngram_spec(prompt="hi", **KW))
    assert out == "fast-result"
    for k, v in KW.items():
        assert got[k] == v, k


def test_reasoning_gate_fallback_passes_breakpoints_and_priority(monkeypatch):
    import mlx_lm.models.cache as c

    from yunshu_engine import reasoning_gate

    monkeypatch.setattr(c, "make_prompt_cache", lambda m: [])
    monkeypatch.setattr(c, "can_trim_prompt_cache", lambda cache: True)
    monkeypatch.setattr(
        reasoning_gate, "prompt_opens_reasoning", lambda ids, tok: (True, None)
    )
    got: dict = {}
    e = _engine(got)
    e._new_request_proposer = lambda: object()
    out = asyncio.run(
        e._generate_ngram_spec(prompt="hi", json_schema={"type": "object"}, **KW)
    )
    assert out == "fast-result"
    for k, v in KW.items():
        assert got[k] == v, k


def test_dispatch_call_site_forwards_the_knobs():
    import inspect
    import re

    src = inspect.getsource(BatchedEngine)
    call = re.search(r"self\._generate_ngram_spec\((.*?)\n            \)", src, re.S)
    assert call
    for k in ("priority", "kv_cache_breakpoints", "min_tokens", "ignore_eos"):
        assert f"{k}={k}" in call.group(1), k
