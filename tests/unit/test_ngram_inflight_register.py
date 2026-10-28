"""The n-gram path registers its prefill with the inflight tracker from the Python token list it already
holds (it rebuilt the list with ``[int(t) for t in mx.array]``: one device scalar read per prompt token,
~0.9 s per 32k tokens per request on CPU; port of 9d9c7d8)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from tests.unit.test_ngram_spec_lossless import _engine, _tiny_model, _Tok
from yunshu_engine.inflight_prefix_sharing import InflightPrefixTracker


class _Spy(InflightPrefixTracker):
    def __init__(self):
        super().__init__()
        self.registered = []

    def register(self, request_id, token_ids, kv_cache, model_name=""):
        self.registered.append(token_ids)
        super().register(request_id, token_ids, kv_cache, model_name)


def test_ngram_registers_the_prompt_token_list():
    spy = _Spy()
    eng = _engine(_tiny_model(0))
    prompt = "abcabcabcabc"
    with patch(
        "yunshu_engine.inflight_prefix_sharing.get_inflight_tracker", return_value=spy
    ):
        out = asyncio.run(
            eng._generate_ngram_spec(prompt=prompt, max_tokens=4, temperature=0.0)
        )
    assert out.error is None
    assert spy.registered == [_Tok().encode(prompt)]
    assert type(spy.registered[0]) is list
