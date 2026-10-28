"""The text fast paths register their prefill with the inflight tracker from the Python
token list they already hold. They used to rebuild it with ``[int(t) for t in mx.array]``
(one device scalar read per prompt token, twice per request) and to run a
``find_prefix`` lookup whose result was discarded (live-borrow reuse is disabled)."""

from unittest.mock import patch

from yunshu_engine.inflight_prefix_sharing import InflightPrefixTracker

from .tiny_text_engine import FakeTokenizer, collect_stream, make_engine


class _Spy(InflightPrefixTracker):
    def __init__(self):
        super().__init__()
        self.registered: list[list[int]] = []
        self.lookups = 0

    def register(self, request_id, token_ids, kv_cache, model_name=""):
        self.registered.append(token_ids)
        super().register(request_id, token_ids, kv_cache, model_name)

    def find_prefix(self, *a, **k):
        self.lookups += 1
        return super().find_prefix(*a, **k)


PROMPT = "the quick brown fox jumps over the lazy dog " * 4


async def test_generate_fast_registers_the_python_list_without_a_lookup():
    spy = _Spy()
    eng = make_engine()
    with patch(
        "yunshu_engine.inflight_prefix_sharing.get_inflight_tracker", return_value=spy
    ):
        out = await eng._generate_fast(prompt=PROMPT, max_tokens=4, temperature=0.0)
    assert out.error is None and out.completion_tokens == 4
    expected = FakeTokenizer().encode(PROMPT)
    assert spy.registered == [expected]
    assert type(spy.registered[0]) is list
    assert spy.lookups == 0


async def test_stream_generate_fast_registers_the_python_list_without_a_lookup():
    spy = _Spy()
    eng = make_engine()
    with patch(
        "yunshu_engine.inflight_prefix_sharing.get_inflight_tracker", return_value=spy
    ):
        text, last = await collect_stream(
            eng, prompt=PROMPT, max_tokens=4, temperature=0.0
        )
    assert last.error is None and last.completion_tokens == 4
    assert spy.registered == [FakeTokenizer().encode(PROMPT)]
    assert spy.lookups == 0
