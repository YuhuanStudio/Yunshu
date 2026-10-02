"""Cache-state extraction, reasoning tracking and stop matchers against the locked mlx-lm."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from yunshu_engine.scheduler import Scheduler, _ReasoningTracker
from yunshu_kv.mlx_cache import extract_cache_state


def test_extract_cache_state_trims_to_offset():
    c = KVCache()
    k = mx.ones((1, 2, 3, 4))
    c.update_and_fetch(k, k)
    state = extract_cache_state(c)
    assert state["offset"] == 3 and state["keys"].shape[2] == 3


def test_extract_cache_state_rotating():
    c = RotatingKVCache(max_size=8)
    k = mx.ones((1, 2, 3, 4))
    c.update_and_fetch(k, k)
    state = extract_cache_state(c)
    assert state["keys"].shape[2] == 3 and state["max_size"] == 8


class _Tok:
    has_thinking = True
    think_start_tokens = (10,)
    think_end_tokens = (11, 12)


def test_reasoning_tracker_follows_think_markers():
    t = _ReasoningTracker(_Tok())
    states = [t.advance(x) for x in (1, 10, 5, 11, 12, 6)]
    assert states == [
        "normal",
        "reasoning",
        "reasoning",
        "reasoning",
        "normal",
        "normal",
    ]


def test_reasoning_tracker_without_thinking_stays_normal():
    class _Plain:
        has_thinking = False

    t = _ReasoningTracker(_Plain())
    assert [t.advance(x) for x in (10, 11)] == ["normal", "normal"]


def test_stop_kwargs_and_state_machine_use_stop_sequences():
    from types import SimpleNamespace

    tok = SimpleNamespace(
        eos_token_ids=[7],
        has_thinking=False,
        encode=lambda text, add_special_tokens=False: [ord(c) for c in text],
    )
    sm = Scheduler._make_state_machine(SimpleNamespace(tokenizer=tok), ["ab"], [9])
    assert type(sm).__name__ == "StopSequences"
    assert Scheduler._stop_kwargs(sm) == {"stop_sequences": [sm]}
