"""Cache and stop-matcher helpers that read the same under mlx-lm 0.31 and 0.32."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from yunshu_engine.scheduler import Scheduler, _ReasoningTracker
from yunshu_kv.mlx_cache import cache_keys_values, extract_cache_state


class _OldKV:
    """mlx-lm 0.31 shape: ``.state`` is ``(keys, values)``, no keys_and_values()."""

    def __init__(self, k, v):
        self.keys, self.values, self.offset = k, v, k.shape[2]
        self.state = (k, v)


def test_cache_keys_values_old_style_state():
    k = mx.zeros((1, 2, 5, 4))
    got = cache_keys_values(_OldKV(k, k))
    assert got[0].shape == (1, 2, 5, 4)


def test_cache_keys_values_real_cache_is_trimmed_to_offset():
    c = KVCache()
    k = mx.ones((1, 2, 3, 4))
    c.update_and_fetch(k, k)
    keys, values = cache_keys_values(c)
    assert keys.shape == (1, 2, 3, 4) and values.shape == (1, 2, 3, 4)
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


def test_stop_kwargs_name_the_right_insert_argument():
    class StopSequences:  # mlx-lm >= 0.32 type, matched by name
        pass

    assert Scheduler._stop_kwargs(StopSequences()).keys() == {"stop_sequences"}
    assert Scheduler._stop_kwargs(object()).keys() == {"state_machines"}


def test_make_state_machine_matches_the_installed_mlx_lm():
    import importlib
    from types import SimpleNamespace

    tok = SimpleNamespace(
        eos_token_ids=[7],
        has_thinking=False,
        encode=lambda text, add_special_tokens=False: [ord(c) for c in text],
    )
    sm = Scheduler._make_state_machine(SimpleNamespace(tokenizer=tok), ["ab"], [9])
    old = hasattr(importlib.import_module("mlx_lm.generate"), "SequenceStateMachine")
    key = next(iter(Scheduler._stop_kwargs(sm)))
    assert key == ("state_machines" if old else "stop_sequences")
