"""YUNSHU_NEXT_TURN_PREFILL=decode (default, user 2026-11-02): the cache a reply was decoded into is
stored under prompt + the tokens it holds, so the follow-up prefills only its new message (turn-2 TTFT
0.6 s -> TensorFold / mlx-serve's ~0.1 s, which reuse the reply KV)."""

from types import SimpleNamespace

from yunshu_engine import settings
from yunshu_engine.vlm_batch_runner import VLMBatchRunner, _Job


class _KV:
    def __init__(self, offset):
        self.keys, self.offset = object(), offset


class _Coord:
    def __init__(self):
        self.calls = []

    def store_checkpoint(self, tokens, caches, *, extra_hash=0):
        self.calls.append((list(tokens), extra_hash))
        return True


def _group(offset):
    coord = _Coord()
    gen = SimpleNamespace(
        apc=coord,
        _generation_batch=SimpleNamespace(prompt_cache=[_KV(offset), object()]),
    )
    return SimpleNamespace(gen=gen, last_caches=None, spec=True, jobs={}), coord


def _job(prompt, generated, salt=7):
    j = _Job.__new__(_Job)
    j.ids, j.generated, j.apc_salt = list(prompt), list(generated), salt
    return j


def test_the_reply_cache_is_stored_under_prompt_plus_held_tokens():
    group, coord = _group(
        offset=6
    )  # 4 prompt + 2 of 3 reply tokens fed (the last one is pending)
    VLMBatchRunner._store_decode_cache(group, _job([1, 2, 3, 4], [10, 11, 12]))
    assert coord.calls == [([1, 2, 3, 4, 10, 11], 7)]


def test_a_cache_that_disagrees_with_the_tokens_is_not_stored():
    for offset in (4, 9):  # no reply token held / more than were emitted
        group, coord = _group(offset)
        VLMBatchRunner._store_decode_cache(group, _job([1, 2, 3, 4], [10, 11, 12]))
        assert coord.calls == []


def test_decode_is_the_default_mode():
    assert settings.get("YUNSHU_NEXT_TURN_PREFILL") == "decode"


def test_emitted_tokens_are_recorded():
    import queue

    r = VLMBatchRunner.__new__(VLMBatchRunner)
    r._vocab_owner, r.gc_guard = None, None
    j = _job([1], [])
    j.terminal, j.out = False, queue.Queue()
    j.stats, j.first_delivery = SimpleNamespace(generated=0), None
    r._emit(j, (42, None))
    r._emit(j, (43, None))
    assert j.generated == [42, 43]


def test_the_lanes_batch_cache_counts_rows_past_its_left_padding():
    """The speculative lane's single-row BatchKVCache has an array offset; the rows it holds are
    _idx past the left padding (the first 27B check stored nothing: only int offsets were read)."""
    import numpy as np

    class _BatchKV:
        keys = object()
        offset = np.array([99])
        _idx = 7
        left_padding = np.array([1])

    coord = _Coord()
    gen = SimpleNamespace(
        apc=coord, _generation_batch=SimpleNamespace(prompt_cache=[_BatchKV(), object()])
    )
    group = SimpleNamespace(gen=gen, last_caches=None, spec=True, jobs={})
    VLMBatchRunner._store_decode_cache(group, _job([1, 2, 3, 4], [10, 11, 12]))
    assert coord.calls == [([1, 2, 3, 4, 10, 11], 7)]
