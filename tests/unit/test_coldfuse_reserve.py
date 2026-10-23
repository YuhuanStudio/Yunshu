"""Native KV reserve preserves bits, offsets and detached APC snapshots."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.models.cache import KVCache  # noqa: E402

from yunshu_engine.cache_prefill import reserve_prompt_kv  # noqa: E402


def test_reservation_preserves_prefix_and_subsequent_updates():
    mx.random.seed(1023)
    base, cand = KVCache(), KVCache()
    reserve_prompt_kv([cand], 32768)
    assert cand.keys is None  # head shape is established by the first chunk
    snapshot = None
    for step in range(4):
        k = mx.random.normal((1, 4, 2048, 256)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape).astype(mx.bfloat16)
        expected = base.update_and_fetch(k, v)
        if step:
            reserve_prompt_kv([cand], 32768)
        got = cand.update_and_fetch(k, v)
        mx.eval(expected, got)
        assert base.offset == cand.offset == (step + 1) * 2048
        for a, b in zip(expected, got, strict=True):
            assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
        if step == 0:
            snapshot = tuple(mx.contiguous(mx.array(x)) for x in got)
            mx.eval(snapshot)
        else:
            assert cand.keys.shape[2] == 32768
            for saved, x in zip(snapshot, got, strict=True):
                assert mx.array_equal(saved, x[..., :2048, :]).item()
    backing = cand.keys
    reserve_prompt_kv([cand], 32768)
    assert cand.keys is backing  # later chunks do not grow or copy the backing


def test_other_cache_types_do_not_reserve():
    class Other:
        def prefix_cache_reserve(self, end):
            raise AssertionError("only exact native KV cache is supported")

    reserve_prompt_kv([Other()], 32768)
