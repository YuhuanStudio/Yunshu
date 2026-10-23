"""Native KV reserve preserves bits, offsets and detached APC snapshots."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.models.cache import BatchKVCache, KVCache  # noqa: E402

from yunshu_engine.cache_prefill import reserve_prompt_kv  # noqa: E402


@pytest.mark.parametrize(
    "factory", [KVCache, lambda: BatchKVCache([0])], ids=["native", "singleton-batch"]
)
def test_reservation_preserves_prefix_and_subsequent_updates(factory):
    mx.random.seed(1023)
    base, cand = factory(), factory()
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


def test_singleton_forward_retains_reserved_capacity():
    from yunshu_engine.kernels.singleton_cache import extract_rows, merge_rows

    caches = [BatchKVCache([0])]
    reference = KVCache()
    for step in range(3):
        reserve_prompt_kv(caches, 32768)
        rows = extract_rows(caches)
        x = mx.full((1, 4, 2048, 256), step + 1, dtype=mx.bfloat16)
        expected = reference.update_and_fetch(x, x)
        actual = rows[0].update_and_fetch(x, x)
        caches = merge_rows(rows)
        mx.eval(expected, actual)
        assert caches[0]._idx == reference.offset
        if step:
            assert caches[0].keys.shape[2] == 32768
        for a, b in zip(expected, actual, strict=True):
            assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


@pytest.mark.parametrize("padding", [[1], [0, 0]])
def test_padded_or_multirow_batch_not_reserved(padding):
    cache = BatchKVCache(padding)
    x = mx.zeros((len(padding), 4, 64, 256), dtype=mx.bfloat16)
    cache.update_and_fetch(x, x)
    backing = cache.keys
    reserve_prompt_kv([cache], 32768)
    assert cache.keys is backing and cache.keys.shape[2] == 256
