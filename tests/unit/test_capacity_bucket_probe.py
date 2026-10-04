"""Adjacent native restore capacities share a size without changing valid state."""

import pytest
from scripts.research.capacity_bucket import bucket_capacity, install


def test_same_size_bucket_spans_warm_and_turn2_without_global_step_change():
    assert bucket_capacity(32775) == bucket_capacity(33057) == 33280
    assert bucket_capacity(8199) == bucket_capacity(8481) == 8704
    assert bucket_capacity(33281) == 33792
    with pytest.raises(ValueError):
        bucket_capacity(3, 0)


def test_buckets_only_native_restore_not_stores_or_custom_cache(monkeypatch):
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import ArraysCache, KVCache

    calls = []

    def original(cache, **kwargs):
        calls.append((cache, kwargs))
        return cache

    monkeypatch.setattr(apc_adapters, "clone_cache_entry", original)
    before = KVCache.step
    counts, undo = install()
    try:
        native, arrays, custom = KVCache(), ArraysCache(2), object()
        for cache, tokens in (
            (native, 32775),
            (native, None),
            (arrays, 32775),
            (custom, 32775),
        ):
            assert (
                apc_adapters.clone_cache_entry(
                    cache, min_capacity_tokens=tokens, eval_targets=[]
                )
                is cache
            )
        assert [kwargs["min_capacity_tokens"] for _, kwargs in calls] == [
            33280,
            None,
            32775,
            32775,
        ]
        assert counts["restores"] == 1
        assert KVCache.step == before
    finally:
        undo()
    assert apc_adapters.clone_cache_entry is original


def test_bucket_keeps_production_restore_marker_across_engine_startup(monkeypatch):
    from mlx_vlm import apc_adapters

    from yunshu_engine.kernels import cache_restore

    def original(cache, **kwargs):
        return kwargs["min_capacity_tokens"]

    original._yunshu_restore_views = True
    monkeypatch.setattr(apc_adapters, "clone_cache_entry", original)
    counts, undo = install()
    try:
        wrapper = apc_adapters.clone_cache_entry
        cache_restore.install()
        assert apc_adapters.clone_cache_entry is wrapper
        counts["enabled"] = False
        assert wrapper(object(), min_capacity_tokens=11, eval_targets=[]) == 11
    finally:
        undo()


def test_bucket_restore_and_append_keep_valid_bits_and_source_ownership(monkeypatch):
    import mlx.core as mx
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.kernels.cache_restore import clone_native_restore

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    undo = None
    try:
        source = KVCache()
        source.update_and_fetch(
            mx.arange(32 * 8).reshape(1, 1, 32, 8).astype(mx.bfloat16),
            mx.ones((1, 1, 32, 8), mx.bfloat16) * 3,
        )
        mx.eval(source.state)
        expected = clone_native_restore(source, min_capacity_tokens=33, eval_targets=[])
        monkeypatch.setattr(apc_adapters, "clone_cache_entry", clone_native_restore)
        _, undo = install()
        actual = apc_adapters.clone_cache_entry(
            source, min_capacity_tokens=33, eval_targets=[]
        )
        assert expected.keys.shape[2] == 256 and actual.keys.shape[2] == 512
        assert actual.meta_state == expected.meta_state
        for length in (17, 250):
            keys = mx.ones((1, 1, length, 8), mx.bfloat16) * 7
            values = mx.ones((1, 1, length, 8), mx.bfloat16) * 9
            for cache in (expected, actual):
                cache.update_and_fetch(keys, values)
            mx.eval(expected.state, actual.state)
            assert actual.offset == expected.offset
            assert all(
                mx.array_equal(a, b).item()
                for a, b in zip(expected.state, actual.state, strict=True)
            )
        source.keys[..., :32, :] = -1
        assert not mx.all(actual.keys[..., :32, :] == -1).item()
        actual.values[..., :32, :] = 2
        assert mx.all(source.values[..., :32, :] == 3).item()
    finally:
        if undo:
            undo()
        mx.set_default_device(previous)
