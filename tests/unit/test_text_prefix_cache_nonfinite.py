"""A cache with a non-finite recurrent state is never published to the text path's prefix cache."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from yunshu_engine.kv_prefix_cache import KVPrefixCache, recurrent_state_finite

N = 64


def _cache(value=0.5, dtype=mx.float16):
    rec = ArraysCache(size=2)
    rec.cache = [mx.ones((1, 4, 8), dtype=dtype), mx.full((1, 2, 3), value, dtype)]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, N, 4)), mx.ones((1, 2, N, 4)))
    mx.eval(rec.cache, kv.keys)
    return [rec, kv]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_recurrent_state_is_not_stored(bad):
    pc = KVPrefixCache(max_entries=4, min_prefix_length=8)
    pc.add(mx.arange(N), _cache(bad))
    assert pc.size == 0
    assert recurrent_state_finite(_cache(bad)) is False


def test_finite_state_is_still_stored():
    pc = KVPrefixCache(max_entries=4, min_prefix_length=8)
    pc.add(mx.arange(N), _cache())
    assert pc.size == 1
    assert recurrent_state_finite(_cache()) is True


def test_dense_kv_only_cache_is_not_inspected():
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, N, 4)), mx.ones((1, 2, N, 4)))
    assert recurrent_state_finite([kv]) is True


def _boundary_run(monkeypatch, bad=None):
    """Drive the real ``_capture_hybrid_prefix`` (lazy mid-prefill snapshots) and count the
    blocking finite checks it triggers."""
    import yunshu_engine.kv_prefix_cache as kpc
    from yunshu_engine.engine_fast import EngineFastMixin

    syncs = []
    real = kpc.recurrent_state_finite
    monkeypatch.setattr(
        kpc, "recurrent_state_finite", lambda c: syncs.append(1) or real(c)
    )
    pc = KVPrefixCache(max_entries=8, min_prefix_length=8)
    cache = _cache(bad if bad is not None else 0.5)
    model = lambda chunk, cache: None  # noqa: E731 - recurrent state is fixed above
    ids = mx.arange(8 * 5 + 1)
    EngineFastMixin._capture_hybrid_prefix(None, model, ids, cache, pc, 0, 8)
    return pc, syncs


def test_hybrid_boundary_snapshots_do_not_sync_per_boundary(monkeypatch):
    pc, syncs = _boundary_run(monkeypatch)
    assert pc.size == 5  # one snapshot per block boundary
    assert syncs == []  # was one host sync per boundary
    assert pc.get(mx.arange(8 * 5 + 1))[0] is not None  # restorable; flags read in one batch


def test_deferred_check_still_drops_a_poisoned_boundary_before_any_restore(monkeypatch):
    pc, syncs = _boundary_run(monkeypatch, bad=float("nan"))
    assert syncs == []
    assert pc.get(mx.arange(8 * 5 + 1))[0] is None
    assert pc.size == 0


def test_prompt_cache_store_refuses_a_non_finite_state():
    from yunshu_engine.prompt_cache import PromptCacheManager

    pm = PromptCacheManager()
    assert pm.store("bad", _cache(float("nan")), token_count=N) is False
    assert pm.lookup("bad") is None
    assert pm.store("ok", _cache(), token_count=N) is True
