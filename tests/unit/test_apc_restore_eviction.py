"""Rapid-MLX 66b39f56 re-check: a restore evicts other HOT entries before the one it restores from,
and the restored cache is independent of the stored one.

Known and documented (docs/guides/KV_CACHE_MATRIX.md): when the restore cannot get room at all
(nothing but the source left and still no headroom), upstream ``_make_room`` evicts the source too
(spilled to the SSD tier when there is one) and then refuses. That case is pinned here as it is,
so a change to it is deliberate.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_apc_manager import _cache, _mgr  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


SOURCE = list(range(100, 356))
OTHER = list(range(1000, 1256))


def _tight(headroom_entries):
    """OTHER (older) and SOURCE (newest) in HOT; free memory for ``headroom_entries`` entries."""
    m = _mgr()
    m.store_exact_cache(OTHER, _cache(256), extra_hash=0)
    m.store_exact_cache(SOURCE, _cache(256), extra_hash=0)
    one = m._resident_bytes_locked() // 2
    m.memory_reserve_bytes = 0
    m._prefill_reserve_bytes = 0
    m.memory_max_bytes = 100 * one
    m._memory_headroom = lambda: int(headroom_entries * one)
    return m


def _hot(m):
    return sorted(e.token_ids[0] for e in m._exact_cache.values())


def test_restore_that_fits_evicts_nothing():
    m = _tight(2.5)
    cache, n = m.lookup_exact_cache(SOURCE + [5, 6, 7], extra_hash=0)
    assert n == 256 and cache is not None
    assert _hot(m) == [100, 1000]


def test_the_other_entry_is_evicted_before_the_source():
    m = _tight(1.5)  # no room for a clone even after dropping the older entry
    cache, n = m.lookup_exact_cache(SOURCE + [5], extra_hash=0)
    assert (cache, n) == (None, 0) and m.stats.memory_skips == 1
    assert _hot(m) == [100]  # OTHER went first; the source is still HOT


def test_with_no_room_at_all_the_source_is_evicted_too():
    m = _tight(0)
    cache, n = m.lookup_exact_cache(SOURCE + [5], extra_hash=0)
    assert (cache, n) == (None, 0) and m.stats.memory_skips == 1
    assert _hot(m) == []  # documented limitation, see the module docstring


def test_restored_cache_is_independent_of_the_stored_entry():
    m = _tight(10)
    cache, n = m.lookup_exact_cache(SOURCE + [5], extra_hash=0)
    assert n == 256
    stored = next(e for e in m._exact_cache.values() if e.token_ids[0] == 100)
    before = stored.prompt_cache[1].keys[..., :4, :].tolist()
    cache[1].keys[...] = 9
    mx.eval(cache[1].keys)
    assert stored.prompt_cache[1].keys[..., :4, :].tolist() == before
