"""Text engine prefix cache: which tier served the latest lookup (x_yunshu.cache.tier)."""

import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine.batched_engine import _prefix_cache_provenance  # noqa: E402
from yunshu_engine.fast_path_stats import FastPathStats  # noqa: E402
from yunshu_engine.kv_prefix_cache import KVPrefixCache  # noqa: E402


def _kv(n, layers=2):
    from mlx_lm.models.cache import KVCache

    out = []
    for _ in range(layers):
        c = KVCache()
        c.keys = mx.zeros((1, 2, n, 64))
        c.values = mx.zeros((1, 2, n, 64))
        c.offset = n
        out.append(c)
    return out


def test_hot_hit_then_miss_are_recorded():
    cache = KVPrefixCache(max_entries=8, min_prefix_length=32)
    base = list(range(1, 129))
    cache.add(mx.array(base), _kv(128))
    _, _, matched = cache.get(mx.array(base + [900, 901]))
    assert matched > 0
    assert cache.last_lookup["tier"] == "hot"
    assert cache.last_lookup["matched"] == matched
    assert cache.last_lookup["ms"] >= 0
    cache.get(mx.array([5000 + i for i in range(64)]))
    assert cache.last_lookup["tier"] == "none" and cache.last_lookup["matched"] == 0


def test_warm_entry_reports_warm():
    cache = KVPrefixCache(max_entries=8, hot_limit=1, min_prefix_length=32)
    prompts = [[i * 1000 + j for j in range(128)] for i in range(1, 4)]
    for p in prompts:
        cache.add(mx.array(p), _kv(128))
    assert any(cache._warm_flags)
    tiers = set()
    for p in prompts:
        cache.get(mx.array(p + [7, 8]))
        tiers.add(cache.last_lookup["tier"])
    assert "warm" in tiers and "hot" in tiers


def test_provenance_maps_to_the_api_vocabulary():
    cache = KVPrefixCache(max_entries=4, min_prefix_length=32)
    base = list(range(1, 129))
    cache.add(mx.array(base), _kv(128))
    _, _, matched = cache.get(mx.array(base + [1, 2]))
    assert _prefix_cache_provenance(cache, False, matched)[0] == "ram"
    assert _prefix_cache_provenance(cache, True, 10) == ("ram", None)
    assert _prefix_cache_provenance(cache, False, 0) == ("none", None)
    assert _prefix_cache_provenance(None, False, 5) == ("none", None)
    cache.last_lookup = {"tier": "ssd", "matched": 96, "ms": 12.5}
    assert _prefix_cache_provenance(cache, False, 96) == ("ssd", 12.5)


def test_fast_path_stats_carry_the_tier():
    fp = FastPathStats(None, 100)
    fp.admit(64, 36, "ssd", 8.0)
    assert (fp.stats.cache_tier, fp.stats.cache_reload_ms) == ("ssd", 8.0)
    fp2 = FastPathStats(None, 10)
    fp2.admit(0, 10)
    assert fp2.stats.cache_tier is None
