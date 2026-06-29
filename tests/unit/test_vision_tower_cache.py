"""cross-request vision-feature cache (wraps the VLM vision tower)."""

from __future__ import annotations

import mlx.core as mx

from yunshu_engine.vlm_engine import _CachingVisionTower, _wrap_vision_towers


class _FakeTower:
    def __init__(self):
        self.calls = 0
        self.patch_embed = "sentinel"  # for attribute-proxy test

    def __call__(self, pixel_values, *args, **kwargs):
        self.calls += 1
        # deterministic "features" derived from input so we can check identity
        return pixel_values * 2.0


def test_repeat_image_is_a_cache_hit():
    t = _FakeTower()
    w = _CachingVisionTower(t, max_entries=4)
    px = mx.ones((3, 4, 4))
    out1 = w(px)
    out2 = w(px)  # same image → must NOT re-encode
    assert t.calls == 1, "vision tower should run once for a repeated image"
    assert bool(mx.all(out1 == out2).item())
    s = w.cache_stats()
    assert s["vision_tower_cache_hits"] == 1
    assert s["vision_tower_cache_misses"] == 1


def test_different_image_is_a_miss():
    t = _FakeTower()
    w = _CachingVisionTower(t, max_entries=4)
    w(mx.ones((3, 4, 4)))
    w(mx.zeros((3, 4, 4)))  # different pixels → re-encode
    assert t.calls == 2


def test_grid_args_are_part_of_key():
    t = _FakeTower()
    w = _CachingVisionTower(t, max_entries=4)
    px = mx.ones((3, 4, 4))
    w(px, mx.array([1, 1, 4]))
    w(px, mx.array([1, 2, 2]))  # same pixels, different grid → miss
    assert t.calls == 2


def test_lru_eviction():
    t = _FakeTower()
    w = _CachingVisionTower(t, max_entries=2)
    a, b, c = mx.ones((1, 2)), mx.full((1, 2), 2.0), mx.full((1, 2), 3.0)
    w(a)
    w(b)
    w(c)  # evicts a
    w(a)  # a was evicted → miss (re-encode)
    assert t.calls == 4
    assert w.cache_stats()["vision_tower_cache_entries"] == 2


def test_attribute_proxy():
    t = _FakeTower()
    w = _CachingVisionTower(t, max_entries=2)
    assert w.patch_embed == "sentinel"  # proxied to the wrapped tower


def test_wrap_vision_towers_finds_direct_and_nested():
    class _DirectModel:
        def __init__(self):
            self.vision_tower = _FakeTower()

    class _Thinker:
        def __init__(self):
            self.vision_tower = _FakeTower()

    class _NestedModel:
        def __init__(self):
            self.thinker = _Thinker()

    dm = _DirectModel()
    wraps = _wrap_vision_towers(dm)
    assert len(wraps) == 1
    assert isinstance(dm.vision_tower, _CachingVisionTower)

    nm = _NestedModel()
    wraps2 = _wrap_vision_towers(nm)
    assert len(wraps2) == 1
    assert isinstance(nm.thinker.vision_tower, _CachingVisionTower)

    # idempotent — second call does not double-wrap
    assert _wrap_vision_towers(dm) == []


def test_distinct_mlx_images_get_distinct_keys_no_collision():
    """regression: two DIFFERENT images must never share a cache key.

    The original _key did np.asarray(x, dtype=float32) and, on failure (e.g. an
    mlx.array), fell back to repr(x) — whose large-array form is TRUNCATED, so two
    distinct images collided to one key and the cache returned the WRONG image's
    features (read QUASAR as BANANA in the VLM-OCR gate). The fix materializes real
    bytes via np.array and never uses a colliding repr.
    """
    import mlx.core as mx

    from yunshu_engine.vlm_engine import _CachingVisionTower

    a = mx.random.normal((1, 3, 64, 64))
    b = mx.random.normal((1, 3, 64, 64))
    ka = _CachingVisionTower._key(a, ())
    kb = _CachingVisionTower._key(b, ())
    # Same array → same key (cache hit); different arrays → different key (no collision).
    assert _CachingVisionTower._key(a, ()) == ka
    assert ka != kb
