"""Unit tests for the backbone-agnostic serving-capability layer ."""

from yunshu_engine.model_backend import (
    BackendCapabilities,
    BackendKind,
    CacheClass,
    classify_cache,
    derive_capabilities,
)


# ── fake cache layers ───────────────────────────────────────────────────────
class FakeKVCache:
    """Plain full-attention cache: sliceable keys/values, trimmable."""

    def __init__(self):
        self.keys = object()
        self.values = object()

    def is_trimmable(self):
        return True


class FakeRotatingKVCache:
    """Sliding-window cache: has max_size, circular buffer."""

    def __init__(self, max_size=512):
        self.keys = object()
        self.values = object()
        self.max_size = max_size

    def is_trimmable(self):
        return True


class FakeArraysCache:
    """Recurrent/linear-attention state: no sliceable keys/values."""

    # deliberately no .keys/.values


def test_full_attention_is_resumable_and_reusable():
    cc = classify_cache([FakeKVCache(), FakeKVCache()])
    assert not cc.has_sliding_window
    assert not cc.is_hybrid
    assert cc.resumable
    caps = derive_capabilities(BackendKind.LM, [FakeKVCache()], is_mrope=False)
    assert caps.supports_kv_prefix_reuse
    assert caps.bypass_reason() is None


def test_sliding_window_blocks_reuse():
    cc = classify_cache([FakeKVCache(), FakeRotatingKVCache()])
    assert cc.has_sliding_window
    assert not cc.resumable
    caps = derive_capabilities(BackendKind.VLM, [FakeKVCache(), FakeRotatingKVCache()])
    assert not caps.supports_kv_prefix_reuse
    assert "sliding-window" in caps.bypass_reason()


def test_hybrid_recurrent_blocks_reuse():
    cc = classify_cache([FakeKVCache(), FakeArraysCache()])
    assert cc.is_hybrid
    assert not cc.resumable
    caps = derive_capabilities(BackendKind.VLM, [FakeKVCache(), FakeArraysCache()])
    assert not caps.supports_kv_prefix_reuse
    assert "hybrid" in caps.bypass_reason()


def test_mrope_reusable_but_requires_explicit_positions():
    # Plain full-attention cache + mRoPE: the cache IS reusable, but the caller
    # must supply explicit position_ids (sequential from the cache offset).
    caps = derive_capabilities(BackendKind.VLM, [FakeKVCache()], is_mrope=True)
    assert caps.cache.resumable
    assert caps.supports_kv_prefix_reuse  # cache reuse is fine
    assert caps.requires_explicit_positions  # but positions must be supplied
    assert caps.bypass_reason() is None  # not bypassed (mRoPE alone)


def test_non_mrope_does_not_require_explicit_positions():
    caps = derive_capabilities(BackendKind.VLM, [FakeKVCache()], is_mrope=False)
    assert caps.supports_kv_prefix_reuse
    assert not caps.requires_explicit_positions


def test_non_trimmable_flag_marks_hybrid():
    class NonTrimmable(FakeKVCache):
        def is_trimmable(self):
            return False

    cc = classify_cache([NonTrimmable()])
    assert cc.is_hybrid
    assert not cc.resumable


def test_layer_types_recorded():
    cc = classify_cache([FakeKVCache(), FakeRotatingKVCache(), FakeArraysCache()])
    assert cc.layer_types == ("FakeKVCache", "FakeRotatingKVCache", "FakeArraysCache")


def test_empty_cache_is_trivially_reusable():
    # No layers → nothing disqualifies (caller still gates on cache existence).
    caps = derive_capabilities(BackendKind.LM, [])
    assert caps.supports_kv_prefix_reuse


def test_capabilities_frozen():
    caps = derive_capabilities(BackendKind.LM, [FakeKVCache()])
    assert isinstance(caps, BackendCapabilities)
    assert isinstance(caps.cache, CacheClass)
    # frozen dataclass — cannot mutate
    try:
        caps.is_mrope = True  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except Exception:
        pass
