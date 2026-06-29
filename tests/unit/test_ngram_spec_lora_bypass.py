"""the KV-prefix-cache LoRA bypass (the cache is keyed on token ids + model
name but NOT the active adapter, so reusing/storing KV under a different adapter serves the
wrong adapter's KV) was wired into the two LIVE fast paths (_generate_fast / _stream_fast)
but NOT into _generate_ngram_spec — the un-propagated keystone sibling found by a
KVPrefixCache + spec-decode hunt. Latent today (a LoRA request fails _gemma4_spec_eligible
so spec_decode is forced off and this path isn't entered with an adapter), but a fail-safe
keystone propagation: bypass the prefix cache there too when an adapter is active.
"""

from __future__ import annotations

import inspect

from yunshu_engine.batched_engine import BatchedEngine


def test_ngram_spec_bypasses_prefix_cache_under_lora():
    src = inspect.getsource(BatchedEngine._generate_ngram_spec)
    # the n-gram spec path nulls the prefix cache when an adapter is active
    i = src.index("prefix_cache = self._kv_prefix_cache")
    window = src[i : i + 1100]
    assert "if lora_adapter is not None:" in window
    assert "prefix_cache = None" in window
    # and every prefix_cache use is guarded on `is not None`, so the None disables get+add
    assert "prefix_cache.get(ids)\n                    if prefix_cache is not None" in src
    assert "if prefix_cache is not None:" in src  # the add-site guard
