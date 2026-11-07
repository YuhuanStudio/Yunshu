"""Native KV capacity reservation for a prompt's prefill (the round driver reserves once per row).

The logical offset and canonical prefill boundaries stay unchanged; reservation preserves the existing
prefix and APC snapshots keep their detached ownership contract."""

from __future__ import annotations


def reserve_prompt_kv(prompt_cache, end: int) -> None:
    """Reserve native KV backing once its first chunk establishes the head shape.

    The logical offset and canonical prefill boundaries stay unchanged. Native
    reservation preserves the existing prefix and APC snapshots keep their
    detached ownership contract. Attention reads the reserved backing's strides.
    """
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    for cache in prompt_cache:
        if type(cache) is KVCache:
            cache.prefix_cache_reserve(end)
        elif (
            type(cache) is BatchKVCache
            and cache.keys is not None
            and cache.values is not None
            and cache.keys.shape[0] == 1
            and cache.left_padding.tolist() == [0]
            and cache._right_padding is None
        ):
            # The singleton forward extracts a native view, then merges it back
            # into this batch wrapper after every chunk. Reserve that wrapper's
            # backing so the next native view retains the capacity too.
            row = KVCache()
            row.keys, row.values = cache.keys, cache.values
            row.offset = cache._idx
            row.prefix_cache_reserve(end)
            cache.keys, cache.values = row.keys, row.values
