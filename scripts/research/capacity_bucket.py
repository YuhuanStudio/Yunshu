"""Research-only native restore buckets; cache arithmetic/append step unchanged."""

import functools
import logging


def bucket_capacity(tokens, bucket=512):
    if bucket <= 0:
        raise ValueError("capacity bucket must be positive")
    return ((int(tokens) + bucket - 1) // bucket) * bucket


def install(bucket=512):
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import KVCache

    bucket_capacity(0, bucket)
    original = apc_adapters.clone_cache_entry
    counts = dict(enabled=True, restores=0, extra_tokens=0)

    @functools.wraps(original)
    def clone(cache, *, min_capacity_tokens, eval_targets):
        if (
            counts["enabled"]
            and type(cache) is KVCache
            and min_capacity_tokens is not None
        ):
            capacity = bucket_capacity(min_capacity_tokens, bucket)
            if counts["restores"] == 0:
                logging.getLogger("yunshu_engine.research.capacity_bucket").info(
                    "Native restore capacity bucket engaged: %d tokens", bucket
                )
            counts["restores"] += 1
            counts["extra_tokens"] += capacity - min_capacity_tokens
            min_capacity_tokens = capacity
        return original(
            cache, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
        )

    apc_adapters.clone_cache_entry = clone

    def uninstall():
        apc_adapters.clone_cache_entry = original

    return counts, uninstall
