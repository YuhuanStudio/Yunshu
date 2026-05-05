"""Yunshu MLX Cache Integration — Real KVCache manipulation.

Handles MLX's actual cache types (KVCache, RotatingKVCache, ArraysCache,
CacheList, BatchKVCache) with proper tensor operations.

Based on deep study of mlx-lm's cache.py and oMLX's type_handlers.py.

Key MLX KVCache contract:
- keys/values shape: (B, n_kv_heads, L, head_dim), axis 2 = sequence dim
- Direct attribute assignment: cache.keys = k; cache.values = v; cache.offset = n
- cache.state returns (keys, values) tuple
- Slicing: keys[:, :, start:end, :] for block extraction
- Concatenation: mx.concatenate(list, axis=2) for block assembly
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional

import mlx.core as mx


class CacheType(Enum):
    KVCACHE = auto()
    ROTATING_KVCACHE = auto()
    ARRAYS_CACHE = auto()
    CACHE_LIST = auto()
    QUANTIZED_KVCACHE = auto()
    BATCH_KVCACHE = auto()
    BATCH_ROTATING_KVCACHE = auto()
    UNKNOWN = auto()


def detect_cache_type(cache_obj: Any) -> CacheType:
    """Detect the MLX cache type from a cache object.

    Priority order matches oMLX's type_registry.py.
    """
    class_name = type(cache_obj).__name__

    name_map = {
        "KVCache": CacheType.KVCACHE,
        "RotatingKVCache": CacheType.ROTATING_KVCACHE,
        "BatchKVCache": CacheType.BATCH_KVCACHE,
        "BatchRotatingKVCache": CacheType.BATCH_ROTATING_KVCACHE,
        "ArraysCache": CacheType.ARRAYS_CACHE,
        "QuantizedKVCache": CacheType.QUANTIZED_KVCACHE,
        "CacheList": CacheType.CACHE_LIST,
    }

    if class_name in name_map:
        return name_map[class_name]

    # Heuristic fallbacks
    if hasattr(cache_obj, "caches") and isinstance(getattr(cache_obj, "caches", None), (list, tuple)):
        return CacheType.CACHE_LIST
    if hasattr(cache_obj, "max_size") and hasattr(cache_obj, "_idx"):
        return CacheType.ROTATING_KVCACHE
    if hasattr(cache_obj, "keys") and hasattr(cache_obj, "values"):
        return CacheType.KVCACHE
    if hasattr(cache_obj, "cache") and isinstance(getattr(cache_obj, "cache", None), list):
        return CacheType.ARRAYS_CACHE

    return CacheType.UNKNOWN


def is_sliceable(cache_obj: Any) -> bool:
    """Check if a cache can be sliced along the sequence dimension.

    Only standard KVCache and QuantizedKVCache support slicing.
    RotatingKVCache (circular buffer) and ArraysCache (non-KV) do NOT.
    """
    ct = detect_cache_type(cache_obj)
    return ct in (CacheType.KVCACHE, CacheType.QUANTIZED_KVCACHE,
                  CacheType.BATCH_KVCACHE)


def extract_cache_state(cache_obj: Any) -> dict:
    """Extract serializable state from an MLX cache object.

    Returns dict with 'keys', 'values', 'offset', and type-specific metadata.
    """
    ct = detect_cache_type(cache_obj)

    if ct in (CacheType.KVCACHE, CacheType.BATCH_KVCACHE, CacheType.QUANTIZED_KVCACHE):
        keys, values = cache_obj.state
        return {
            "keys": keys,
            "values": values,
            "offset": getattr(cache_obj, "offset", keys.shape[2] if keys is not None else 0),
            "cache_type": ct,
        }

    elif ct == CacheType.ROTATING_KVCACHE:
        keys, values = cache_obj.state
        meta = getattr(cache_obj, "meta_state", "")
        return {
            "keys": keys,
            "values": values,
            "offset": cache_obj.offset,
            "meta_state": meta,
            "max_size": cache_obj.max_size,
            "_idx": cache_obj._idx,
            "keep": cache_obj.keep,
            "cache_type": ct,
        }

    elif ct == CacheType.ARRAYS_CACHE:
        # ArraysCache stores a list of arbitrary arrays in .cache
        inner = cache_obj
        arrays = []
        if hasattr(inner, "cache") and isinstance(inner.cache, list):
            for item in inner.cache:
                arrays.append(item)
        return {
            "arrays": arrays,
            "cache_type": ct,
        }

    elif ct == CacheType.CACHE_LIST:
        sub_states = []
        for sub_cache in cache_obj.caches:
            sub_states.append(extract_cache_state(sub_cache))
        return {
            "sub_states": sub_states,
            "cache_type": ct,
        }

    else:
        # Fallback: try to extract keys/values
        if hasattr(cache_obj, "state"):
            state = cache_obj.state
            if isinstance(state, tuple) and len(state) >= 2:
                return {"keys": state[0], "values": state[1],
                        "cache_type": CacheType.UNKNOWN}
        return {"cache_type": CacheType.UNKNOWN}


def slice_kv_at_offsets(
    keys: mx.array,
    values: mx.array,
    start: int,
    end: int,
) -> tuple[mx.array, mx.array]:
    """Slice KV tensors along the sequence dimension (axis 2).

    Args:
        keys: Shape (B, n_kv_heads, L, head_dim)
        values: Shape (B, n_kv_heads, L, head_dim)
        start: Start offset (inclusive)
        end: End offset (exclusive)

    Returns:
        (sliced_keys, sliced_values) with shape (B, n_kv_heads, end-start, head_dim)
    """
    return keys[:, :, start:end, :], values[:, :, start:end, :]


def concatenate_kv_blocks(
    kv_pairs: list[tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array]:
    """Concatenate KV block tensors along the sequence dimension.

    Args:
        kv_pairs: List of (keys, values) each with shape (B, n_kv_heads, block_size, head_dim)

    Returns:
        (concatenated_keys, concatenated_values)
    """
    keys_list = [pair[0] for pair in kv_pairs]
    values_list = [pair[1] for pair in kv_pairs]
    return mx.concatenate(keys_list, axis=2), mx.concatenate(values_list, axis=2)


def reconstruct_kvcache(
    keys: mx.array,
    values: mx.array,
    offset: int | None = None,
) -> Any:
    """Reconstruct an MLX KVCache from saved tensors.

    Critical: always set offset = keys.shape[2] (not the saved offset),
    because the cache's update_and_fetch uses offset to determine write position.

    Args:
        keys: Shape (B, n_kv_heads, L, head_dim)
        values: Shape (B, n_kv_heads, L, head_dim)
        offset: Override offset. If None, uses keys.shape[2].

    Returns:
        A KVCache instance with the tensors set.
    """
    from mlx_lm.models.cache import KVCache

    cache = KVCache()
    cache.keys = keys
    cache.values = values
    cache.offset = offset if offset is not None else keys.shape[2]
    return cache


def get_cache_seq_length(cache_obj: Any) -> int:
    """Get the number of tokens stored in a cache object."""
    ct = detect_cache_type(cache_obj)

    if ct in (CacheType.KVCACHE, CacheType.BATCH_KVCACHE, CacheType.QUANTIZED_KVCACHE):
        offset = getattr(cache_obj, "offset", 0)
        if isinstance(offset, int):
            return offset
        # BatchKVCache offset is mx.array
        if hasattr(offset, "shape"):
            return int(offset.max().item()) if offset.size > 0 else 0
        return 0

    elif ct == CacheType.ROTATING_KVCACHE:
        return cache_obj.offset

    elif ct == CacheType.ARRAYS_CACHE:
        return getattr(cache_obj, "size", lambda: 0)()

    elif ct == CacheType.CACHE_LIST:
        # Return the max across sub-caches
        if hasattr(cache_obj, "caches"):
            return max((get_cache_seq_length(c) for c in cache_obj.caches), default=0)
        return 0

    return 0


def make_prompt_cache(model: Any, max_kv_size: int | None = None) -> list:
    """Create a prompt cache list for a model.

    Delegates to mlx-lm's cache.make_prompt_cache.
    """
    from mlx_lm.models.cache import make_prompt_cache as _make

    return _make(model, max_kv_size=max_kv_size)


def merge_caches_into_batch(
    cache_list: list,
) -> list:
    """Merge per-layer caches into batch-capable versions.

    For each layer's cache, if it's a KVCache, merge into BatchKVCache.
    Otherwise keep as-is (non-batchable types handled separately).

    This is what PromptProcessingBatch does when admitting multiple requests.
    """
    from mlx_lm.models.cache import BatchKVCache, KVCache

    batch_caches = []
    for cache in cache_list:
        ct = detect_cache_type(cache)
        if ct == CacheType.KVCACHE:
            batch_caches.append(BatchKVCache.merge([cache]))
        else:
            # For non-standard caches, wrap as-is
            batch_caches.append(cache)
    return batch_caches


def extract_single_cache(
    batch_cache: Any,
    index: int,
) -> Any:
    """Extract a single request's cache from a batch cache.

    Uses the batch cache's .extract(index) method which produces
    a clean standalone cache with contiguous tensors.
    """
    if hasattr(batch_cache, "extract"):
        return batch_cache.extract(index)
    return batch_cache


def materialize_cache(cache_list: list) -> None:
    """Force evaluation of all lazy tensors in a cache list.

    Must be called on the Metal thread before extracting raw bytes
    or transferring to background I/O threads.
    """
    states = []
    for c in cache_list:
        if hasattr(c, "state"):
            s = c.state
            if isinstance(s, tuple):
                for t in s:
                    if hasattr(t, "shape"):
                        states.append(t)
        if hasattr(c, "caches"):
            for sc in c.caches:
                if hasattr(sc, "state"):
                    s = sc.state
                    if isinstance(s, tuple):
                        for t in s:
                            if hasattr(t, "shape"):
                                states.append(t)
    if states:
        mx.eval(states)
