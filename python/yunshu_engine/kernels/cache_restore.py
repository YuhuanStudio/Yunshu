# Patches upstream mlx-vlm symbols (registered in vendor.json).
"""Restore native APC state without an intermediate detached prefix copy.

Independent MLX handles preserve copy-on-write ownership. Capacity reservation
copies directly from the stored prefix; stores and custom adapters still use
upstream's detached snapshot contract.
"""

from typing import Any


def clone_native_restore(
    cache: Any, *, min_capacity_tokens: int | None, eval_targets: list[Any]
) -> Any | None:
    import mlx.core as mx
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import ArraysCache, KVCache

    if min_capacity_tokens is None or type(cache) not in (KVCache, ArraysCache):
        return None

    def own(tree: Any) -> Any:
        if isinstance(tree, mx.array):
            return mx.contiguous(tree.view(tree.dtype))
        if isinstance(tree, list):
            return [own(value) for value in tree]
        if isinstance(tree, tuple):
            return tuple(own(value) for value in tree)
        if isinstance(tree, dict):
            return {key: own(value) for key, value in tree.items()}
        return tree

    if type(cache) is KVCache:
        restored = KVCache.from_state(own(cache.state), cache.meta_state)
    else:
        restored = ArraysCache(len(cache.cache))
        restored.prefix_cache_restore(own(cache.prefix_cache_snapshot()))
    apc_adapters._eval_tree(restored.state, eval_targets)
    apc_adapters.reserve_checkpoint_capacity(
        restored,
        min_capacity_tokens=min_capacity_tokens,
        eval_targets=eval_targets,
    )
    return restored


def extract_single_row(cache: Any) -> Any | None:
    """``BatchKVCache.extract(0)`` for a one-row cache without a GPU round trip per layer.

    Upstream reads ``left_padding[idx].item()``: slicing builds a graph node, so every full-attention
    layer commits a command buffer and waits (about 0.25 ms each, 4 ms for the 27B's 16 layers) on the
    first-token path. ``left_padding`` itself is already computed, so reading it as a list is free.
    The arrays produced are the same slices.
    """
    import mlx.core as mx
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    if (
        type(cache) is not BatchKVCache
        or not cache.is_single_row()
        or cache.empty()
        or cache.keys is None
    ):
        return None
    padding = int(cache.left_padding.tolist()[0])
    out = KVCache()
    out.keys = mx.contiguous(cache.keys[0:1, :, padding : cache._idx])
    out.values = mx.contiguous(cache.values[0:1, :, padding : cache._idx])
    out.offset = out.keys.shape[2]
    return out


def install() -> None:
    """Install once; only exact native cache restores use view handles."""
    from mlx_vlm import apc_adapters

    original = apc_adapters.clone_cache_entry
    if getattr(original, "_yunshu_restore_views", False):
        return

    def clone(cache: Any, *, min_capacity_tokens: int | None, eval_targets: list[Any]):
        row = extract_single_row(cache)
        if row is not None:
            cache = row
        restored = clone_native_restore(
            cache, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
        )
        if restored is not None:
            return restored
        return original(
            cache, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
        )

    clone._yunshu_restore_views = True  # type: ignore[attr-defined]
    clone._yunshu_restore_original = original  # type: ignore[attr-defined]
    apc_adapters.clone_cache_entry = clone
