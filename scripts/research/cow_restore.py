"""Research-only native restore handles, reserving capacity in one copy.

Stores still detach. Restore arrays get independent Python/MLX handles and rely
on MLX copy-on-write until growth; a capacity reservation then copies directly
from the stored prefix without an intermediate full-prefix clone.
"""


def install():
    import mlx.core as mx
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import ArraysCache, KVCache

    original = apc_adapters.clone_cache_entry
    counts = {"enabled": True, "view_restores": 0}

    def own(tree):
        if isinstance(tree, mx.array):
            return mx.contiguous(tree.view(tree.dtype))
        if isinstance(tree, list):
            return [own(value) for value in tree]
        if isinstance(tree, tuple):
            return tuple(own(value) for value in tree)
        if isinstance(tree, dict):
            return {key: own(value) for key, value in tree.items()}
        return tree

    def clone(c, *, min_capacity_tokens, eval_targets):
        if (
            not counts["enabled"]
            or min_capacity_tokens is None
            or type(c) not in (KVCache, ArraysCache)
        ):
            return original(
                c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
        if type(c) is KVCache:
            out = KVCache.from_state(own(c.state), c.meta_state)
        else:
            out = ArraysCache(len(c.cache))
            out.prefix_cache_restore(own(c.prefix_cache_snapshot()))
        apc_adapters._eval_tree(out.state, eval_targets)
        apc_adapters.reserve_checkpoint_capacity(
            out,
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=eval_targets,
        )
        counts["view_restores"] += 1
        return out

    apc_adapters.clone_cache_entry = clone

    def uninstall():
        apc_adapters.clone_cache_entry = original

    return counts, uninstall
