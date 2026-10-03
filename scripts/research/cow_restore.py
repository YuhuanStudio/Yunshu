"""Research-only native restore handles, reserving capacity in one copy.

Stores still detach. Restore arrays get independent Python/MLX handles and rely
on MLX copy-on-write until growth; a capacity reservation then copies directly
from the stored prefix without an intermediate full-prefix clone.
"""


def install():
    from mlx_vlm import apc_adapters
    from yunshu_engine.kernels.cache_restore import clone_native_restore

    original = apc_adapters.clone_cache_entry
    counts = {"enabled": True, "view_restores": 0}

    def clone(c, *, min_capacity_tokens, eval_targets):
        restored = (
            clone_native_restore(
                c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
            if counts["enabled"]
            else None
        )
        if restored is not None:
            counts["view_restores"] += 1
            return restored
        return original(
            c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
        )

    apc_adapters.clone_cache_entry = clone

    def uninstall():
        apc_adapters.clone_cache_entry = original

    return counts, uninstall
