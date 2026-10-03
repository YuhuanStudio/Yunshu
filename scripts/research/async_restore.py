"""Research-only overlap: submit native restore before suffix graph construction.

No arithmetic changes. Only native RAM checkpoint restores are asynchronous;
checkpoint stores and custom contracts keep upstream synchronous publication.
"""


def install():
    import mlx.core as mx
    from mlx_vlm import apc, apc_adapters
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    from yunshu_engine.apc_manager import _Coordinator, _single_native_arrays_row

    original_clone = apc._clone_prompt_cache_for_apc
    original_merge = _Coordinator.merge_rows
    counts = {"enabled": True, "async_clones": 0, "async_merges": 0}

    def clone(caches, *, min_capacity_tokens=None):
        if (
            not counts["enabled"]
            or min_capacity_tokens is None
            or not caches
            or not all(type(c) in (ArraysCache, KVCache) for c in caches)
        ):
            return original_clone(caches, min_capacity_tokens=min_capacity_tokens)
        targets = []
        out = [
            apc_adapters.clone_cache_entry(
                c, min_capacity_tokens=min_capacity_tokens, eval_targets=targets
            )
            for c in caches
        ]
        if any(c is None for c in out):
            return None
        mx.async_eval(targets)
        counts["async_clones"] += 1
        return out

    def merge(self, picks, prefix_lens, *, kv_quant_config=None):
        hit = picks[0] if len(picks) == 1 else None
        rows = hit.get("warm_cache") if hit else None
        prefix = int(prefix_lens[0]) if len(prefix_lens) == 1 else 0
        lengths = self.manager.memory_plan.lengths
        if not (
            counts["enabled"]
            and self.enabled
            and self.is_checkpoint
            and kv_quant_config is None
            and prefix > 0
            and rows
            and len(lengths) == 1
            and lengths[0] - prefix >= 64
            and all(
                type(c) is ArraysCache
                or (
                    type(c) is KVCache
                    and c.offset == prefix
                    and c.keys is not None
                    and c.values is not None
                    and c.keys.shape[0] == 1
                )
                for c in rows
            )
        ):
            return original_merge(
                self, picks, prefix_lens, kv_quant_config=kv_quant_config
            )
        merged = []
        for c in rows:
            if type(c) is KVCache:
                batch = BatchKVCache([0])
                batch.keys = c.keys.view(c.keys.dtype)
                batch.values = c.values.view(c.values.dtype)
                batch._idx = prefix
                batch.offset += prefix
                merged.append(batch)
            else:
                row = _single_native_arrays_row(c)
                if row is None:
                    return original_merge(
                        self, picks, prefix_lens, kv_quant_config=kv_quant_config
                    )
                merged.append(row)
        with self.manager.lock:
            self.manager.stats.restored_tokens += prefix
        counts["async_merges"] += 1
        return merged, prefix

    apc._clone_prompt_cache_for_apc = clone
    _Coordinator.merge_rows = merge

    def uninstall():
        apc._clone_prompt_cache_for_apc = original_clone
        _Coordinator.merge_rows = original_merge

    return counts, uninstall
