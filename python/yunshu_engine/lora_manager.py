from __future__ import annotations
"""LoRA Adapter Manager for dynamic adapter loading and serving.

Provides runtime LoRA adapter management:
- Load/unload LoRA adapters on a loaded model
- Merge adapters into base weights for zero-overhead inference
- Track active adapters per model
- Enforce max_loras memory constraints (vLLM pattern)
- Reference counting for concurrent request safety

Uses mlx-lm's tuner utilities (linear_to_lora_layers, load_adapters)
for the actual LoRA weight application.

Architecture:
  LoRAAdapterManager — singleton managing adapters across engines
  LoRAAdapterEntry — per-adapter metadata + merge state + ref count
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoRAAdapterEntry:
    adapter_id: str
    adapter_path: str
    rank: int = 8
    scale: float = 20.0
    is_loaded: bool = False
    is_merged: bool = False
    estimated_bytes: int = 0
    ref_count: int = 0  # Number of active requests using this adapter


# ── Module-level singleton ──

_global_lora_manager: LoRAAdapterManager | None = None
_global_lora_lock = threading.Lock()


def get_lora_manager() -> LoRAAdapterManager | None:
    """Return the global LoRAAdapterManager singleton, or None if not initialized.

    Thread-safe: acquires _global_lora_lock for visibility guarantee matching
    set_lora_manager().
    """
    with _global_lora_lock:
        return _global_lora_manager


def set_lora_manager(mgr: LoRAAdapterManager | None) -> None:
    """Set the global LoRAAdapterManager singleton."""
    global _global_lora_manager
    with _global_lora_lock:
        _global_lora_manager = mgr


class LoRAAdapterManager:
    """Manages LoRA adapters for a single loaded model.

    Follows the vLLM pattern:
    - max_loras: maximum number of simultaneously loaded adapters
    - When limit exceeded, unload least-recently-used adapters
    - Merge option: permanently merge adapter weights into base model
    - Reference counting: in-use adapters cannot be evicted

    Integration with BatchedEngine:
    - Engine calls acquire_adapter() before generation (increments ref count)
    - Engine calls release_adapter() after generation (decrements ref count)
    - Or engine calls merge_adapter() for zero-overhead serving
    """

    def __init__(self, max_loras: int = 4) -> None:
        self.max_loras = max_loras
        self._adapters: dict[str, LoRAAdapterEntry] = {}
        self._lru_order: list[str] = []  # most recent at end
        self._base_model = None
        self._base_model_copy = None  # saved before any merge
        self._lock = threading.RLock()  # RLock to avoid deadlock with _loaded_adapters property
        self._gpu_lock = threading.RLock()  # serializes GPU work (apply/restore); RLock for reentrant LRU eviction
        self._active_adapter_id: str | None = None  # Currently applied adapter

    def set_base_model(self, model) -> None:
        """Set the base model for LoRA operations.

        Thread-safe: acquires _lock to prevent races with concurrent
        load_adapter / unload_adapter calls that read _base_model.
        """
        with self._lock:
            self._base_model = model

    def shutdown(self) -> None:
        """Release all adapters and saved weights on engine shutdown."""
        with self._lock:
            for entry in self._adapters.values():
                entry.is_loaded = False
                entry.is_merged = False
                entry.ref_count = 0
            self._adapters.clear()
            self._lru_order.clear()
            self._base_model_copy = None
            self._base_model = None
            self._active_adapter_id = None
        # Force GC and clear Metal buffer pool to release GPU memory
        # held by the dropped model weight references.
        try:
            import gc
            import mlx.core as mx
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        except Exception:
            logger.debug("post-shutdown GC/cache clear failed", exc_info=True)
        logger.info("LoRA manager shut down, all adapters and base model released")

    def save_base_weights(self) -> None:
        """Save a deep copy of base model weights before merging adapters.

        IMPORTANT: Must be called BEFORE any LoRALinear layers are applied.
        If called after, the saved copy would include LoRA parameters, and
        future restores would re-inject stale LoRA weights into the model.
        This method is idempotent — only the first call actually saves.
        """
        if self._base_model is not None and self._base_model_copy is None:
            import mlx.core as mx
            self._base_model_copy = mx.tree_map(lambda x: mx.array(x), self._base_model.parameters())

    def register_adapter(
        self,
        adapter_id: str,
        adapter_path: str,
        estimated_bytes: int = 0,
    ) -> None:
        """Register a LoRA adapter without loading it.

        Reads rank and scale from adapter_config.json (if present) so
        that list_adapters()/get_stats() show accurate metadata before
        the adapter is actually loaded.  Missing or unreadable config
        falls back to defaults (rank=8, scale=20.0).
        """
        rank = 8
        scale = 20.0
        config_path = Path(adapter_path) / "adapter_config.json"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                lora_params = config.get("lora_parameters", {})
                rank = lora_params.get("rank", config.get("r", rank))
                explicit_scale = lora_params.get("scale", None)
                if explicit_scale is not None:
                    scale = explicit_scale
                else:
                    alpha = lora_params.get("alpha", config.get("lora_alpha", rank))
                    if rank > 0:
                        scale = alpha / rank
        except Exception:
            logger.debug(
                "Could not read adapter config for %s, using defaults",
                adapter_id, exc_info=True,
            )

        with self._lock:
            if adapter_id in self._adapters:
                return
            self._adapters[adapter_id] = LoRAAdapterEntry(
                adapter_id=adapter_id,
                adapter_path=adapter_path,
                rank=rank,
                scale=scale,
                estimated_bytes=estimated_bytes,
            )
        logger.info(f"Registered LoRA adapter: {adapter_id} ({adapter_path})")

    def load_adapter(self, adapter_id: str) -> bool:
        """Load and apply a LoRA adapter to the base model.

        Returns True if adapter was loaded successfully.
        If max_loras exceeded, unloads least-recently-used adapters
        that have zero references (not actively serving requests).
        Only one non-merged adapter can be active at a time (base model
        weights are shared). If a different adapter is loaded, it is
        unloaded first (only if it has zero references).

        Thread safety: _lock is held throughout to prevent TOCTOU races
        with concurrent unload_adapter calls. _gpu_lock serializes GPU
        work (apply/restore) so unload waits until apply completes.
        """
        with self._lock:
            if adapter_id not in self._adapters:
                logger.error(f"LoRA adapter not registered: {adapter_id}")
                return False

            entry = self._adapters[adapter_id]
            if entry.is_loaded and not entry.is_merged:
                # Adapter already loaded — just update LRU
                self._touch(adapter_id)
                return True

            if not self._base_model:
                logger.error("No base model set for LoRA adapter loading")
                return False

            # If a different adapter is currently active, must switch.
            # Only allow switching if the active adapter has no in-flight refs.
            if (self._active_adapter_id is not None
                    and self._active_adapter_id != adapter_id
                    and not entry.is_merged):
                active_entry = self._adapters.get(self._active_adapter_id)
                if active_entry and active_entry.ref_count > 0:
                    logger.error(
                        f"Cannot load adapter {adapter_id}: adapter "
                        f"{self._active_adapter_id} has {active_entry.ref_count} "
                        f"active requests"
                    )
                    return False

            # Enforce max_loras limit — unload LRU adapters with zero refs
            while len(self._loaded_adapters) >= self.max_loras:
                if not self._unload_lru_unlocked():
                    # Could not evict any adapter (all in use)
                    logger.error(
                        f"Cannot load adapter {adapter_id}: max_loras={self.max_loras} "
                        f"reached and all adapters are in use"
                    )
                    return False

            # CRITICAL: restore base model BEFORE applying new adapter to
            # prevent double-wrap (LoRALinear wrapping LoRALinear).
            # The active adapter was already verified to have zero refs above.
            if self._active_adapter_id is not None and not entry.is_merged:
                with self._gpu_lock:
                    self._restore_base()
                old_active = self._adapters.get(self._active_adapter_id)
                if old_active and old_active is not entry:
                    old_active.is_loaded = False
                    if old_active.adapter_id in self._lru_order:
                        self._lru_order.remove(old_active.adapter_id)
                self._active_adapter_id = None

            # Mark as loading to prevent concurrent load of same adapter
            entry.is_loaded = True  # tentative — will be reverted on failure

            # CRITICAL: Save base weights BEFORE applying any LoRA adapter.
            # If save_base_weights() is only called in merge_adapter(), and an
            # adapter was already loaded via load_adapter() first, then the saved
            # "base" weights would include LoRA parameters (lora_a, lora_b).
            # Later _restore_base() would re-inject stale LoRA params into the
            # structurally-unwrapped model, corrupting output quality.
            # Idempotent — only the first call actually saves.
            self.save_base_weights()

            # Apply adapter under gpu_lock while still holding _lock.
            # _lock is an RLock so reentrant acquisition is safe.
            # This prevents unload_adapter from seeing is_loaded=True and
            # starting a restore while we're still applying.
            with self._gpu_lock:
                try:
                    self._apply_adapter(entry)
                    self._touch(adapter_id)
                    self._active_adapter_id = adapter_id
                    logger.info(f"Loaded LoRA adapter: {adapter_id}")
                    return True
                except Exception as e:
                    entry.is_loaded = False
                    self._active_adapter_id = None
                    logger.error(f"Failed to load LoRA adapter {adapter_id}: {e}", exc_info=True)
                    return False

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload a LoRA adapter, restoring base model weights.

        Raises RuntimeError if the adapter has active references.
        """
        with self._lock:
            if adapter_id not in self._adapters:
                return False
            entry = self._adapters[adapter_id]
            if not entry.is_loaded or entry.is_merged:
                return False
            if entry.ref_count > 0:
                logger.error(
                    f"Cannot unload adapter {adapter_id}: "
                    f"{entry.ref_count} active requests still using it"
                )
                return False

            # Mark as unloading to prevent concurrent use
            entry.is_loaded = False
            if adapter_id in self._lru_order:
                self._lru_order.remove(adapter_id)
            if self._active_adapter_id == adapter_id:
                self._active_adapter_id = None

            # Restore under gpu_lock while still holding _lock.
            # This prevents load_adapter from seeing is_loaded=False and
            # starting an apply while we're still restoring.
            with self._gpu_lock:
                try:
                    self._restore_base()
                    logger.info(f"Unloaded LoRA adapter: {adapter_id}")
                    return True
                except Exception as e:
                    # Revert state on failure
                    entry.is_loaded = True
                    self._touch(adapter_id)
                    self._active_adapter_id = adapter_id
                    logger.error(f"Failed to unload LoRA adapter {adapter_id}: {e}", exc_info=True)
                    return False

    def acquire_adapter(self, adapter_id: str) -> bool:
        """Acquire a reference to a loaded adapter.

        Call before starting a request that uses the adapter.
        Increments ref_count to prevent eviction during generation.
        Returns True if the adapter was successfully acquired.
        """
        with self._lock:
            if adapter_id not in self._adapters:
                logger.error(f"LoRA adapter not registered: {adapter_id}")
                return False

            entry = self._adapters[adapter_id]
            if not entry.is_loaded:
                # Try to load it first
                if not self.load_adapter(adapter_id):
                    return False
                # Re-fetch entry after load
                entry = self._adapters[adapter_id]

            entry.ref_count += 1
            self._touch(adapter_id)
            logger.debug(f"Acquired adapter {adapter_id} (refs={entry.ref_count})")
            return True

    def release_adapter(self, adapter_id: str) -> None:
        """Release a reference to a loaded adapter.

        Call after finishing a request that used the adapter.
        Decrements ref_count. Does NOT unload the adapter —
        that happens via LRU eviction or explicit unload_adapter().
        """
        with self._lock:
            if adapter_id not in self._adapters:
                logger.warning(f"Release called for unknown adapter: {adapter_id}")
                return
            entry = self._adapters[adapter_id]
            if entry.ref_count <= 0:
                logger.warning(
                    f"Release called for adapter {adapter_id} with ref_count={entry.ref_count}"
                )
                return
            entry.ref_count -= 1
            logger.debug(f"Released adapter {adapter_id} (refs={entry.ref_count})")

    def merge_adapter(self, adapter_id: str) -> bool:
        """Merge LoRA weights permanently into base model.

        After merging, the adapter cannot be unloaded individually.
        The merged model has zero LoRA inference overhead.
        Base weights are saved only once (before the first merge) to
        prevent memory leaks from repeated save_base_weights calls.

        Thread safety: _lock (RLock) is held throughout the entire
        operation, including the load_adapter() call which re-enters
        via RLock.  _gpu_lock serializes GPU work (fuse/restore).
        This eliminates the TOCTOU window where a concurrent
        unload_adapter could have cleared the adapter between the
        load and the merge.
        """
        with self._lock:
            if adapter_id not in self._adapters:
                return False
            entry = self._adapters[adapter_id]
            if entry.is_merged:
                return True  # Already merged
            needs_load = not entry.is_loaded

            # CRITICAL: Save base weights BEFORE loading any adapter.
            # If saved after load_adapter(), the copy would include LoRA
            # parameters (lora_a, lora_b) from the active LoRALinear layers.
            # Later _restore_base() would then re-inject stale LoRA params
            # into the structurally-unwrapped model, corrupting it.
            # Idempotent — only the first call actually saves.
            self.save_base_weights()

            # Load under the same RLock — load_adapter's nested
            # with self._lock re-enters safely.  This prevents the
            # TOCTOU race where _lock was previously dropped between
            # save_base_weights() and the final merge block.
            if needs_load:
                if not self.load_adapter(adapter_id):
                    return False

            with self._gpu_lock:
                try:
                    # Re-verify adapter is still loaded
                    entry = self._adapters.get(adapter_id)
                    if entry is None or not entry.is_loaded:
                        logger.warning(
                            "Adapter %s was unloaded during merge", adapter_id
                        )
                        return False

                    import mlx.nn as nn
                    from mlx.utils import tree_unflatten
                    from mlx_lm.tuner.lora import LoRALinear

                    # Use fuse() to bake LoRA delta (scale * lora_b @ lora_a) into
                    # the base weight.  Simply taking module.linear would silently
                    # drop the LoRA contribution, making the merge a no-op.
                    merged_layers = []
                    for name, module in self._base_model.named_modules():
                        if isinstance(module, LoRALinear):
                            fused = module.fuse(dequantize=False)
                            merged_layers.append((name, fused))

                    if merged_layers:
                        self._base_model.update_modules(tree_unflatten(merged_layers))

                    # Do NOT overwrite _base_model_copy here — save_base_weights()
                    # already saved the pre-merge copy. Overwriting would permanently
                    # lose the original base weights, making adapter switching impossible.

                    # Re-fetch entry — adapter may have been unregistered
                    if adapter_id not in self._adapters:
                        logger.warning(
                            "Adapter %s was unregistered during merge", adapter_id
                        )
                        return False
                    entry = self._adapters[adapter_id]
                    entry.is_merged = True
                    # The LoRA structure has been unwrapped back to nn.Linear
                    # and the weights baked into the base model.  Marking
                    # is_loaded=False prevents _restore_base from incorrectly
                    # skipping weight restoration when a merged adapter's
                    # is_merged=True would trigger the guard.
                    entry.is_loaded = False
                    # Clear _active_adapter_id so subsequent load_adapter()
                    # calls don't try to "restore" from this merged adapter.
                    if self._active_adapter_id == adapter_id:
                        self._active_adapter_id = None
                    logger.info(f"Merged LoRA adapter: {adapter_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to merge LoRA adapter {adapter_id}: {e}", exc_info=True)
                    return False

    def list_adapters(self) -> list[dict]:
        """List all registered adapters with their status."""
        with self._lock:
            result = []
            for aid, entry in self._adapters.items():
                result.append({
                    "adapter_id": aid,
                    "adapter_path": entry.adapter_path,
                    "rank": entry.rank,
                    "scale": entry.scale,
                    "is_loaded": entry.is_loaded,
                    "is_merged": entry.is_merged,
                    "estimated_bytes": entry.estimated_bytes,
                    "ref_count": entry.ref_count,
                })
            return result

    def get_stats(self) -> dict:
        with self._lock:
            loaded = [a for a in self._adapters.values() if a.is_loaded]
            return {
                "max_loras": self.max_loras,
                "registered": len(self._adapters),
                "loaded": len(loaded),
                "merged": sum(1 for a in self._adapters.values() if a.is_merged),
                "adapters": self.list_adapters(),
            }

    def discover_adapters(self, model_path: str) -> list[str]:
        """Discover LoRA adapters in the model directory or adapters/ subdirectory."""
        discovered = []
        base = Path(model_path)

        # Check for adapters in model_path/adapters/ or model_path itself
        for search_path in [base / "adapters", base]:
            if not search_path.exists():
                continue
            for child in sorted(search_path.iterdir()):
                if child.is_dir() and (child / "adapter_config.json").exists():
                    adapter_id = child.name
                    size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                    self.register_adapter(adapter_id, str(child), estimated_bytes=size)
                    discovered.append(adapter_id)

        return discovered

    # ── Internal ──

    @property
    def _loaded_adapters(self) -> list[LoRAAdapterEntry]:
        """Return list of loaded (non-merged) adapters. Caller must hold self._lock."""
        return [a for a in self._adapters.values() if a.is_loaded and not a.is_merged]

    def _touch(self, adapter_id: str) -> None:
        """Update LRU position. Caller must hold self._lock."""
        if adapter_id in self._lru_order:
            self._lru_order.remove(adapter_id)
        self._lru_order.append(adapter_id)

    def _unload_lru(self) -> bool:
        """Unload least-recently-used adapter with zero references.

        Returns True if an adapter was evicted, False if all are in use.
        Caller must hold self._lock.
        """
        for candidate_id in list(self._lru_order):
            entry = self._adapters.get(candidate_id)
            if entry and entry.is_loaded and not entry.is_merged and entry.ref_count == 0:
                entry.is_loaded = False
                self._lru_order.remove(candidate_id)
                if self._active_adapter_id == candidate_id:
                    self._active_adapter_id = None
                # Restore base model weights under gpu_lock so the LoRA
                # layers don't remain stale on the model after eviction.
                with self._gpu_lock:
                    try:
                        self._restore_base()
                    except Exception as e:
                        # Revert state on failure so the adapter isn't lost
                        entry.is_loaded = True
                        self._touch(candidate_id)
                        self._active_adapter_id = candidate_id
                        logger.error(
                            f"Failed to restore base during LRU eviction of "
                            f"{candidate_id}: {e}", exc_info=True,
                        )
                        return False
                logger.info(f"Unloaded LoRA adapter (LRU eviction): {candidate_id}")
                return True
        # All loaded adapters have active requests
        return False

    # Alias for clarity in contexts where lock is already held
    _unload_lru_unlocked = _unload_lru

    def _apply_adapter(self, entry: LoRAAdapterEntry) -> None:
        """Apply LoRA adapter by inspecting actual adapter weight keys.

        Parses the adapters.safetensors key names to determine exactly which
        base model modules need LoRA wrappers, instead of guessing from
        num_layers. This ensures correct layer targeting regardless of which
        layers the adapter was trained on.
        """
        import mlx.nn as nn
        from mlx.utils import tree_unflatten

        adapter_path = Path(entry.adapter_path)
        config_path = adapter_path / "adapter_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Config can be in two formats:
        #  1. MLX-LM nested:  {"lora_parameters": {"rank": N, "scale": S}}
        #  2. HuggingFace flat: {"r": N, "lora_alpha": A}  where scale = alpha / rank
        # Priority: explicit "scale" > alpha/rank computation > default.
        lora_params = config.get("lora_parameters", {})
        rank = lora_params.get("rank", config.get("r", 8))
        scale = lora_params.get("scale", None)
        if scale is None:
            # HuggingFace convention: effective scale = lora_alpha / rank
            alpha = lora_params.get("alpha", config.get("lora_alpha", rank))
            if rank > 0:
                scale = alpha / rank
            else:
                logger.warning(
                    "LoRA adapter %s has rank=0, defaulting scale to 1.0",
                    entry.adapter_id,
                )
                scale = 1.0
        entry.rank = rank
        entry.scale = scale

        # Parse adapter weight keys to determine target modules
        weights_path = adapter_path / "adapters.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        from safetensors import safe_open
        sf = safe_open(str(weights_path), framework="mlx")
        all_keys = list(sf.keys())
        del sf

        lora_suffixes = {".lora_a", ".lora_b"}
        target_module_paths = set()
        for key in all_keys:
            for suffix in lora_suffixes:
                if key.endswith(suffix):
                    base_path = key[:-len(suffix)]
                    target_module_paths.add(base_path)
                    break

        if not target_module_paths:
            # Fall back to linear_to_lora_layers for backward compat
            num_layers = config.get("num_layers", 16)
            try:
                from mlx_lm.tuner.utils import linear_to_lora_layers
                linear_to_lora_layers(self._base_model, num_layers, lora_params)
            except (ImportError, Exception) as e:
                logger.warning(f"linear_to_lora_layers fallback failed: {e}")
            self._base_model.load_weights(str(weights_path), strict=False)
            return

        # Wrap only the target modules with LoRALinear
        from mlx_lm.tuner.lora import LoRALinear

        rank = entry.rank
        scale = entry.scale

        lora_wrappers = []
        for name, module in self._base_model.named_modules():
            if name in target_module_paths and isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale)
                lora_wrappers.append((name, lora_layer))

        if lora_wrappers:
            self._base_model.update_modules(tree_unflatten(lora_wrappers))

        # Load adapter weights into the wrapped model
        self._base_model.load_weights(str(weights_path), strict=False)

        wrapped_names = {name for name, _ in lora_wrappers}
        missing = target_module_paths - wrapped_names
        if missing:
            logger.warning(
                f"LoRA adapter references {len(missing)} modules not in base model"
            )
        logger.info(
            f"LoRA applied: {len(lora_wrappers)}/{len(target_module_paths)} modules "
            f"wrapped (rank={rank}, scale={scale})"
        )

    def _restore_base(self) -> None:
        """Restore base model weights and structure from saved copy.

        IMPORTANT: modifies the model IN-PLACE via update_modules() so
        that external references (engine._model, etc.) remain valid.
        Never reassigns self._base_model to a new object.
        """
        if self._base_model is None:
            return

        from mlx.utils import tree_unflatten

        # Structurally unwrap LoRALinear → nn.Linear in-place.
        # Both paths use update_modules() which mutates the existing
        # model object rather than creating a new one.
        unwrapped = []
        try:
            from mlx_lm.tuner.lora import LoRALinear
            for name, module in self._base_model.named_modules():
                if isinstance(module, LoRALinear):
                    unwrapped.append((name, module.linear))
        except ImportError:
            # LoRALinear unavailable — try remove_lora_layers as fallback
            try:
                from mlx_lm.tuner.utils import remove_lora_layers
                restored = remove_lora_layers(self._base_model)
                # remove_lora_layers returns a NEW model; graft its
                # modules back into the original to keep ext refs valid.
                for name, module in restored.named_modules():
                    unwrapped.append((name, module))
            except ImportError:
                logger.error(
                    "Cannot restore base model: neither LoRALinear nor "
                    "remove_lora_layers are available"
                )
                return

        if unwrapped:
            self._base_model.update_modules(tree_unflatten(unwrapped))

        # Only restore pre-merge weights if there are no merged adapters at all.
        # Merged adapters have is_loaded=False (structure unwrapped during merge),
        # so we must check ALL adapters, not just loaded ones. Restoring base
        # weights when a merge is active would undo the merged adapter's weights.
        if self._base_model_copy is not None and not any(
            e.is_merged for e in self._adapters.values()
        ):
            import mlx.core as mx
            self._base_model.update(self._base_model_copy)
            mx.eval(self._base_model.parameters())
