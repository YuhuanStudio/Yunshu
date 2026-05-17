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
import os
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
    """Return the global LoRAAdapterManager singleton, or None if not initialized."""
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
        self._gpu_lock = threading.Lock()  # serializes GPU work (apply/restore)
        self._active_adapter_id: str | None = None  # Currently applied adapter

    def set_base_model(self, model) -> None:
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
        logger.info("LoRA manager shut down, all adapters and base model released")

    def save_base_weights(self) -> None:
        """Save a copy of base model weights before merging adapters."""
        if self._base_model is not None and self._base_model_copy is None:
            import mlx.core as mx
            self._base_model_copy = mx.tree_map(lambda x: x, self._base_model.parameters())

    def register_adapter(
        self,
        adapter_id: str,
        adapter_path: str,
        estimated_bytes: int = 0,
    ) -> None:
        """Register a LoRA adapter without loading it."""
        with self._lock:
            if adapter_id in self._adapters:
                return
            self._adapters[adapter_id] = LoRAAdapterEntry(
                adapter_id=adapter_id,
                adapter_path=adapter_path,
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

            # Mark as loading to prevent concurrent load of same adapter
            entry.is_loaded = True  # tentative — will be reverted on failure

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
        """
        with self._lock:
            if adapter_id not in self._adapters:
                return False
            entry = self._adapters[adapter_id]
            if entry.is_merged:
                return True  # Already merged
            needs_load = not entry.is_loaded

        # Load outside lock to avoid holding _lock during GPU work;
        # load_adapter takes its own _lock (RLock, reentrant-safe).
        if needs_load:
            if not self.load_adapter(adapter_id):
                return False

        # Save base weights once (idempotent — only saves if not already saved)
        self.save_base_weights()

        try:
            import mlx.nn as nn
            from mlx.utils import tree_flatten, tree_unflatten
            from mlx_lm.tuner.lora import LoRALinear

            merged_layers = []
            for name, module in self._base_model.named_modules():
                if isinstance(module, LoRALinear):
                    merged_layers.append((name, module.linear))

            if merged_layers:
                self._base_model.update_modules(tree_unflatten(merged_layers))

            with self._lock:
                entry.is_merged = True
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
        """Apply LoRA adapter to base model using mlx-lm tuner utilities."""
        adapter_path = Path(entry.adapter_path)
        config_path = adapter_path / "adapter_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")

        with open(config_path) as f:
            config = json.load(f)

        lora_params = config.get("lora_parameters", {})
        entry.rank = lora_params.get("rank", 8)
        entry.scale = lora_params.get("scale", 20.0)
        num_layers = config.get("num_layers", 16)

        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
            linear_to_lora_layers(
                self._base_model,
                num_layers,
                lora_params,
            )
        except ImportError:
            # Fallback: manual LoRA application
            logger.warning("mlx_lm.tuner.utils not available, attempting manual LoRA load")
            self._apply_lora_manual(entry, lora_params, num_layers)

        # Load adapter weights
        weights_path = adapter_path / "adapters.safetensors"
        if weights_path.exists():
            self._base_model.load_weights(str(weights_path), strict=False)

    def _apply_lora_manual(self, entry: LoRAAdapterEntry, lora_params: dict, num_layers: int) -> None:
        """Fallback manual LoRA layer application when tuner utils unavailable."""
        import mlx.nn as nn
        from mlx.utils import tree_unflatten

        rank = lora_params.get("rank", 8)
        scale = lora_params.get("scale", 20.0)

        # Apply LoRA to attention layers only
        from mlx_lm.tuner.lora import LoRALinear

        lora_layers = []
        for name, module in self._base_model.named_modules():
            if isinstance(module, nn.Linear) and num_layers > 0:
                # Apply LoRA to Q and V projection layers
                if any(k in name for k in ("q_proj", "v_proj", "query", "value")):
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        scale=scale,
                    )
                    lora_layer.linear = module
                    lora_layers.append((name, lora_layer))
                    num_layers -= 1

        # Actually update the model with the new LoRA layers
        if lora_layers:
            self._base_model.update_modules(tree_unflatten(lora_layers))

    def _restore_base(self) -> None:
        """Restore base model weights from saved copy."""
        if self._base_model is None:
            return

        if self._base_model_copy is not None:
            import mlx.core as mx
            # Restore original weights
            self._base_model.update(self._base_model_copy)
            mx.eval(self._base_model.parameters())
        else:
            # No copy saved — try to remove LoRA layers
            try:
                from mlx_lm.tuner.utils import remove_lora_layers
                self._base_model = remove_lora_layers(self._base_model)
            except ImportError:
                logger.warning("Cannot restore base model — no copy and tuner utils unavailable")
