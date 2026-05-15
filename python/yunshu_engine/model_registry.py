from __future__ import annotations
"""Model Registry — tracks model ownership across engines (oMLX pattern).

Prevents BatchKVCache conflicts when multiple EngineCore instances
share a model. mlx-lm's BatchGenerator maintains internal KV cache
state tied to the model — multiple engines using the same model
causes incompatible cache objects and NoneType errors.

Uses weak references for automatic cleanup on GC.
"""

import logging
import threading
import weakref
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelOwnershipError(Exception):
    pass


class ModelRegistry:
    """Global registry tracking model ownership via weak references."""

    _instance: Optional[ModelRegistry] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._owners: dict[int, tuple[weakref.ref, str]] = {}
                    cls._instance._lock = threading.Lock()
        return cls._instance

    def acquire(
        self,
        model: Any,
        engine: Any,
        engine_id: str,
        force: bool = False,
    ) -> bool:
        mid = id(model)
        with self._lock:
            if mid in self._owners:
                ref, owner_id = self._owners[mid]
                owner = ref()
                if owner is not None and owner_id != engine_id:
                    if force:
                        logger.warning(
                            "Model ownership transfer: %s -> %s",
                            owner_id, engine_id,
                        )
                        self._reset_owner(owner)
                    else:
                        raise ModelOwnershipError(
                            f"Model owned by engine {owner_id}. "
                            f"Use force=True or release() first."
                        )
            self._owners[mid] = (weakref.ref(engine), engine_id)
            return True

    def release(self, model: Any, engine_id: str) -> bool:
        mid = id(model)
        with self._lock:
            if mid in self._owners:
                _, owner_id = self._owners[mid]
                if owner_id == engine_id:
                    del self._owners[mid]
                    return True
        return False

    def is_owned(self, model: Any) -> tuple[bool, Optional[str]]:
        mid = id(model)
        with self._lock:
            if mid in self._owners:
                ref, owner_id = self._owners[mid]
                if ref() is not None:
                    return (True, owner_id)
                del self._owners[mid]
        return (False, None)

    def _reset_owner(self, owner: Any) -> None:
        try:
            if hasattr(owner, "scheduler") and hasattr(owner.scheduler, "deep_reset"):
                owner.scheduler.deep_reset()
        except Exception as e:
            logger.warning("Failed to reset previous owner: %s", e)

    def cleanup(self) -> int:
        with self._lock:
            stale = [
                mid for mid, (ref, _) in self._owners.items()
                if ref() is None
            ]
            for mid in stale:
                del self._owners[mid]
        return len(stale)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            active = sum(1 for _, (ref, _) in self._owners.items() if ref() is not None)
            return {
                "total_entries": len(self._owners),
                "active_owners": active,
            }


_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
