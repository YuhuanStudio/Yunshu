"""Process-level memory enforcer — oMLX pattern.

Background asyncio task that polls mx.get_active_memory() and evicts
LRU models from ModelManager when the process memory limit is exceeded.

Adapted from oMLX's ProcessMemoryEnforcer but simplified for Yunshu's
async-native architecture (no threading pool, single asyncio loop).
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


def _fmt_gb(b: int) -> str:
    return f"{b / 1024**3:.1f}GB"


class ProcessMemoryEnforcer:
    """Background task that enforces process-level memory limits.

    Polls mx.get_active_memory() at a configurable interval and evicts
    LRU models from ModelManager when the limit is exceeded.

    oMLX pattern: handles three scenarios:
    1. Multiple models loaded: evict LRU (idle) model
    2. Single model: abort all requests, keep model loaded (frees KV cache)
    3. Model loading: request abort of in-progress load
    """

    def __init__(
        self,
        model_manager: ModelManager,
        max_bytes: int,
        poll_interval: float = 1.0,
        ttl_seconds: float | None = None,
    ):
        self._manager = model_manager
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._ttl_seconds = ttl_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        self._max_bytes = value

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(limit: {_fmt_gb(self._max_bytes)}, interval: {self._poll_interval}s)"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _enforcement_loop(self) -> None:
        while self._running:
            try:
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_ttl(self) -> None:
        """Unload models past their TTL (oMLX TTL expiration pattern)."""
        if self._ttl_seconds is not None:
            self._manager.ttl_seconds = self._ttl_seconds
        await self._manager.check_ttl()

    async def _check_and_enforce(self) -> None:
        """Check memory and evict if over limit (oMLX pattern)."""
        if self._max_bytes <= 0:
            return

        current = mx.get_active_memory()
        if current <= self._max_bytes:
            return

        overage = current - self._max_bytes
        logger.warning(
            f"Memory limit exceeded: {_fmt_gb(current)} / {_fmt_gb(self._max_bytes)} "
            f"(+{_fmt_gb(overage)})"
        )

        # Evict LRU models until under limit
        while mx.get_active_memory() > self._max_bytes:
            victim = self._manager._find_lru_victim()
            if victim is None:
                # Check for loading models — request abort
                aborted_any = False
                for entry in self._manager._entries.values():
                    if entry.is_loading:
                        logger.warning(f"Aborting load of '{entry.model_id}' — memory limit")
                        # For Yunshu, is_loading is a flag — the loading coroutine
                        # should check this and abort
                        aborted_any = True
                if not aborted_any:
                    logger.warning("Memory limit exceeded but no models to evict")
                break

            loaded_non_pinned = [
                e for e in self._manager._entries.values()
                if e.is_loaded and not e.is_pinned
            ]

            if len(loaded_non_pinned) > 1:
                # Multiple models: evict LRU victim entirely
                logger.warning(f"Evicting '{victim.model_id}' to enforce memory limit")
                await self._manager.unload_model(victim.model_id)
            else:
                # Single model: abort requests, keep loaded (frees KV cache)
                if victim.engine and hasattr(victim.engine, 'has_active_requests'):
                    if victim.engine.has_active_requests():
                        logger.warning(
                            f"Aborting active requests on '{victim.model_id}' "
                            f"due to memory pressure (model kept loaded)"
                        )
                        # Signal abort
                        if hasattr(victim.engine, '_abort_set'):
                            for rid in list(victim.engine._active.keys()):
                                victim.engine._abort_set.add(rid)
                break

        # Force GC + cache clear after eviction (oMLX pattern: on MLX executor)
        gc.collect()
        try:
            from .mlx_executor import sync_and_clear_cache
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._manager._get_mlx_executor(),
                sync_and_clear_cache,
            )
        except Exception:
            pass

    def get_status(self) -> dict:
        current = mx.get_active_memory() if self._running else 0
        return {
            "enabled": self._running,
            "max_bytes": self._max_bytes,
            "max_gb": round(self._max_bytes / 1024**3, 1),
            "current_bytes": current,
            "current_gb": round(current / 1024**3, 1),
            "utilization": current / self._max_bytes if self._max_bytes > 0 else 0.0,
        }
