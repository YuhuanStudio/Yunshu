# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Keep a bounded MLX allocator pool across prompt chunks and requests.

A whole-pool clear after a temporary overflow discards the long-prefix restore
buffers too. Set MLX's cache limit instead: its next allocation reclaims excess
freed memory while retaining reusable buffers. Freed buffers can transiently
exceed the limit until that allocation. Zero retains upstream boundary clears.
"""

from __future__ import annotations

import logging
import types

import mlx.core as mx

logger = logging.getLogger(__name__)

_STATE = {"limit": 0}


def auto_limit_gib(total_bytes: int) -> float:
    """A reusable allocator pool scales with RAM (128 GiB: 6, 8 GiB: 0.4)."""
    return min(6.0, max(0.0, total_bytes / (1 << 30) * 0.05))


def clear_if_over() -> None:
    """Boundary compatibility hook: the allocator reclaims a positive pool."""
    limit = _STATE["limit"]
    if limit <= 0:
        mx.clear_cache()


class _MxView(types.ModuleType):
    """``mlx.core`` with a bounded ``clear_cache`` (everything else forwarded)."""

    def clear_cache(self) -> None:
        clear_if_over()

    def __getattr__(self, name: str):
        return getattr(mx, name)


def install(limit_gib: float) -> bool:
    """Bound allocator reuse; 0 restores upstream's clear at each boundary."""
    from mlx_vlm.generate import ar

    limit = int(max(0.0, limit_gib) * (1 << 30))
    if limit > 0:
        previous = mx.set_cache_limit(limit)
        _STATE.setdefault("previous_cache_limit", previous)
    elif "previous_cache_limit" in _STATE:
        mx.set_cache_limit(_STATE.pop("previous_cache_limit"))
    _STATE["limit"] = limit
    if not isinstance(ar.mx, _MxView):
        view = _MxView("mlx.core")
        ar.mx = view
        logger.info("prefill buffer cache kept up to %.1f GiB", limit_gib)
    return _STATE["limit"] > 0
