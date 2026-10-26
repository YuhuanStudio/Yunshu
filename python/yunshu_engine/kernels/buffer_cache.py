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
# Storage only: checkpoints always overwrite every row before using these arrays.
_RESTORE_BUFFERS: list = []
# A pooled set nobody takes (every later request is under half its capacity) is
# released after this many finished requests, so it cannot pin memory forever.
MAX_IDLE_OFFERS = 3


def restore_buffer_bytes() -> int:
    return sum(a.nbytes for a in _RESTORE_BUFFERS)


def _restore_pool_limit() -> None:
    mx.set_cache_limit(max(0, _STATE["limit"] - restore_buffer_bytes()))


def clear_restore_buffers() -> None:
    _RESTORE_BUFFERS.clear()
    _restore_pool_limit()


def _age_pool() -> None:
    """One finished request passed: release a pooled set that none of them took."""
    if not _RESTORE_BUFFERS:
        return
    if _STATE.get("pool_used"):
        _STATE["idle_offers"] = 0
    else:
        _STATE["idle_offers"] = _STATE.get("idle_offers", 0) + 1
    _STATE["pool_used"] = False
    if _STATE["idle_offers"] >= MAX_IDLE_OFFERS:
        clear_restore_buffers()


def offer_restore_buffers(caches) -> None:
    """Retain at most one finished KV set within the existing allocator budget.

    Keep the larger set across short re-sends. Detach the arrays from the finished
    cache objects so the next overwrite can donate their physical buffers.
    """
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    _age_pool()
    candidates = [c for c in caches if type(c) in (KVCache, BatchKVCache)]
    arrays = [a for c in candidates for a in (c.keys, c.values)]
    if not arrays or any(a is None for a in arrays):
        return
    size = sum(a.nbytes for a in arrays)
    if not restore_buffer_bytes() < size <= _STATE["limit"]:
        return
    _RESTORE_BUFFERS[:] = arrays
    _STATE["idle_offers"] = 0
    _STATE["pool_used"] = False
    for c in candidates:
        c.keys = c.values = None
    _restore_pool_limit()
    # MLX enforces a reduced cache limit on the next allocation, not at
    # set_cache_limit. Reclaim now so idle storage + allocator stay bounded.
    mx.eval(mx.empty((1,), dtype=mx.uint8))


def take_restore_buffer(a, capacity):
    """A sole-owner destination with identical layout and sufficient capacity.

    Best fit, at most twice the requested capacity: a small request must not pin (and
    zero-fill) a buffer sized for a much longer finished prompt."""
    best = None
    for i, dst in enumerate(_RESTORE_BUFFERS):
        if (
            dst.dtype == a.dtype
            and dst.shape[:2] == a.shape[:2]
            and dst.shape[3:] == a.shape[3:]
            and capacity <= dst.shape[2] <= 2 * capacity
            and (best is None or dst.shape[2] < _RESTORE_BUFFERS[best].shape[2])
        ):
            best = i
    if best is None:
        return None
    out = _RESTORE_BUFFERS.pop(best)
    _STATE["pool_used"] = True
    _restore_pool_limit()
    return out


def auto_limit_gib(total_bytes: int) -> float:
    """A reusable allocator pool scales with RAM (128 GiB: 6, 8 GiB: 0.4)."""
    return min(6.0, max(0.0, total_bytes / (1 << 30) * 0.05))


def auto_limit_gib_for_weights(
    total_bytes: int,
    weight_bytes: int | None,
    limit_bytes: int | None = None,
) -> float:
    """The pool limit when resident weights leave little headroom: a 0.5 GiB pool instead of the RAM-scaled one.

    Headroom is measured against the user's memory ceiling when one is set (``YUNSHU_MAX_MEMORY_GB``: weights
    above 80% of it), otherwise against physical RAM (weights above 45% of it; Qwen3.8-Flash-Next oQ4e is 74 GB
    of 137 GB).  The pool only holds freed buffers for reuse, so output is unchanged; on that pack the 8K
    cold-prefill peak fell from 83.5-89 GB to 78 GB at equal decode speed."""
    base = auto_limit_gib(total_bytes)
    if not weight_bytes or not total_bytes:
        return base
    tight = (
        weight_bytes > 0.8 * limit_bytes
        if limit_bytes
        else weight_bytes > 0.45 * total_bytes
    )
    return min(base, 0.5) if tight else base


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

    _RESTORE_BUFFERS.clear()
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
