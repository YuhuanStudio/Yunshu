# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Keep MLX's buffer cache across prefill steps.

mlx-vlm's ``BatchGenerator`` calls ``mx.clear_cache()`` after every prompt chunk. The next large
allocation then gets fresh pages from the OS: cloning a 32K-token Qwen3.8-27B cache (APC restore, each
checkpoint store) costs 54 ms with the cache cleared and 9 ms with it warm. Here ``clear_cache`` in the
generator module only clears once the cache holds more than ``limit_gib``, so the buffers a chunk freed
are reused by the next chunk and by the clones. Output is untouched (allocator behavior only).
"""

from __future__ import annotations

import logging
import types

import mlx.core as mx

logger = logging.getLogger(__name__)

_STATE = {"limit": 0}


def _clear_cache() -> None:
    limit = _STATE["limit"]
    if limit <= 0 or mx.get_cache_memory() > limit:
        mx.clear_cache()


class _MxView(types.ModuleType):
    """``mlx.core`` with a bounded ``clear_cache`` (everything else forwarded)."""

    def __getattr__(self, name: str):
        return getattr(mx, name)


def install(limit_gib: float) -> bool:
    """Bound the generator's ``mx.clear_cache()`` calls; 0 restores upstream's clear-every-time."""
    from mlx_vlm.generate import ar

    _STATE["limit"] = int(max(0.0, limit_gib) * (1 << 30))
    if not isinstance(ar.mx, _MxView):
        view = _MxView("mlx.core")
        view.clear_cache = _clear_cache
        ar.mx = view
        logger.info("prefill buffer cache kept up to %.1f GiB", limit_gib)
    return _STATE["limit"] > 0
