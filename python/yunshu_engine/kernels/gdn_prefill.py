# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Chunked GatedDeltaNet core for prefill chunks.

mlx-vlm's Qwen3.5 GatedDeltaNet runs the per-token step kernel over a whole prompt chunk. MLX's own
``mx.fast.gated_delta_update`` is the chunked (tensor-unit on M5) form of the same recurrence:
Qwen3.8-27B, T=4096, one layer: 6.5 -> 2.4 ms. It rounds differently from the step kernel (bf16
outputs differ by at most one ulp; the fp32 state by ~0.2% of its scale), so it is only used for
chunks of at least ``MIN_TOKENS`` tokens when the runner enables it, and its id is part of every APC
key (``kernel_id``). Decode, speculative verify and the round driver never take it.
"""

from __future__ import annotations

import contextlib
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

MIN_TOKENS = 64
_STATE = {"installed": False, "enabled": False, "bypass": 0}


@contextlib.contextmanager
def step_kernel():
    """Run ``gated_delta_kernel`` calls inside the block on the per-token step
    kernel (the round driver's prefill: a token's bits must not depend on how
    the prompt was cut into spans)."""
    _STATE["bypass"] += 1
    try:
        yield
    finally:
        _STATE["bypass"] -= 1


def kernel_id() -> str:
    return f"gdn-chunked-ge{MIN_TOKENS}" if _STATE["enabled"] else "gdn-step"


def install() -> bool:
    """Patch ``mlx_vlm.models.qwen3_5.gated_delta.gated_delta_kernel`` (idempotent); True when active."""
    if not hasattr(mx.fast, "gated_delta_update"):
        return False
    from mlx_vlm.models.qwen3_5 import gated_delta as gd

    if not _STATE["installed"]:
        step = gd.gated_delta_kernel

        def gated_delta_kernel(q, k, v, g, beta, state, mask=None):
            if (
                _STATE["enabled"]
                and not _STATE["bypass"]
                and q.shape[1] >= MIN_TOKENS
                and g.ndim == 3
                and state is not None
            ):
                return mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
            return step(q, k, v, g, beta, state, mask)

        gd.gated_delta_kernel = gated_delta_kernel
        _STATE["installed"] = True
        logger.info(
            "GDN prefill: mx.fast.gated_delta_update for chunks >= %d", MIN_TOKENS
        )
    _STATE["enabled"] = True
    return True


def disable() -> None:
    _STATE["enabled"] = False
