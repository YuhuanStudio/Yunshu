# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Chunked GatedDeltaNet core for prefill chunks.

mlx-vlm's Qwen3.5 GatedDeltaNet runs the per-token step kernel over a whole prompt chunk. MLX's own
``mx.fast.gated_delta_update`` is the chunked (tensor-unit on M5) form of the same recurrence:
Qwen3.8-27B, T=4096, one layer: 6.5 -> 2.4 ms. It rounds differently from the step kernel (bf16
outputs differ by at most one ulp; the fp32 state by ~0.2% of its scale), so it is only used for
chunks of at least ``MIN_TOKENS`` tokens on the legacy path. Qualified split-invariant
prefill uses it for every length inside an explicit prompt scope, with identity padding to 64
rows. Its id is part of every APC
key (``kernel_id``). Decode, speculative verify and the round driver never take it.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar

import mlx.core as mx

logger = logging.getLogger(__name__)

MIN_TOKENS = 64
_STATE = {"installed": False, "enabled": False, "bypass": 0, "split": False}


_PREFILL: ContextVar[bool] = ContextVar("split_invariant_prefill", default=False)


@contextlib.contextmanager
def prefill_scope():
    """Mark prompt graph construction, including its final logits, never decode/verify.

    Context-local and nestable: exceptions must not leak prefill arithmetic into
    the next request on the single MLX executor.
    """
    token = _PREFILL.set(True)
    try:
        yield
    finally:
        _PREFILL.reset(token)


def in_prefill() -> bool:
    return _PREFILL.get()


def configure_split(enabled: bool) -> None:
    """Enable only after all three prefill families have qualified together."""
    _STATE["split"] = bool(enabled)


def split_enabled() -> bool:
    return bool(_STATE["split"] and _STATE["enabled"])


def install_split_forward(language_model) -> None:
    """Keep the last prompt row on the ordinary prefill forward and convolution."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    supports = getattr(language_model, "_supports_batch_invariant_decode", None)
    if not getattr(supports, "_yunshu_split_prefill", False):

        def supported():
            return not (split_enabled() and in_prefill()) and (
                supports() if callable(supports) else True
            )

        supported._yunshu_split_prefill = True
        language_model._supports_batch_invariant_decode = supported
    original = Qwen3_5GatedDeltaNet._causal_conv1d_decode
    if not getattr(original, "_yunshu_split_prefill", False):

        def conv(layer, inputs):
            if split_enabled() and in_prefill():
                return layer.conv1d(inputs)
            return original(layer, inputs)

        conv._yunshu_split_prefill = True
        Qwen3_5GatedDeltaNet._causal_conv1d_decode = conv


def chunked_padded(q, k, v, g, beta, state, mask=None):
    """Chunked recurrence with an identity suffix to complete the final 64 rows.

    Padding is confined to the recurrence, so it neither advances KV/conv caches
    nor changes RoPE positions. Zero k/v and beta, unit decay preserve the fp32
    recurrent state; the synthetic outputs are discarded.
    """
    rows = q.shape[1]
    pad = (-rows) % 64
    if pad:

        def padded(x, value=0):
            shape = list(x.shape)
            shape[1] = pad
            return mx.concatenate([x, mx.full(shape, value, dtype=x.dtype)], axis=1)

        q, k, v = padded(q), padded(k), padded(v)
        g, beta = padded(g, 1), padded(beta)
        if mask is not None:
            mask = padded(mask, False)
    out, state = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
    return out[:, :rows], state


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
    if split_enabled():
        return "gdn-chunked-pad64-v1"
    return f"gdn-chunked-ge{MIN_TOKENS}" if _STATE["enabled"] else "gdn-step"


def install() -> bool:
    """Patch ``mlx_vlm.models.qwen3_5.gated_delta.gated_delta_kernel`` (idempotent); True when active."""
    if not hasattr(mx.fast, "gated_delta_update"):
        return False
    from mlx_vlm.models.qwen3_5 import gated_delta as gd

    if not _STATE["installed"]:
        step = gd.gated_delta_kernel

        def gated_delta_kernel(q, k, v, g, beta, state, mask=None):
            if split_enabled() and in_prefill() and not _STATE["bypass"]:
                if g.ndim != 3 or state is None:
                    raise ValueError(
                        "split-invariant prefill needs scalar gates and fp32 state"
                    )
                return chunked_padded(q, k, v, g, beta, state, mask)
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
    _STATE["split"] = False
