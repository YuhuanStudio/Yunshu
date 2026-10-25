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
from typing import Any

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


_TAIL: ContextVar[bool] = ContextVar("split_invariant_prompt_tail", default=False)


@contextlib.contextmanager
def tail_scope():
    """The short prompt tail after the last aligned boundary: projections and the recurrence take
    the decode/verify kernels, attention keeps the prefill kernel (the stock fallback for a few
    query rows over a long cache was ~8 ms per layer at 32K)."""
    token = _TAIL.set(True)
    try:
        yield
    finally:
        _TAIL.reset(token)


def in_tail() -> bool:
    return _TAIL.get()


def in_prefill_math() -> bool:
    """Prompt rows whose attention, recurrence and convolution keep the prefill kernels: the
    long prefill scope and the prompt tail (only the tail's projections differ). Spec on and
    off must agree there, and the decode-side paths differ between DFlash's capturing per-row
    forward and the plain batched one."""
    return _PREFILL.get() or _TAIL.get()


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
            return not (split_enabled() and in_prefill_math()) and (
                supports() if callable(supports) else True
            )

        supported._yunshu_split_prefill = True  # type: ignore[attr-defined]
        language_model._supports_batch_invariant_decode = supported
    install_fused_conv(Qwen3_5GatedDeltaNet)
    install_norm_scale(Qwen3_5GatedDeltaNet)
    original = Qwen3_5GatedDeltaNet._causal_conv1d_decode
    if not getattr(original, "_yunshu_split_prefill", False):

        def conv(layer, inputs):
            if split_enabled() and in_prefill_math():
                return layer.conv1d(inputs)
            return original(layer, inputs)

        conv._yunshu_split_prefill = True  # type: ignore[attr-defined]
        Qwen3_5GatedDeltaNet._causal_conv1d_decode = conv


# Match MLX RMSNorm's four consecutive values per SIMD lane, including the
# BF16 materialization before the scalar multiply. Never fold the scale into
# rsqrt: that would move a rounding boundary.
_NORM_SCALE_SOURCE = r"""
uint lane = thread_position_in_threadgroup.x;
uint row = thread_position_in_grid.y;
uint head = row % x_shape[2];
uint token = (row / x_shape[2]) % x_shape[1];
uint batch = row / (x_shape[2] * x_shape[1]);
size_t offset = batch * x_strides[0] + token * x_strides[1] + head * x_strides[2];
float vals[4];
float ss = 0.0f;
for (uint i=0; i<4; ++i) {
    vals[i] = float(x[offset + (lane * 4 + i) * x_strides[3]]);
    ss += vals[i] * vals[i];
}
ss = simd_sum(ss);
float inv = metal::precise::rsqrt(ss / 128.0f + 1e-6f);
for (uint i=0; i<4; ++i) {
    bfloat16_t normed = bfloat16_t(vals[i] * inv);
    out[row * 128 + lane * 4 + i] = bfloat16_t(float(normed) * float(bfloat16_t(INVERSE ? 0.0078125f : 0.08838834764831845f)));
}
"""
_NORM_SCALE_KERNEL: Any = None


def fused_norm_scale(x, *, inverse: bool):
    """128-wide BF16 RMSNorm followed by the Qwen GDN Q/K scale."""
    global _NORM_SCALE_KERNEL
    if _NORM_SCALE_KERNEL is None:
        _NORM_SCALE_KERNEL = mx.fast.metal_kernel(
            name="yunshu_gdn_norm_scale",
            input_names=["x"],
            output_names=["out"],
            source=_NORM_SCALE_SOURCE,
            ensure_row_contiguous=False,
        )
    rows = x.size // 128
    return _NORM_SCALE_KERNEL(
        inputs=[x],
        template=[("INVERSE", inverse)],
        grid=(32, rows, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def install_norm_scale(cls) -> None:
    original = cls._normalize_qk
    if getattr(original, "_yunshu_norm_scale", False):
        return

    def normalize(self, q, k):
        if (
            split_enabled()
            and in_prefill_math()
            and q.ndim == k.ndim == 4
            and q.dtype == k.dtype == mx.bfloat16
            and q.shape[-1] == k.shape[-1] == 128
            and mx.default_device() == mx.gpu
        ):
            return fused_norm_scale(q, inverse=True), fused_norm_scale(k, inverse=False)
        return original(self, q, k)

    normalize._yunshu_norm_scale = True  # type: ignore[attr-defined]
    cls._normalize_qk = normalize


_CONV_SRC = """
    uint c = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint rows = rows_[0];
    if (c >= CH || t >= rows) return;
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        int src = int(t) + j - 3;
        float v = src < 0 ? float(state[(src + 3) * CH + c]) : float(x[src * CH + c]);
        acc = acc + v * float(w[c * 4 + j]);
    }
    bfloat16_t y = bfloat16_t(acc);
    bfloat16_t ab = bfloat16_t(metal::abs(float(y)));
    bfloat16_t ex = bfloat16_t(metal::precise::exp(float(ab)));
    bfloat16_t den = bfloat16_t(1.0f + float(ex));
    bfloat16_t e = bfloat16_t(1.0f / float(den));
    bfloat16_t sg = float(y) < 0 ? e : bfloat16_t(1.0f - float(e));
    out[t * CH + c] = bfloat16_t(float(y) * float(sg));
"""
_CONV_KERNELS: dict = {}
_COUNTS = {"fused_conv": 0}


def fused_conv_calls() -> int:
    """Engagement counter: fused conv+silu launches since process start."""
    return _COUNTS["fused_conv"]


def fused_conv_silu(state, x, weight):
    """``silu(conv1d(concat([state, x], 1)))`` for B=1, kernel 4, bf16, bit-identical to MLX.

    Reads the three carried rows and the S input rows directly and applies MLX's
    bf16 sigmoid step by step (each intermediate rounded to bf16).
    """
    channels = x.shape[-1]
    rows = x.shape[1]
    kernel = _CONV_KERNELS.get(channels)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"yunshu_gdn_conv_silu_c{channels}",
            input_names=["state", "x", "w", "rows_"],
            output_names=["out"],
            source=_CONV_SRC.replace("CH", str(channels)),
        )
        _CONV_KERNELS[channels] = kernel
    if not _COUNTS["fused_conv"]:
        logger.info("GDN prefill fused conv+silu engaged: channels=%d", channels)
    _COUNTS["fused_conv"] += 1
    return kernel(
        inputs=[
            state,
            x,
            weight.reshape(channels, 4),
            mx.array([rows], dtype=mx.int32),
        ],
        grid=(channels, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, channels)],
        output_dtypes=[mx.bfloat16],
    )[0]


def install_fused_conv(cls) -> None:
    """Prefill-only fused conv+silu (no concatenate); every other call is upstream's."""
    if getattr(cls, "_yunshu_fused_conv", False):
        return
    original = cls.__call__

    def call(self, inputs, mask=None, cache=None, **kwargs):
        B, S, _ = inputs.shape
        if not (
            split_enabled()
            and in_prefill_math()
            and not kwargs
            and B == 1
            and S >= 3
            and inputs.dtype == mx.bfloat16
            and self.conv_kernel_size == 4
            and self.conv1d.weight.dtype == mx.bfloat16
            and getattr(self.conv1d, "bias", None) is None
            and not self.training
            and mx.default_device() == mx.gpu
            and (
                cache is None or (not cache.history_capacity and cache.lengths is None)
            )
        ):
            return original(self, inputs, mask=mask, cache=cache, **kwargs)
        from mlx_vlm.models.qwen3_5.language import (
            _qwen3_5_advance_lengths_info,
            gated_delta_update,
        )

        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs)
        b, a = self._project_gates(inputs)
        z = z.reshape(B, S, -1, self.head_v_dim)
        state = None
        if cache is not None and cache[0] is not None:
            state = cache[0]
            if state.shape[0] != B:
                state = None
        if state is None:
            state = mx.zeros((B, 3, self.conv_dim), dtype=inputs.dtype)
        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        if cache is not None:
            cache.update_window(0, mixed_qkv[:, S - 3 :], 3)
        conv_out = fused_conv_silu(state, mixed_qkv, self.conv1d.weight)
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]
        q, k = self._normalize_qk(q, k)
        out, _ = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            mask=mask,
            use_kernel=not self.training,
            cache=cache,
        )
        if cache is not None and hasattr(cache, "advance"):
            cache.advance(S)
            _qwen3_5_advance_lengths_info(cache, S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    cls.__call__ = call
    cls._yunshu_fused_conv = True


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
            if split_enabled() and in_prefill_math() and not _STATE["bypass"]:
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
