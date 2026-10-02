# Patches upstream mlx-vlm dispatch; kernels remain unchanged.
"""Shared GDN decode dispatch using the existing singleton prework kernel.

The singleton's fused prework is row-invariant already, but upstream dispatch
only uses it for an ArraysCache with B=1. Reuse it for unpadded batch caches, including a batch that shrank to B=1.
No kernel, weight layout, recurrence or checkpoint representation is changed.
"""

from __future__ import annotations

import logging
import threading

import mlx.core as mx

_STATE = threading.local()
logger = logging.getLogger(__name__)


def set_active(active):
    _STATE.active = bool(active)


def fused_gdn_step(layer, inputs, cache):
    from mlx_vlm.models.qwen3_5 import language as q35

    from .kernels.omlx.qwen35_gdn_prework import gdn_prework_fused

    batch = inputs.shape[0]
    qkv = layer.in_proj_qkv(inputs)
    z = layer.in_proj_z(inputs).reshape(batch, 1, layer.num_v_heads, layer.head_v_dim)
    b, a = layer._project_gates(inputs)
    inv = layer.head_k_dim**-0.5
    q, k, v, conv = gdn_prework_fused(
        qkv,
        cache[0],
        layer.conv1d.weight,
        mx.array(inv * inv, dtype=inputs.dtype),
        mx.array(inv, dtype=inputs.dtype),
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_k_dim,
        layer.head_v_dim,
    )
    out, state = q35.gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state=cache[1],
        use_kernel=True,
    )
    result = layer.out_proj(layer.norm(out, z).reshape(batch, 1, -1))
    cache[0], cache[1] = conv, state
    if hasattr(cache, "advance"):
        cache.advance(1)
        q35._qwen3_5_advance_lengths_info(cache, 1)
    return result


def canonical_attention(cache, queries, scale):
    from .kernels.ragged_attention import ragged_decode_attention

    return ragged_decode_attention(
        queries,
        cache.keys,
        cache.values,
        cache.offset,
        scale,
        max_length=cache._idx,
        k_scales=cache.k_scales,
        v_scales=cache.v_scales,
        row_lengths=cache.lengths,
        slots=cache.slot_ids,
        impl="tile",
    )


def install_attention():
    from .kernels.ragged_attention import tile_ready
    from .kernels.ragged_kv import RaggedKVCache

    cls = RaggedKVCache
    if getattr(cls, "_yunshu_prefix_attention", False):
        return
    original = cls.attend
    logged = False

    def attend(self, queries, scale):
        nonlocal logged
        if (
            getattr(_STATE, "active", False)
            and queries.dtype == mx.bfloat16
            and self.keys.dtype == mx.bfloat16
            and queries.shape[-1] == 256
            and queries.shape[1] // self.keys.shape[1] <= 8
            and tile_ready()
        ):
            if not logged:
                logged = True
                logger.info("APC shared attention tile arithmetic engaged")
            return canonical_attention(self, queries, scale)
        return original(self, queries, scale)

    cls.attend = attend
    cls._yunshu_prefix_attention = True


def install():
    from mlx_vlm.models.qwen3_5 import language as q35

    install_attention()
    cls = q35.Qwen3_5GatedDeltaNet
    if getattr(cls, "_yunshu_prefix_decode", False):
        return
    original = cls.__call__
    logged = False

    def decode(self, inputs, mask=None, cache=None, **kwargs):
        nonlocal logged
        eligible = (
            getattr(_STATE, "active", False)
            and not kwargs
            and not self.training
            and inputs.ndim == 3
            and inputs.shape[0] >= 1
            and inputs.shape[1] == 1
            and mx.default_device() == mx.gpu
            and inputs.dtype in (mx.bfloat16, mx.float16)
            and mask is None
            and cache is not None
            and not cache.is_speculating
            and (cache.lengths is None or (q35._qwen3_5_lengths_info(cache) or 0) >= 1)
            and self.head_k_dim == self.head_v_dim == 128
            and self.conv_kernel_size == 4
            and self.conv1d.weight.dtype == inputs.dtype
            and getattr(self.conv1d, "bias", None) is None
            and cache[0] is not None
            and cache[1] is not None
            and cache[0].shape == (inputs.shape[0], 3, self.conv_dim)
            and cache[0].dtype == inputs.dtype
            and cache[1].shape == (inputs.shape[0], self.num_v_heads, 128, 128)
            and cache[1].dtype == mx.float32
        )
        if not eligible:
            return original(self, inputs, mask=mask, cache=cache, **kwargs)
        if not logged:
            logged = True
            logger.info(
                "APC shared GDN singleton arithmetic engaged: rows=%d", inputs.shape[0]
            )
        return fused_gdn_step(self, inputs, cache)

    cls.__call__ = decode
    cls._yunshu_prefix_decode = True
