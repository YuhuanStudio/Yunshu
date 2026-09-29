# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_multi.py, src/tensorfold/kernels/qwen/dense/v1/lane_glue.py @ 34bae79a
"""Prefill: one packed forward over prompt chunks of several rows.

A segment is one row's chunk (a fixed span of its prompt) with that row's own
caches (a ``KVCache`` per attention layer, an ``ArraysCache`` per GDN layer).
Every weight-bearing op runs once over the packed tokens of all segments
(norms, GDN / attention in and out projections, MLP): with the target's
projections converted to ``LaneLinear`` those are row-invariant, so a chunk's
hidden states have the bits it would get in a forward of its own. Only the
sequence mixers run per segment, on the row's caches: attention is causal SDPA
over the row's keys, the GDN conv + recurrence run on the chunk. Decoding rows
run in ``batch.DecodeBatch``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Segment:
    """One row's prompt chunk for this step."""

    cache: list
    tokens: mx.array  # [T] int32

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


def supports(language_model: Any) -> bool:
    """Dense Qwen3.5-family text decoder (hybrid GDN + full attention)."""
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:  # pragma: no cover
        return False
    inner = getattr(language_model, "model", None)
    if not isinstance(inner, q35.Qwen3_5Model):
        return False
    for layer in inner.layers:
        mlp = getattr(layer, "mlp", None)
        if not all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj")):
            return False  # MoE layers: not yet
    return True


def _attention_mix(attn, q, k, v, seg: Segment, cache) -> mx.array:
    T = seg.length
    queries, keys, values, gate, _ = attn._prepare_projected_qkv(
        q, k, v, cache, None, None, None
    )
    out = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn.scale, mask="causal" if T > 1 else None
    )
    out = out.transpose(0, 2, 1, 3).reshape(1, T, -1)
    return out * mx.sigmoid(gate)


def _gdn_mix(layer, qkv, z, b, a, cache) -> mx.array:
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update

    _, S, _ = qkv.shape
    z = z.reshape(1, S, -1, layer.head_v_dim)
    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros(
            (1, layer.conv_kernel_size - 1, layer.conv_dim), dtype=qkv.dtype
        )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache.update_window(0, conv_input, layer.conv_kernel_size - 1)
    conv_out = nn.silu(layer.conv1d(conv_input))
    q, k, v = [
        t.reshape(1, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [layer.key_dim, 2 * layer.key_dim], -1),
            [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
            [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            strict=True,
        )
    ]
    inv = k.shape[-1] ** -0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    out, _ = gated_delta_update(
        q, k, v, a, b, layer.A_log, layer.dt_bias, use_kernel=True, cache=cache
    )
    if hasattr(cache, "advance"):
        cache.advance(S)
    return layer.norm(out, z).reshape(1, S, -1)


def _slices(segments: list[Segment]) -> list[tuple[int, int]]:
    out, at = [], 0
    for s in segments:
        out.append((at, at + s.length))
        at += s.length
    return out


def forward(language_model: Any, segments: list[Segment]) -> mx.array:
    """Run prompt ``segments`` through the decoder in one packed forward;
    returns the final-norm hidden states ``[N, D]`` in segment order."""
    model = language_model.model
    spans = _slices(segments)
    tokens = mx.concatenate([s.tokens for s in segments]).astype(mx.int32)
    x = model.embed_tokens(tokens)[None]
    for i, layer in enumerate(model.layers):
        xn = layer.input_layernorm(x)
        if layer.is_linear:
            g = layer.linear_attn
            qkv, z = g.in_proj_qkv(xn), g.in_proj_z(xn)
            b, a = g.in_proj_b(xn), g.in_proj_a(xn)
            parts = [
                _gdn_mix(g, qkv[:, s:e], z[:, s:e], b[:, s:e], a[:, s:e], seg.cache[i])
                for seg, (s, e) in zip(segments, spans, strict=True)
            ]
            r = g.out_proj(mx.concatenate(parts, axis=1))
        else:
            at = layer.self_attn
            q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
            parts = [
                _attention_mix(at, q[:, s:e], k[:, s:e], v[:, s:e], seg, seg.cache[i])
                for seg, (s, e) in zip(segments, spans, strict=True)
            ]
            r = at.o_proj(mx.concatenate(parts, axis=1))
        h = x + r
        x = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.norm(x)[0]


def logits(language_model: Any, hidden: mx.array) -> mx.array:
    """LM head over hidden rows ``[R, D]`` (a lane projection once converted;
    tied embeddings use ``round_driver`` 's lane head when available)."""
    head = getattr(language_model, "_yunshu_lane_head", None)
    if head is not None:
        return head(hidden)
    if language_model.args.tie_word_embeddings:
        return language_model.model.embed_tokens.as_linear(hidden)
    return language_model.lm_head(hidden)


__all__ = ["Segment", "forward", "logits", "supports"]
