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

ATTN_BLOCK = 512  # default prefill attention query block (absolute grid)
LANE_TAIL = 256  # partial atoms up to this many tokens use lane projections
EVAL_EVERY = 4  # layers between evaluations of a prefill forward
PROFILE: dict[str, float] | None = None  # research: stage name -> synchronized seconds
_T = [0.0]


def _tick(name: str, *arrays) -> None:
    """With PROFILE set, evaluate ``arrays`` and charge the wall time since the
    last tick to ``name`` (adds a sync per stage; timing probes only)."""
    if PROFILE is None:
        return
    import time

    mx.eval(*arrays)
    now = time.perf_counter()
    PROFILE[name] = PROFILE.get(name, 0.0) + now - _T[0]
    _T[0] = now


@dataclass
class Segment:
    """One row's prompt chunk for this step."""

    cache: list
    tokens: mx.array  # [T] int32
    block: int = ATTN_BLOCK  # prefill attention query block (the driver's chunk)
    long_from: int = 1 << 60  # positions from here on use the coarser grid
    long_block: int = ATTN_BLOCK

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def lane(self) -> bool:
        """Short partial atom (a prompt's tail, the token after a checkpoint):
        its projections run the row-invariant lane matmul, which reads each
        weight once per <= 128 rows and needs no stock-layout copy of the
        weights (a stock call of 1..256 rows costs ~100 ms of untiling on a
        27B target). A function of the atom alone, so hit == miss."""
        return self.length < self.block and self.length <= LANE_TAIL


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
    if T == 1:
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=attn.scale, mask=None
        )
    else:
        # SDPA's bits follow its query count, so the queries go through in
        # blocks on the absolute ``seg.block`` grid (a span's edge cuts a
        # block): a token's attention is the same whatever span it prefilled in.
        n = int(keys.shape[2])
        start = n - T
        parts = []
        b0 = start
        while b0 < n:
            blk = seg.long_block if b0 >= seg.long_from else seg.block
            b1 = min((b0 // blk + 1) * blk, n)
            parts.append(
                mx.fast.scaled_dot_product_attention(
                    queries[:, :, b0 - start : b1 - start],
                    keys[:, :, :b1],
                    values[:, :, :b1],
                    scale=attn.scale,
                    mask="causal",
                )
            )
            b0 = b1
        out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
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


def _cat(parts: list) -> mx.array:
    """Token-axis concatenation; one part is returned as is (a copy of every
    projection output of a 4096-row step costs ~0.2 s per 8K prompt)."""
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def _slices(segments: list[Segment]) -> list[tuple[int, int]]:
    out, at = [], 0
    for s in segments:
        out.append((at, at + s.length))
        at += s.length
    return out


def _project(
    lin, x: mx.array, spans: list[tuple[int, int]], lane: list[bool]
) -> mx.array:
    """``lin`` over the packed tokens, one matmul per segment: a segment's
    result is what it would be as the only segment of a step. Lane projections
    run MLX's quantized matmul on a transient stock-layout weight
    (``LaneLinear.prefill``); anything else is called per segment."""
    parts = [x[:, s:e] for s, e in spans]
    if not hasattr(lin, "prefill"):
        return _cat([lin(p) for p in parts])
    outs: list = [None] * len(parts)
    stock = [i for i, short in enumerate(lane) if not short]
    if stock:
        for i, o in zip(stock, lin.prefill([parts[i] for i in stock]), strict=True):
            outs[i] = o
    for i, short in enumerate(lane):
        if short:
            outs[i] = lin(parts[i])
    return _cat(outs)


def _mlp(mlp, x: mx.array, spans: list[tuple[int, int]], lane: list[bool]) -> mx.array:
    from mlx_vlm.models.qwen3_5.language import swiglu

    h = swiglu(
        _project(mlp.gate_proj, x, spans, lane), _project(mlp.up_proj, x, spans, lane)
    )
    return _project(mlp.down_proj, h, spans, lane)


def forward(language_model: Any, segments: list[Segment]) -> mx.array:
    """Run prompt ``segments`` through the decoder in one packed forward;
    returns the final-norm hidden states ``[N, D]`` in segment order.

    Every weight-bearing op runs once per segment, so a segment's hidden
    states depend only on its own tokens and caches (never on which prompts
    share the step); the layer loop stays outside so a lane projection
    untiles its weight once for all segments of the step."""
    model = language_model.model
    spans = _slices(segments)
    lane = [seg.lane for seg in segments]
    tokens = mx.concatenate([s.tokens for s in segments]).astype(mx.int32)
    x = model.embed_tokens(tokens)[None]
    if PROFILE is not None:
        import time

        mx.eval(x)
        _T[0] = time.perf_counter()
    for i, layer in enumerate(model.layers):
        xn = layer.input_layernorm(x)
        if layer.is_linear:
            g = layer.linear_attn
            qkv, z = (
                _project(g.in_proj_qkv, xn, spans, lane),
                _project(g.in_proj_z, xn, spans, lane),
            )
            b, a = (
                _project(g.in_proj_b, xn, spans, lane),
                _project(g.in_proj_a, xn, spans, lane),
            )
            _tick("gdn_proj", qkv, z, b, a)
            parts = [
                _gdn_mix(g, qkv[:, s:e], z[:, s:e], b[:, s:e], a[:, s:e], seg.cache[i])
                for seg, (s, e) in zip(segments, spans, strict=True)
            ]
            _tick("gdn_mix", *parts)
            r = _project(g.out_proj, _cat(parts), spans, lane)
            _tick("gdn_out", r)
        else:
            at = layer.self_attn
            q = _project(at.q_proj, xn, spans, lane)
            k, v = (
                _project(at.k_proj, xn, spans, lane),
                _project(at.v_proj, xn, spans, lane),
            )
            _tick("attn_proj", q, k, v)
            parts = [
                _attention_mix(at, q[:, s:e], k[:, s:e], v[:, s:e], seg, seg.cache[i])
                for seg, (s, e) in zip(segments, spans, strict=True)
            ]
            _tick("attn_mix", *parts)
            r = _project(at.o_proj, _cat(parts), spans, lane)
            _tick("attn_out", r)
        h = x + r
        x = h + _mlp(layer.mlp, layer.post_attention_layernorm(h), spans, lane)
        _tick("mlp", x)
        # transient stock weights are freed every few layers (the graph would
        # otherwise hold every layer's)
        if i % EVAL_EVERY == EVAL_EVERY - 1:
            mx.eval(x)
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
