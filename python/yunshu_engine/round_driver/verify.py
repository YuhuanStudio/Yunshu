"""Packed lane projections with per-sequence tree attention and commit records.

This is a round-driver primitive, not a serving path. Each window owns its KV,
GDN state and tree. Only weight-bearing operations see the concatenated windows.
The caller accepts and commits each returned transaction with tree_commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .. import tree_verify as tv


@dataclass
class Window:
    tokens: Any  # [1, W] int32
    shape: Any
    cache: list


def forward(lm, windows: list[Window], capture_ids=(), *, verifier=None):
    """Verify all windows with one projection/MLP call per layer.

    At most 32 nodes total; no padding, no cross-sequence attention. The solo
    case delegates to the existing lane, retaining its fusion and submissions.
    Any failure before returning rolls back every appended attention window.
    """
    if not windows:
        return []
    if len({id(w.cache) for w in windows}) != len(windows):
        raise ValueError("packed verify requires independent caches")
    total = sum(w.shape.width for w in windows)
    if total > tv.FAST_MAX_ROWS:
        raise ValueError("packed verify exceeds the shared node budget")
    if any(w.tokens.shape != (1, w.shape.width) for w in windows):
        raise ValueError("packed verify tokens must match each window width")
    if len(windows) == 1:
        w = windows[0]
        return [
            tv.tree_forward(
                lm, w.tokens, w.shape, w.cache, capture_ids, verifier=verifier
            )
        ]
    if not tv.lane_projections(lm):
        raise ValueError("packed verify requires row-invariant lane projections")
    from mlx_vlm.models.qwen3_5 import language as q35

    verifier = q35._EXACT_SPECULATIVE_VERIFIER if verifier is None else verifier
    model = lm.model
    offsets, at = [], 0
    results, contexts, positions = [], [], []
    for w in windows:
        n0 = tv.ragged_kv._lane_length(w.cache[model.fa_idx])
        if n0 is None:
            raise ValueError("packed verify needs single-row lane caches")
        offsets.append((at, at + w.shape.width))
        at += w.shape.width
        base = n0 + tv._rope_delta(lm)
        positions.append(
            mx.broadcast_to(
                (base + w.shape.depth_array())[None, None], (3, 1, w.shape.width)
            )
        )
        contexts.append(tv.RoundContext(w.shape, n0))
        results.append(tv.TreeResult(w.shape, n0, None))
    capture = set(capture_ids)
    try:
        h = model.embed_tokens(mx.concatenate([w.tokens for w in windows], axis=1))
        for i, layer in enumerate(model.layers):
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                g = layer.linear_attn
                projected = verifier._linears(
                    (g.in_proj_qkv, g.in_proj_z, g.in_proj_b, g.in_proj_a), normed
                )
                mixed = []
                for w, res, (s, e) in zip(windows, results, offsets, strict=True):
                    out, rec = tv._gdn_layer(
                        verifier,
                        g,
                        normed[:, s:e],
                        w.cache[i],
                        w.shape,
                        projected=tuple(p[:, s:e] for p in projected),
                        project_output=False,
                    )
                    res.records[i] = rec
                    mixed.append(out)
                r = verifier._linear(g.out_proj, mx.concatenate(mixed, axis=1))
            else:
                a = layer.self_attn
                q, k, v = verifier._linears((a.q_proj, a.k_proj, a.v_proj), normed)
                mixed = []
                for b, (w, res, (s, e)) in enumerate(
                    zip(windows, results, offsets, strict=True)
                ):
                    queries, _, _, gate, _ = a._prepare_projected_qkv(
                        q[:, s:e],
                        k[:, s:e],
                        v[:, s:e],
                        w.cache[i],
                        positions[b],
                        None,
                        None,
                    )
                    # Record immediately: a later mixer/submission failure must
                    # undo this append even before its output projection exists.
                    res.records[i] = ("kv",)
                    out = tv.tree_attention(
                        queries, w.cache[i], a.scale, w.shape, res.n0, contexts[b]
                    )
                    out = out.transpose(0, 2, 1, 3).reshape(1, w.shape.width, -1)
                    mixed.append(out * mx.sigmoid(gate))
                r = verifier._linear(a.o_proj, mx.concatenate(mixed, axis=1))
            # The lane's fused add/RMS has the same arithmetic as these ops.
            h = h + r
            h = h + verifier._feed_forward(layer.mlp, layer.post_attention_layernorm(h))
            if i in capture:
                for res, (s, e) in zip(results, offsets, strict=True):
                    res.captured.append(h[:, s:e])
            if (i == 0 or (i + 1) % 4 == 0) and i + 1 < len(model.layers):
                mx.async_eval(h)
        h = model.norm(h)
        for res, (s, e) in zip(results, offsets, strict=True):
            res.hidden = h[:, s:e]
        return results
    except BaseException:
        for w, res in zip(windows, results, strict=True):
            tv.tree_abort(w.cache, res)
        raise
    finally:
        tv.vq.clear_group_sums()
