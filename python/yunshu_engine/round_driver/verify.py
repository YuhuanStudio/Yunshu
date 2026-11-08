# Upstream (inspired): Blaizzy/mlx-vlm (MIT) models/qwen4_exp/language.py, speculative/cache_state.py @ 92b31ad3
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

# The packed verify's row limit. The lane's fast tree widened to 64 rows (copy rounds,
# tree_verify.FAST_MAX_ROWS) with a one-row bit-equality probe; the driver's multi-row packs were
# validated at 32 and keep it until the same probe runs on a packed window.
DRIVER_MAX_ROWS = 32


@dataclass
class Window:
    tokens: Any  # [1, W] int32
    shape: Any
    cache: list
    rope_delta: int = 0  # the row's mRoPE decode offset (media prompts); 0 for text


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
    if total > DRIVER_MAX_ROWS and len(windows) > 1:
        raise ValueError("packed verify exceeds the shared node budget")
    if any(w.tokens.shape != (1, w.shape.width) for w in windows):
        raise ValueError("packed verify tokens must match each window width")
    if len(windows) == 1:
        w = windows[0]
        return [
            tv.tree_forward(
                lm,
                w.tokens,
                w.shape,
                w.cache,
                capture_ids,
                verifier=verifier,
                rope_delta=w.rope_delta,
            )
        ]
    if not tv.lane_projections(lm):
        raise ValueError("packed verify requires row-invariant lane projections")
    from mlx_vlm.models.qwen3_5 import language as q35

    verifier = q35._EXACT_SPECULATIVE_VERIFIER if verifier is None else verifier
    model = lm.model
    h = model.embed_tokens(mx.concatenate([w.tokens for w in windows], axis=1))
    offsets, at = [], 0
    results, contexts, positions = [], [], []
    for w in windows:
        n0 = tv.ragged_kv._lane_length(w.cache[model.fa_idx])
        if n0 is None:
            raise ValueError("packed verify needs single-row lane caches")
        offsets.append((at, at + w.shape.width))
        at += w.shape.width
        base = n0 + w.rope_delta
        positions.append(
            mx.broadcast_to(
                (base + w.shape.depth_array())[None, None], (3, 1, w.shape.width)
            )
        )
        contexts.append(tv.RoundContext(w.shape, n0))
        results.append(tv.TreeResult(w.shape, n0, h[:, offsets[-1][0] : at]))
    capture = set(capture_ids)
    try:
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


def native_chain_forward(lm, tokens, caches):
    """Flash-Next chain primitive; native per-row cache transactions.

    Returns (hyper hidden, native transaction) per row. Callers must commit
    accepted prefix lengths or abort each transaction. Not yet serving: the
    driver's native MTP lifecycle must be qualified before admission changes.
    Dense projections and routed/shared experts see all real tokens; PLE,
    QSA and GDN state remain owned by each request. No 27B shape assumptions.
    """
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    from ..qwen4_family import qwen4_first_module

    Qwen4ExpBatchInvariantForward = qwen4_first_module(
        "language"
    ).Qwen4ExpBatchInvariantForward

    if not tokens or len(tokens) != len(caches):
        raise ValueError("native chain requires matching nonempty tokens and caches")
    if len({id(c) for c in caches}) != len(caches):
        raise ValueError("native chain requires independent caches")
    if any(
        t.ndim != 2 or t.shape[0] != 1 or not 1 <= t.shape[1] <= DRIVER_MAX_ROWS
        for t in tokens
    ):
        raise ValueError("native chain supports 1..DRIVER_MAX_ROWS positions per row")
    if sum(t.shape[1] for t in tokens) > DRIVER_MAX_ROWS:
        raise ValueError("native chain exceeds the shared node budget")

    native = Qwen4ExpBatchInvariantForward()

    class Projected(Qwen4ExpBatchInvariantForward):
        # Native QSA may split a two-token chain at its sparse boundary. Each
        # projection consumes its own consecutive slice; no global patching.
        def __init__(self, projected, output):
            self.projected = projected
            self.output = output
            self.at = {}

        def _linear(self, linear, x):
            key = id(linear)
            if key == self.output:
                return x
            if key not in self.projected:
                return native._linear(linear, x)
            at = self.at.get(key, 0)
            self.at[key] = at + x.shape[1]
            return self.projected[key][:, at : at + x.shape[1]]

        def _linears(self, linears, x):
            return [self._linear(linear, x) for linear in linears]

    spans, at = [], 0
    for t in tokens:
        spans.append((at, at + t.shape[1]))
        at += t.shape[1]
    model = lm.model
    hidden = mx.tile(
        model.embed_tokens(mx.concatenate(tokens, axis=1)), (1, 1, model.args.hc_count)
    )
    transactions = []
    try:
        positions = []
        for t, cache in zip(tokens, caches, strict=True):
            mx.async_eval([entry.state for entry in cache])
            offset = int(cache[model.fa_idx].offset)
            delta = getattr(lm, "_rope_deltas", None)
            if delta is not None:
                offset += int(delta.reshape(-1)[0].item())
            positions.append(offset + mx.arange(t.shape[1], dtype=mx.int64)[None])
            transactions.append(start_speculative_cache(cache, t.shape[1]))
        for i, layer in enumerate(model.layers):
            if "ple" in layer:
                # Native PLE owns ngram history and convolution windows. Its
                # dense projections are comparatively small; keep this part
                # canonical until its packed version has a separate bit gate.
                hidden = hidden + mx.concatenate(
                    [
                        native._ple(layer.ple, hidden[:, s:e], t, c[i], None)
                        for t, c, (s, e) in zip(tokens, caches, spans, strict=True)
                    ],
                    axis=1,
                )
            mixed, residual, injection = native._hyper_connection(
                layer.attn_hyper_connection, hidden
            )
            if layer.is_linear:
                mixer = layer.linear_attn
                linears = (
                    mixer.in_proj_qkv,
                    mixer.in_proj_z,
                    mixer.in_proj_b,
                    mixer.in_proj_a,
                )
                output = mixer.out_proj
            else:
                mixer = layer.self_attn
                linears = (
                    mixer.indexer.index_qk_proj,
                    mixer.q_proj,
                    mixer.k_proj,
                    mixer.v_proj,
                )
                output = mixer.o_proj
            projected = native._linears(linears, mixed)
            parts = []
            for cache, pos, (s, e) in zip(caches, positions, spans, strict=True):
                view = Projected(
                    {id(p): v[:, s:e] for p, v in zip(linears, projected, strict=True)},
                    id(output),
                )
                if layer.is_linear:
                    part = view._gated_delta(mixer, mixed[:, s:e], None, cache[i])
                else:
                    part = view._qsa_attention(
                        mixer,
                        mixed[:, s:e],
                        cache[i],
                        pos,
                        "causal" if e - s > 1 else None,
                    )
                parts.append(part)
            branch = native._linear(output, mx.concatenate(parts, axis=1))
            hidden = native._inject(branch, residual, injection)
            mixed, residual, injection = native._hyper_connection(
                layer.mlp_hyper_connection, hidden
            )
            # This includes router, routed expert projections and shared expert;
            # the native verifier keeps its canonical per-token reductions.
            hidden = native._inject(
                native._feed_forward(layer.mlp, mixed), residual, injection
            )
            mx.async_eval(hidden)
        return [
            (hidden[:, s:e], transaction)
            for (s, e), transaction in zip(spans, transactions, strict=True)
        ]
    except BaseException:
        for transaction in transactions:
            transaction.abort()
        raise
