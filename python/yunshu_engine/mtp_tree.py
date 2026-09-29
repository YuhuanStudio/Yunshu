"""MTP draft trees for the single-request speculative lane.

The checkpoint's MTP head predicts one token per step from ``(embed(token),
target hidden)``; upstream chains it (each step feeds the previous draft), so a
wrong early draft wastes the whole tail. This module keeps the head's top-k at
every step and fills a *tree* of fixed shape (``spec_topology``: node ``(0, 1)``
is the second-ranked child of the top-ranked child of the pending token), one
head call per tree level (the head is one layer, so a level costs about one
step). Everything is computed on the GPU without a host round trip, so the
verify graph can be built while the head runs. ``tree_verify`` checks the tree
in one forward; the walk keeps the longest root
path equal to the target's greedy tokens, so the output is the plain greedy
decode's.

The head's own KV (upstream's ``_cache``) holds the committed positions; tree
nodes attend to it plus their ancestors' entries, which live in per-call
buffers and are dropped after the round.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlx.core as mx

from . import tree_verify as tv
from .dflash_tree import walk
from .spec_schedule import NodeBudget

logger = logging.getLogger(__name__)

NODES = 7  # draft nodes at most per round (window = pending token + nodes <= 8 rows)
LEVELS = 4  # tree depth: head calls = LEVELS - 1 (plus one for the root's logits)
BEAM = 4  # paths kept and expanded per level
CHILDREN = 3  # candidates taken under each expanded path
# landing probability of the i-th best node before the round history says otherwise
PRIOR = [0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.1]


def _forward_level(
    head, tokens, hidden_in, depth, tree_mask, n_prefix, base_pos, tree_kv
):
    """Head forward for one level's nodes: ``tokens`` [B], ``hidden_in`` [B, D]
    (each node's parent output). ``tree_mask`` [B, earlier + B] says which tree
    entries each node sees (its ancestors and itself); the head's own KV prefix
    is visible to all. Appends the level's keys/values to ``tree_kv``."""
    count = int(tokens.shape[0])
    emb = head._input_embed(tokens[None]) * head._input_embed_scale
    h = head.fc(
        mx.concatenate(
            [
                head.pre_fc_norm_embedding(emb),
                head.pre_fc_norm_hidden(hidden_in[None].astype(emb.dtype)),
            ],
            axis=-1,
        )
    )
    pos = mx.full((1, count), base_pos + depth, dtype=mx.int32)
    mask = mx.concatenate(
        [mx.ones((count, n_prefix), dtype=mx.bool_), tree_mask], axis=1
    )[None, None]
    for li, layer in enumerate(head.layers):
        cache = head._cache[li]
        xn = layer.input_layernorm(h)
        at = layer.self_attn
        q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
        queries, keys, values, gate, _ = at._prepare_projected_qkv(
            q, k, v, None, pos, None, None
        )
        parts_k = [cache.keys[..., :n_prefix, :]]
        parts_v = [cache.values[..., :n_prefix, :]]
        if tree_kv[li] is not None:
            parts_k.append(tree_kv[li][0])
            parts_v.append(tree_kv[li][1])
        parts_k.append(keys)
        parts_v.append(values)
        all_k = mx.concatenate(parts_k, axis=2)
        all_v = mx.concatenate(parts_v, axis=2)
        out = mx.fast.scaled_dot_product_attention(
            queries, all_k, all_v, scale=at.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(1, count, -1) * mx.sigmoid(gate)
        hh = h + at.o_proj(out)
        h = hh + layer.mlp(layer.post_attention_layernorm(hh))
        tree_kv[li] = (
            (keys, values)
            if tree_kv[li] is None
            else (
                mx.concatenate([tree_kv[li][0], keys], axis=2),
                mx.concatenate([tree_kv[li][1], values], axis=2),
            )
        )
    return head.norm(h)[0]


def _top_children(lm, hidden, count):
    """Each row's ``count`` most likely next tokens (best first) and their
    log-probabilities: ([R, count] ids, [R, count] logp)."""
    logits = lm.speculative_logits_from_hidden(hidden).astype(mx.float32)
    idx = mx.argpartition(-logits, count - 1, axis=-1)[..., :count]
    vals = mx.take_along_axis(logits, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1).astype(mx.int32)
    vals = mx.take_along_axis(vals, order, axis=-1)
    lp = vals - mx.logsumexp(logits, axis=-1, keepdims=True)
    return idx, mx.minimum(lp, -1e-4)  # strictly decreasing along a path


def search_tree(head, lm, root_hidden: mx.array, nodes: int):
    """Best-first draft tree from the head, on the GPU, level by level: each
    level keeps its ``BEAM`` most probable paths and expands them with one head
    call; the ``nodes`` most probable nodes overall form the tree. Returns
    ``(tokens [n], parents [n + 1])`` as ``dflash_tree.search_tree`` (parents in
    window rows, best node first, parents before children); nothing is read back."""
    n_prefix = int(head._cache[0].offset)
    base_pos = int(head._next_position)
    tree_kv: list = [None] * len(head.layers)
    idx, lp = _top_children(lm, root_hidden, BEAM)  # the root's children
    tok, cum = idx[0], lp[0]  # [B]
    pool_tok, pool_cum = [tok], [cum]
    pool_par = [mx.full((BEAM,), -1, dtype=mx.int32)]
    prev_out = None
    parent_local = None
    anc = None  # [entries, entries]: which forwarded entries each entry sees
    for lvl in range(LEVELS - 1):
        eye = mx.eye(BEAM, dtype=mx.bool_)
        if lvl == 0:
            hid_in = mx.broadcast_to(root_hidden, (BEAM, root_hidden.shape[-1]))
            tree_mask = anc = eye
        else:
            hid_in = prev_out
            rows = mx.take(anc, (lvl - 1) * BEAM + parent_local, axis=0)
            tree_mask = mx.concatenate([rows, eye], axis=1)
            entries = lvl * BEAM
            anc = mx.concatenate(
                [
                    mx.concatenate(
                        [anc, mx.zeros((entries, BEAM), dtype=mx.bool_)], axis=1
                    ),
                    tree_mask,
                ],
                axis=0,
            )
        out = _forward_level(
            head, tok, hid_in, lvl, tree_mask, n_prefix, base_pos, tree_kv
        )
        # expand every kept path to its best children and keep the best BEAM
        cidx, clp = _top_children(lm, out, CHILDREN)  # [B, C]
        flat = (cum[:, None] + clp).reshape(-1)
        top = mx.argsort(-flat)[:BEAM]
        cum = mx.take(flat, top)
        parent_local = (top // CHILDREN).astype(mx.int32)
        tok = mx.take(cidx.reshape(-1), top)
        pool_tok.append(tok)
        pool_cum.append(cum)
        pool_par.append(lvl * BEAM + parent_local)
        prev_out = mx.take(out, parent_local, axis=0)
    all_tok = mx.concatenate(pool_tok)
    all_cum = mx.concatenate(pool_cum)
    all_par = mx.concatenate(pool_par)
    sel = mx.argsort(-all_cum)[:nodes].astype(mx.int32)
    where = mx.full((int(all_cum.shape[0]),), -1, dtype=mx.int32)
    where[sel] = mx.arange(nodes, dtype=mx.int32)
    parent = mx.take(all_par, sel)
    rows = mx.where(parent < 0, 0, mx.take(where, mx.maximum(parent, 0)) + 1)
    return mx.take(all_tok, sel), mx.concatenate(
        [mx.array([-1], dtype=mx.int32), rows.astype(mx.int32)]
    )


def supported(model: Any, draft_model: Any) -> bool:
    lm = model.language_model if hasattr(model, "language_model") else model
    return (
        hasattr(draft_model, "_forward_tokens")
        and hasattr(draft_model, "prefill_from_target_hidden")
        and tv.supported(lm)
    )


def _rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    prompt_tokens,
    first_bonus,
    max_tokens,
    sampler,
    token_dtype,
):
    """Rounds of tree drafts and verify for one greedy row; yields each round's
    committed tokens (a list)."""
    from mlx_vlm.speculative.common import _record_speculative_round

    lm = model.language_model if hasattr(model, "language_model") else model
    draft_model.reset(model)
    draft_model.prefill_from_target_hidden(
        prompt_tokens, hidden, int(first_bonus), sampler, token_dtype, greedy=True
    )
    root = draft_model._seed_hidden[
        :, -1
    ]  # head output at the last committed position [1, D]
    budget = NodeBudget(NODES, prior=PRIOR)
    root_shape = tv.TreeShape([-1])
    b = int(first_bonus)
    emitted = 1
    while emitted < max_tokens:
        started = time.perf_counter()
        n = budget.choose(max_tokens - emitted)
        if n:
            toks, parents = search_tree(draft_model, lm, root, n)
            window = mx.concatenate([mx.array([b], dtype=mx.int32), toks])[None]
            shape = tv.DynamicShape(parents, LEVELS)
        else:
            window, parents, shape = mx.array([[b]], dtype=mx.int32), None, root_shape
        res = tv.tree_forward(lm, window, shape, prompt_cache)
        target = lm.speculative_argmax_from_hidden(res.hidden)
        try:
            if parents is None:
                mx.async_eval(target, window)
                wparents = [-1]
            else:
                mx.async_eval(target, window, parents)
                wparents = [int(t) for t in parents.tolist()]
            row_tokens = [int(t) for t in target.reshape(-1).tolist()]
            tokens = [int(t) for t in window.reshape(-1).tolist()]
        except BaseException:
            tv.tree_abort(prompt_cache, res)
            raise
        path = walk(tokens, wparents, row_tokens)
        new_tokens = [tokens[r] for r in path[1:]] + [row_tokens[path[-1]]]
        if n:
            _record_speculative_round(draft_model, len(path) - 1, n)
        tv.tree_commit(lm, prompt_cache, res, path)
        # the head absorbs the kept positions: position j pairs the token after
        # row j with that row's target hidden
        kept = res.hidden[:, mx.array(path, dtype=mx.int32)]
        h = draft_model._forward_tokens(
            mx.array([new_tokens], dtype=token_dtype), kept, token_dtype
        )
        root = h[:, -1]
        b = new_tokens[-1]
        budget.observe(
            n,
            [r - 1 for r in path[1:]],
            (time.perf_counter() - started) * 1e3,
            first=emitted == 1,
        )
        emitted += len(new_tokens)
        yield new_tokens


def mtp_tree_rounds_batch(
    model,
    draft_model,
    prompt_cache,
    hidden,
    shared_kv_states,
    *,
    prompt_tokens=None,
    first_bonus,
    max_tokens,
    sampler,
    draft_block_size=None,
    token_dtype=mx.int32,
    stop_check=None,
    eos_token_ids=None,
    greedy_sampling=False,
    row_ids=None,
    _original=None,
):
    """Drop-in for upstream ``_mtp_rounds_batch`` for one greedy row."""
    if (
        int(first_bonus.shape[0]) != 1
        or not greedy_sampling
        or prompt_tokens is None
        or not supported(model, draft_model)
        or not tv.lane_ready(
            model.language_model if hasattr(model, "language_model") else model,
            prompt_cache,
        )
    ):
        yield from _original(
            model,
            draft_model,
            prompt_cache,
            hidden,
            shared_kv_states,
            prompt_tokens=prompt_tokens,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            eos_token_ids=eos_token_ids,
            greedy_sampling=greedy_sampling,
            row_ids=row_ids,
        )
        return
    emitted = 1
    for new_tokens in _rounds(
        model,
        draft_model,
        prompt_cache,
        hidden,
        prompt_tokens=prompt_tokens,
        first_bonus=int(first_bonus.reshape(-1).item()),
        max_tokens=max_tokens,
        sampler=sampler,
        token_dtype=token_dtype,
    ):
        for pos, tok in enumerate(new_tokens):
            emitted += 1
            yield [tok], {"round_pos": pos, "round_len": len(new_tokens)}
            if (
                emitted >= max_tokens
                or (eos_token_ids is not None and tok in eos_token_ids)
                or (stop_check is not None and stop_check(0, tok))
            ):
                return


def install() -> bool:
    """Route upstream's MTP round loop (the server path, one greedy row)
    through the tree rounds (idempotent)."""
    from mlx_vlm.speculative import utils as spec_utils

    current = spec_utils._mtp_rounds_batch
    if getattr(current, "_yunshu_tree", False):
        return True
    original = current

    def rounds(*args, **kwargs):
        return mtp_tree_rounds_batch(*args, _original=original, **kwargs)

    rounds._yunshu_tree = True
    spec_utils._mtp_rounds_batch = rounds
    return True


__all__ = ["install", "mtp_tree_rounds_batch", "search_tree"]
