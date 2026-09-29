"""MTP draft trees for the single-request speculative lane.

The checkpoint's MTP head predicts one token per step from ``(embed(token),
target hidden)``; upstream chains it (each step feeds the previous draft), so a
wrong early draft wastes the whole tail. This module keeps the head's top-k at
every step and grows a *tree* best-first by summed log-probability, a batch of
nodes per head call (the head is one layer, so a batch costs about one step).
``tree_verify`` checks the tree in one forward; the walk keeps the longest root
path equal to the target's greedy tokens, so the output is the plain greedy
decode's.

The head's own KV (upstream's ``_cache``) holds the committed positions; tree
nodes attend to it plus their ancestors' entries, which live in per-call
buffers and are dropped after the round.
"""

from __future__ import annotations

import heapq
import logging
from typing import Any

import mlx.core as mx
import numpy as np

from . import tree_verify as tv
from .dflash_tree import walk

logger = logging.getLogger(__name__)

NODES = 7  # draft nodes per round (window = pending token + nodes <= 8 rows)
KIDS = 3  # candidates kept per expanded node
BATCH = 3  # nodes forwarded per head call


class _Node:
    __slots__ = ("token", "parent", "depth", "logp", "ancestors")

    def __init__(self, token, parent, depth, logp, ancestors):
        self.token, self.parent, self.depth, self.logp = token, parent, depth, logp
        self.ancestors = (
            ancestors  # tree indices visible to the node (its path incl. itself)
        )


def _head_layers(head):
    return head.layers


def _forward_level(
    head,
    lm,
    nodes: list[_Node],
    first: int,
    hidden_in: mx.array,
    base_pos: int,
    tree_kv,
):
    """Head forward for ``nodes`` (tree indices ``first ..``): returns their
    outputs [B, D]. ``hidden_in`` [B, D] is each node's parent output; ``tree_kv``
    holds every earlier tree node's keys/values per layer (appended here)."""
    count = len(nodes)
    tokens = mx.array([n.token for n in nodes], dtype=mx.int32)[None]
    emb = head._input_embed(tokens) * head._input_embed_scale
    h = head.fc(
        mx.concatenate(
            [
                head.pre_fc_norm_embedding(emb),
                head.pre_fc_norm_hidden(hidden_in[None].astype(emb.dtype)),
            ],
            axis=-1,
        )
    )
    pos = mx.array([[base_pos + n.depth - 1 for n in nodes]], dtype=mx.int32)
    total_prev = first
    for li, layer in enumerate(head.layers):
        cache = head._cache[li]
        n_prefix = int(cache.offset)
        xn = layer.input_layernorm(h)
        at = layer.self_attn
        q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
        queries, keys, values, gate, _ = at._prepare_projected_qkv(
            q, k, v, None, pos, None, None
        )
        pk = cache.keys[..., :n_prefix, :]
        pv = cache.values[..., :n_prefix, :]
        parts_k, parts_v = [pk], [pv]
        if tree_kv[li] is not None:
            parts_k.append(tree_kv[li][0])
            parts_v.append(tree_kv[li][1])
        parts_k.append(keys)
        parts_v.append(values)
        all_k = mx.concatenate(parts_k, axis=2)
        all_v = mx.concatenate(parts_v, axis=2)
        # mask: the whole prefix, plus the node's own ancestors among tree entries
        allow = np.zeros((count, n_prefix + total_prev + count), dtype=bool)
        allow[:, :n_prefix] = True
        for i, node in enumerate(nodes):
            for a in node.ancestors:
                allow[i, n_prefix + a] = True
        mask = mx.array(allow)[None, None]
        out = mx.fast.scaled_dot_product_attention(
            queries, all_k, all_v, scale=at.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(1, count, -1) * mx.sigmoid(gate)
        hh = h + at.o_proj(out)
        h = hh + layer.mlp(layer.post_attention_layernorm(hh))
        if tree_kv[li] is None:
            tree_kv[li] = (keys, values)
        else:
            tree_kv[li] = (
                mx.concatenate([tree_kv[li][0], keys], axis=2),
                mx.concatenate([tree_kv[li][1], values], axis=2),
            )
    return head.norm(h)[0]


def _topk(lm, hidden: mx.array, k: int):
    """Log-probabilities and ids of each row's top ``k`` next tokens (host)."""
    logits = lm.speculative_logits_from_hidden(hidden).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    idx = mx.argpartition(-logp, k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(logp, idx, axis=-1)
    mx.eval(idx, vals)
    return np.array(vals), np.array(idx)


def draft_tree(
    head, lm, root_hidden: mx.array, nodes: int, *, kids: int = KIDS, batch: int = BATCH
):
    """Best-first tree of up to ``nodes`` draft tokens from the head's output at
    the last committed position (``root_hidden`` [1, D]). Returns
    ``(tokens, parents)`` with parents in draft space (-1 = the pending token)."""
    base_pos = int(head._next_position)
    tree: list[_Node] = []
    outputs: list[mx.array] = []  # head output per tree node (None until forwarded)
    tree_kv: list = [None] * len(head.layers)
    heap: list = []  # (-cum logp, seq, parent index, token, depth)
    seq = 0

    def push(parent, cum, vals, ids, depth):
        nonlocal seq
        for lp, tk in zip(vals, ids, strict=True):
            heapq.heappush(heap, (-(cum + float(lp)), seq, parent, int(tk), depth))
            seq += 1

    vals, ids = _topk(lm, root_hidden, kids)
    push(-1, 0.0, vals[0], ids[0], 1)
    forwarded = 0
    while heap and len(tree) < nodes:
        take = min(batch, nodes - len(tree), len(heap))
        first = len(tree)
        for _ in range(take):
            neg, _, parent, tk, depth = heapq.heappop(heap)
            anc = ([*tree[parent].ancestors] if parent >= 0 else []) + [len(tree)]
            tree.append(_Node(tk, parent, depth, -neg, anc))
        if len(tree) >= nodes:
            break  # the last batch's children would not fit
        new = tree[first:]
        hid_in = mx.stack(
            [root_hidden[0] if n.parent < 0 else outputs[n.parent] for n in new]
        )
        out = _forward_level(head, lm, new, forwarded, hid_in, base_pos, tree_kv)
        forwarded += len(new)
        outputs.extend(out[i] for i in range(len(new)))
        vals, ids = _topk(lm, out, kids)
        for i, node in enumerate(new):
            push(first + i, node.logp, vals[i], ids[i], node.depth + 1)
    return [n.token for n in tree], [n.parent for n in tree]


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
    b = int(first_bonus)
    emitted = 1
    while emitted < max_tokens:
        nodes = min(NODES, max_tokens - emitted)
        if nodes < 1:
            return
        toks, pars = draft_tree(draft_model, lm, root, nodes)
        window = [b, *toks]
        parents = [-1] + [0 if p < 0 else p + 1 for p in pars]
        shape = tv.TreeShape(parents)
        res = tv.tree_forward(
            lm, mx.array([window], dtype=token_dtype), shape, prompt_cache
        )
        target = lm.speculative_argmax_from_hidden(res.hidden)
        try:
            row_tokens = [int(t) for t in target.reshape(-1).tolist()]
        except BaseException:
            tv.tree_abort(prompt_cache, res)
            raise
        path = walk(window, parents, row_tokens)
        new_tokens = [window[r] for r in path[1:]] + [row_tokens[path[-1]]]
        _record_speculative_round(draft_model, len(path) - 1, len(toks))
        tv.tree_commit(lm, prompt_cache, res, path)
        # the head absorbs the kept positions: position j pairs the token after
        # row j with that row's target hidden
        kept = res.hidden[:, mx.array(path, dtype=mx.int32)]
        h = draft_model._forward_tokens(
            mx.array([new_tokens], dtype=token_dtype), kept, token_dtype
        )
        root = h[:, -1]
        b = new_tokens[-1]
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


__all__ = ["draft_tree", "install", "mtp_tree_rounds_batch"]
