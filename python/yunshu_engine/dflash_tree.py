"""DFlash2 draft trees for the single-request speculative lane.

DFlash2 proposes, for a block of positions after the pending token, the top-K
candidate tokens per position (``candidate_selector``) with a unary score and a
pairwise (predecessor -> successor) score. Upstream's greedy round follows one
argmax path (a chain). This module keeps the whole lattice: a best-first search
over ``score = unary + edge * pairwise`` (log-softmax per depth, summed along
the path) picks the ``nodes`` most probable prefixes as a draft *tree*, which
``tree_verify`` checks in one forward. The walk keeps the longest root path
whose tokens equal the target's greedy tokens, so the output is the plain greedy
decode's (``tree_verify`` gives every row single-step arithmetic).

The best-first search follows TensorFold's (``drafters/dflash_tree.py``, MIT):
per-depth log-softmax of ``(unary + 0.6 * pairwise) / 1.5``. On Qwen3.8-27B
prose, 7 tree nodes commit ~3.7 tokens per cycle where upstream's adaptive chain
commits ~2.3 (docs/reports/PERF_TREND.md).
"""

from __future__ import annotations

import heapq
import logging
from typing import Any

import mlx.core as mx
import numpy as np

from . import tree_verify as tv

logger = logging.getLogger(__name__)

NODES = 7  # draft nodes per round (window = pending token + nodes <= 8 rows)
CHILDREN = 4  # candidates expanded under each node
EDGE = 0.6  # weight of the pairwise score
TAU = 1.5  # softmax temperature of the node scores


class Lattice:
    """Candidate tokens, scores and codebook rows of one drafter forward."""

    __slots__ = ("cands", "unary", "hproj", "succ", "pred", "anchor")

    def __init__(self, cands, unary, hproj, succ, pred, anchor):
        self.cands = cands  # [D, K] int64 token ids
        self.unary = unary  # [D, K] float64
        self.hproj = hproj  # [D, R]
        self.succ = succ  # [D, K, R] successor codebook rows of the candidates
        self.pred = pred  # [D, K, R] predecessor codebook rows of the candidates
        self.anchor = anchor  # [R] predecessor row of the pending token


def compute_lattice(
    drafter, anchor: int, hidden: mx.array, cache, positions: int
) -> Lattice:
    """One drafter forward over ``positions`` masked slots after ``anchor``."""
    sel = drafter.candidate_selector
    block = positions + 1
    inputs = mx.array(
        [[int(anchor)] + [int(drafter.config.mask_token_id)] * positions],
        dtype=mx.int32,
    )
    dh = drafter._hidden(inputs, hidden, cache)[:, 1:]
    logits = drafter._logits(dh)
    cands = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k :]
    unary = mx.take_along_axis(logits, cands, axis=-1).astype(mx.float32)
    hproj = sel.hidden_projection(dh).astype(mx.float32)
    succ = sel.successor_codebook(cands).astype(mx.float32)
    pred = sel.predecessor_codebook(cands).astype(mx.float32)
    anchor_row = sel.predecessor_codebook(mx.array([int(anchor)])).astype(mx.float32)
    mx.eval(cands, unary, hproj, succ, pred, anchor_row)
    del block
    return Lattice(
        np.array(cands[0]).astype(np.int64),
        np.array(unary[0]).astype(np.float64),
        np.array(hproj[0]).astype(np.float64),
        np.array(succ[0]),
        np.array(pred[0]),
        np.array(anchor_row[0]),
    )


def best_first_tree(
    lat: Lattice,
    nodes: int,
    children: int = CHILDREN,
    edge: float = EDGE,
    tau: float = TAU,
) -> tuple[list[int], list[int]]:
    """Up to ``nodes`` draft tokens by summed path log-probability, as
    ``(tokens, parents)`` in pop order (parents before children, -1 = root)."""
    depth_count = int(lat.cands.shape[0])

    def scores(depth: int, pred_row) -> np.ndarray:
        edges = lat.succ[depth].astype(np.float64) @ (
            pred_row.astype(np.float64) * lat.hproj[depth]
        )
        s = (lat.unary[depth] + edge * edges) / tau
        s = s - s.max()
        return s - np.log(np.exp(s).sum())

    tokens: list[int] = []
    parents: list[int] = []
    cand_of: list[int] = []
    heap: list[tuple[float, int, int, int]] = []
    root = scores(0, lat.anchor)
    for i in np.argsort(-root)[:children]:
        heapq.heappush(heap, (-float(root[i]), -1, 0, int(i)))
    while heap and len(tokens) < nodes:
        neg, parent, depth, i = heapq.heappop(heap)
        tokens.append(int(lat.cands[depth][i]))
        parents.append(parent)
        cand_of.append(i)
        me = len(tokens) - 1
        if depth + 1 < depth_count:
            ls = scores(depth + 1, lat.pred[depth][i])
            for j in np.argsort(-ls)[:children]:
                heapq.heappush(heap, (neg - float(ls[j]), me, depth + 1, int(j)))
    return tokens, parents


def walk(window: list[int], parents: list[int], target: list[int]) -> list[int]:
    """Rows of the accepted path (root first): follow the child whose token is
    the target's greedy token after the current row."""
    children: dict[int, list[int]] = {}
    for row in range(1, len(window)):
        children.setdefault(parents[row], []).append(row)
    path = [0]
    while True:
        want = target[path[-1]]
        nxt = next((c for c in children.get(path[-1], ()) if window[c] == want), None)
        if nxt is None:
            return path
        path.append(nxt)


def supported(model: Any, draft_model: Any) -> bool:
    lm = model.language_model if hasattr(model, "language_model") else model
    return hasattr(draft_model, "candidate_selector") and tv.supported(lm)


def dflash_tree_rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    first_bonus,
    max_tokens,
    sampler,
    draft_block_size=None,
    token_dtype=mx.int32,
    use_model_initial_block_size=True,
    greedy_sampling=True,
    _original=None,
):
    """Drop-in for upstream ``_dflash_rounds`` (single row): tree drafts."""
    from mlx_vlm.speculative.common import _record_speculative_round

    lm = model.language_model if hasattr(model, "language_model") else model
    if not greedy_sampling or not supported(model, draft_model):
        yield from _original(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            use_model_initial_block_size=use_model_initial_block_size,
            greedy_sampling=greedy_sampling,
        )
        return
    target_ids = list(draft_model.config.target_layer_ids)
    draft_cache = draft_model.reset(model)
    b = int(first_bonus)
    emitted = 1
    while emitted < max_tokens:
        nodes = min(NODES, max_tokens - emitted)
        if nodes < 1:
            break
        lat = compute_lattice(draft_model, b, hidden, draft_cache, min(nodes, 15))
        toks, pars = best_first_tree(lat, nodes)
        window = [b, *toks]
        parents = [-1] + [0 if p < 0 else p + 1 for p in pars]
        shape = tv.TreeShape(parents)
        res = tv.tree_forward(
            lm, mx.array([window], dtype=token_dtype), shape, prompt_cache, target_ids
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
        hidden = mx.concatenate(res.captured, axis=-1)[
            :, mx.array(path, dtype=mx.int32)
        ]
        b = new_tokens[-1]
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return


def install() -> bool:
    """Route upstream's single-row DFlash round loop through the tree rounds
    (idempotent). Rounds that cannot run a tree fall back to upstream's."""
    from mlx_vlm.speculative import utils as spec_utils

    current = spec_utils._dflash_rounds
    if getattr(current, "_yunshu_tree", False):
        return True
    original = current

    def rounds(*args, **kwargs):
        return dflash_tree_rounds(*args, _original=original, **kwargs)

    rounds._yunshu_tree = True
    spec_utils._dflash_rounds = rounds
    return True


__all__ = [
    "dflash_tree_rounds",
    "install",
    "best_first_tree",
    "compute_lattice",
    "walk",
]
