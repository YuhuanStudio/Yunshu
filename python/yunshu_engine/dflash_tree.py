# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/drafters/dflash_tree.py @ 34bae79a
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""DFlash2 draft trees for the single-request speculative lane.

DFlash2 proposes, for a block of positions after the pending token, the top-K
candidate tokens per position (``candidate_selector``) with a unary score and a
pairwise (predecessor -> successor) score. Upstream's greedy round follows one
argmax path (a chain). This module keeps the whole lattice: the ``r``-th best
candidate under ``(unary + 0.6 * pairwise) / 1.5`` given the parent's token fills
each node of a fixed-shape draft *tree* (``spec_topology``), on the GPU, and
``tree_verify`` checks the tree in one forward. The walk keeps the longest root
path whose tokens equal the target's greedy tokens, so the output is the plain
greedy decode's (``tree_verify`` gives every row single-step arithmetic).

The lattice scoring follows TensorFold's draft-tree search
(``drafters/dflash_tree.py``, MIT): per-position scores ``unary + edge *
pairwise`` softened by ``tau``. TensorFold grows the tree best-first on the CPU;
here the shape is fixed (a rank-path topology) so no host round trip sits
between the drafter and the verify.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlx.core as mx

from . import tree_verify as tv
from .spec_schedule import NodeBudget

logger = logging.getLogger(__name__)

CHILDREN = 4  # candidates expanded under each node
BEAM = 15  # paths kept per level (the tree has at most NODES nodes)
EDGE = 0.6  # weight of the pairwise score
TAU = 1.5  # softmax temperature of the node scores


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


class GpuLattice:
    """A drafter forward's candidates and scores, still on the GPU."""

    __slots__ = ("cands", "unary", "hproj", "succ", "pred", "anchor")

    def __init__(self, cands, unary, hproj, succ, pred, anchor):
        self.cands, self.unary, self.hproj = cands, unary, hproj
        self.succ, self.pred, self.anchor = succ, pred, anchor


def compute_lattice_gpu(
    drafter, anchor: int, hidden: mx.array, cache, positions: int
) -> GpuLattice:
    """``compute_lattice`` without the host round trip (nothing is evaluated)."""
    sel = drafter.candidate_selector
    inputs = mx.array(
        [[int(anchor)] + [int(drafter.config.mask_token_id)] * positions],
        dtype=mx.int32,
    )
    dh = drafter._hidden(inputs, hidden, cache)[:, 1:]
    logits = drafter._logits(dh)
    cands = mx.argpartition(logits, -sel.top_k, axis=-1)[0, ..., -sel.top_k :]  # [D, K]
    unary = mx.take_along_axis(logits[0], cands, axis=-1).astype(mx.float32)
    hproj = sel.hidden_projection(dh)[0].astype(mx.float32)
    succ = sel.successor_codebook(cands).astype(mx.float32)  # [D, K, R]
    pred = sel.predecessor_codebook(cands).astype(mx.float32)
    anchor_row = sel.predecessor_codebook(mx.array([int(anchor)]))[0].astype(mx.float32)
    return GpuLattice(cands, unary, hproj, succ, pred, anchor_row)


def search_tree(
    lat: GpuLattice,
    nodes: int,
    children: int = CHILDREN,
    beam: int = BEAM,
    edge: float = EDGE,
    tau: float = TAU,
):
    """Best-first draft tree on the GPU: the ``nodes`` most probable prefixes of
    the lattice by summed log-probability, ``(unary + edge * pairwise) / tau``
    normalized per position (each node expands to its ``children`` best
    candidates). A level keeps its ``beam`` best paths (enough to contain every
    member of the global top ``nodes``, whose ancestors all rank above them).

    Returns ``(tokens [n], parents [n + 1])``: tokens in decreasing path
    probability, parents in window rows (row 0 = the pending token, parent -1),
    every parent before its children so any prefix is a tree. Nothing is read
    back, and the shape depends on the data: use it with ``DynamicShape``."""
    depth_count = int(lat.cands.shape[0])
    neg = -1e9
    floor = -1e-4  # strictly decreasing along a path: parents sort before children

    def log_probs(scores):
        lp = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        return mx.minimum(lp, floor)

    root = (lat.unary[0] + edge * ((lat.anchor * lat.hproj[0]) @ lat.succ[0].T)) / tau
    lp = log_probs(root)
    order = mx.argsort(-lp)[:children]
    cum = mx.concatenate([mx.take(lp, order), mx.full((beam - children,), neg)])
    cidx = mx.concatenate(
        [order.astype(mx.int32), mx.zeros((beam - children,), mx.int32)]
    )
    tok = [mx.take(lat.cands[0], cidx)]
    par = [mx.full((beam,), -1, dtype=mx.int32)]
    tot = [cum]
    for d in range(1, depth_count):
        pr = mx.take(lat.pred[d - 1], cidx, axis=0)  # [B, R]
        scores = (
            lat.unary[d][None] + edge * ((pr * lat.hproj[d][None]) @ lat.succ[d].T)
        ) / tau
        lp = log_probs(scores)
        order = mx.argsort(-lp, axis=-1)[:, :children]  # [B, C]
        flat = (cum[:, None] + mx.take_along_axis(lp, order, axis=1)).reshape(-1)
        top = mx.argsort(-flat)[:beam]
        cum = mx.take(flat, top)
        cidx = mx.take(order.reshape(-1), top).astype(mx.int32)
        tok.append(mx.take(lat.cands[d], cidx))
        par.append((d - 1) * beam + (top // children).astype(mx.int32))
        tot.append(cum)
    pool_tok = mx.concatenate(tok).astype(mx.int32)
    pool_par = mx.concatenate(par)
    pool_cum = mx.concatenate(tot)
    sel = mx.argsort(-pool_cum)[:nodes].astype(mx.int32)
    where = mx.full((int(pool_cum.shape[0]),), -1, dtype=mx.int32)
    where[sel] = mx.arange(nodes, dtype=mx.int32)
    parent = mx.take(pool_par, sel)
    rows = mx.where(parent < 0, 0, mx.take(where, mx.maximum(parent, 0)) + 1)
    return mx.take(pool_tok, sel), mx.concatenate(
        [mx.array([-1], dtype=mx.int32), rows.astype(mx.int32)]
    )


def quantize_drafter(drafter: Any, bits: int = 8, group_size: int = 64) -> int:
    """Quantize the drafter's projections in place (before ``bind``): drafts are
    verified, so this changes only how often they land, and the drafter reads
    ``bits / 16`` of the bytes per cycle. Returns the layers converted."""
    import mlx.nn as nn

    before = sum(
        1 for _, m in drafter.named_modules() if isinstance(m, nn.QuantizedLinear)
    )
    nn.quantize(
        drafter,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda _p, m: (
            isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0
        ),
    )
    mx.eval(drafter.parameters())
    return (
        sum(1 for _, m in drafter.named_modules() if isinstance(m, nn.QuantizedLinear))
        - before
    )


def supported(model: Any, draft_model: Any) -> bool:
    lm = model.language_model if hasattr(model, "language_model") else model
    return hasattr(draft_model, "candidate_selector") and tv.supported(lm)


NODES = 15  # draft nodes at most per round (window = pending token + nodes <= 16 rows)
# landing probability of the i-th best node before the round history says otherwise
PRIOR = [
    0.75,
    0.5,
    0.35,
    0.25,
    0.2,
    0.15,
    0.1,
    0.1,
    0.08,
    0.08,
    0.06,
    0.06,
    0.05,
    0.05,
    0.05,
]
POSITIONS = 7  # masked positions the drafter fills (its trained block minus one)


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
    if (
        not greedy_sampling
        or not supported(model, draft_model)
        or not tv.lane_ready(lm, prompt_cache)
    ):
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
    budget = NodeBudget(NODES, prior=PRIOR)
    root_shape = tv.TreeShape([-1])
    b = int(first_bonus)
    emitted = 1
    while emitted < max_tokens:
        started = time.perf_counter()
        n = budget.choose(max_tokens - emitted)
        if n:
            lat = compute_lattice_gpu(draft_model, b, hidden, draft_cache, POSITIONS)
            toks, parents = search_tree(lat, n)
            window = mx.concatenate(
                [mx.array([b], dtype=mx.int32), toks.astype(mx.int32)]
            )[None]
            shape = tv.DynamicShape(parents, POSITIONS)
            hidden = None  # the drafter has read the committed positions
        else:
            window, parents, shape = mx.array([[b]], dtype=mx.int32), None, root_shape
        res = tv.tree_forward(lm, window, shape, prompt_cache, target_ids)
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
        kept = mx.concatenate(res.captured, axis=-1)[:, mx.array(path, dtype=mx.int32)]
        # a round without drafts leaves the committed positions for the next draft
        hidden = kept if hidden is None else mx.concatenate([hidden, kept], axis=1)
        b = new_tokens[-1]
        budget.observe(
            n,
            [r - 1 for r in path[1:]],
            (time.perf_counter() - started) * 1e3,
            first=emitted == 1,
        )
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
    "walk",
]
