# Upstream (inspired): TensorFold src/tensorfold/engine/exact_sampling.py (keyed Gumbel draws)
"""Position-keyed sampling: the token drawn at generation index ``g`` is a pure function of
(logits row, seed, g).

Gumbel-max sampling draws ``argmax(logprobs / T + G)`` with ``G`` independent Gumbel noise per
(position, token id). Here the noise is a hash of ``(seed, position, token id)`` instead of a
stateful PRNG, so a drafted token is accepted exactly when it equals the token serial sampling
would have drawn at that position. That keeps speculative decoding lossless for sampled
requests: the emitted stream is distributed exactly like serial sampling (each position uses
fresh, independent noise, whatever was drafted), and with the same seed it is the same stream
whether or not drafts are used (given batch-invariant logits).

Filters (top-p, min-p, top-k, temperature) run in the order of ``vlm_batch_runner.RowSampler``
so the distribution matches the non-speculative path.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np

_C0 = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
_C3 = 0xD1B54A32D192ED03


def _u64(x: int) -> mx.array:
    return mx.array(np.uint64(x))


def _mix(x: mx.array) -> mx.array:
    x = x ^ (x >> _u64(30))
    x = x * _u64(_C1)
    x = x ^ (x >> _u64(27))
    x = x * _u64(_C2)
    return x ^ (x >> _u64(31))


_M64 = 0xFFFFFFFFFFFFFFFF


def _mix_int(x: int) -> int:
    x ^= x >> 30
    x = (x * _C1) & _M64
    x ^= x >> 27
    x = (x * _C2) & _M64
    return x ^ (x >> 31)


def seed_base(seed: int) -> int:
    """Per-request constant of the noise hash (python int, no device work)."""
    return _mix_int((int(seed) & _M64) ^ _C0)


def _gumbel_at(bases: mx.array, positions: mx.array, ids: mx.array) -> mx.array:
    """Gumbel(0, 1) noise for token ``ids`` ([1, V] or [N, K], uint64): row n is keyed by
    (bases[n], positions[n], id), so any subset of ids gets the same values as the full row."""
    pos = positions.astype(mx.uint64).reshape(-1, 1)
    x = _mix(bases.astype(mx.uint64).reshape(-1, 1) ^ (pos * _u64(_C3)))
    x = _mix(x ^ ids)
    # 23 random bits -> u in [2^-24, 1 - 2^-24], exactly representable in float32. (With 24
    # bits the top bucket is 1 - 2^-25, which float32 rounds to 1.0: -log(-log(1)) = +inf, a
    # token with infinite noise that wins even when filtered out, or NaN with its -inf row.)
    u = ((x >> _u64(41)).astype(mx.float32) + 0.5) * (1.0 / (1 << 23))
    return -mx.log(-mx.log(u))


def gumbel_rows(bases: mx.array, positions: mx.array, vocab: int) -> mx.array:
    """Gumbel(0, 1) noise ``[N, vocab]``: row n is keyed by (bases[n], positions[n], token id).

    ``bases`` are ``seed_base`` values as a uint64 array, so rows with different seeds are
    drawn in one graph."""
    return _gumbel_at(bases, positions, mx.arange(vocab, dtype=mx.uint64)[None, :])


def gumbel(seed: int, positions: mx.array, vocab: int) -> mx.array:
    """Gumbel(0, 1) noise ``[N, vocab]`` keyed by (seed, positions[n], token id)."""
    n = int(positions.size)
    bases = mx.array(np.full((n,), seed_base(seed), dtype=np.uint64))
    return gumbel_rows(bases, positions, vocab)


def sample_rows(
    logprobs: mx.array, params: Any, bases: mx.array, positions: mx.array
) -> mx.array:
    """Tokens ``[N]`` for rows that share ``params`` but have their own seed and position."""
    row = logprobs.astype(mx.float32)
    cand = candidate_rows(row, params)
    if cand is not None:
        ids, scaled = cand
        noisy = mx.where(
            scaled == -mx.inf,
            -mx.inf,
            scaled + _gumbel_at(bases, positions, ids.astype(mx.uint64)),
        )
        pick = mx.argmax(noisy, axis=-1, keepdims=True)
        return mx.take_along_axis(ids, pick, axis=-1)[:, 0]
    row = filter_logprobs(row, params)
    noisy = mx.where(
        row == -mx.inf, -mx.inf, row + gumbel_rows(bases, positions, row.shape[-1])
    )
    return mx.argmax(noisy, axis=-1)


# Nucleus / min-p / top-k filters only ever keep a few hundred tokens, so they are applied
# to the CAND_K most probable ones (one partition + a sort of CAND_K, not a sort of the whole
# vocabulary). A row whose kept set may spill past them is detected and the caller falls back
# to the dense path, so the result is the dense result.
CAND_K = 1024


def candidate_rows(logprobs: mx.array, params: Any):
    """``(ids, scaled)`` ``[N, K]``: the ``CAND_K`` best tokens per row, most probable first, with
    ``scaled`` = the filtered, temperature-scaled logprob (``-inf`` where filtered out), equal
    to ``filter_logprobs`` at those ids. ``None`` when the dense path must be used: no
    truncating filter, a small vocabulary, or a row whose kept set is not provably inside the
    candidates (one device sync)."""
    n_vocab = logprobs.shape[-1]
    top_k = int(params.top_k)
    by_top_k = 0 < top_k <= CAND_K
    top_p = float(params.top_p)
    if n_vocab <= 2 * CAND_K or not (by_top_k or top_p < 1.0 or params.min_p):
        return None
    row = logprobs.astype(mx.float32)
    idx = mx.argpartition(-row, CAND_K - 1, axis=-1)[:, :CAND_K]
    vals = mx.take_along_axis(row, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    first = mx.arange(CAND_K) == 0
    keep = mx.ones(vals.shape, dtype=mx.bool_)
    covered = mx.full((vals.shape[0],), by_top_k, dtype=mx.bool_)
    if top_p < 1.0:
        probs = mx.exp(vals - mx.logsumexp(row, axis=-1, keepdims=True))
        cum = mx.cumsum(probs, axis=-1)
        keep = keep & (((cum - probs) < max(top_p, 0.0)) | first)
        covered = covered | (cum[:, -1] >= max(top_p, 0.0))
    if params.min_p:
        floor = vals[:, :1] + math.log(params.min_p)
        keep = keep & (vals >= floor)
        covered = covered | (vals[:, -1] < floor[:, 0])
    if by_top_k:
        keep = keep & (mx.arange(CAND_K) < top_k)
    if not bool(mx.all(covered).item()):
        return None
    return idx, mx.where(keep, vals, -mx.inf) * (1 / params.temperature)


def top_p_filter(logprobs: mx.array, top_p: float) -> mx.array:
    """Nucleus filter that always keeps the most probable token.

    A token is kept when the probability mass of the tokens ranked above it is below
    ``top_p``, so the top token survives any ``top_p`` (``0`` keeps only it, which is greedy).
    Rows need not be normalized: the mass is taken against their own total.
    """
    row = logprobs.astype(mx.float32)
    probs = mx.exp(row - mx.logsumexp(row, axis=-1, keepdims=True))
    order = mx.argsort(-row, axis=-1)
    before = mx.cumsum(mx.take_along_axis(probs, order, axis=-1), axis=-1) - (
        mx.take_along_axis(probs, order, axis=-1)
    )
    keep_sorted = (before < top_p) | (mx.arange(row.shape[-1]) == 0)
    keep = mx.put_along_axis(
        mx.zeros(keep_sorted.shape, dtype=mx.bool_), order, keep_sorted, axis=-1
    )
    return mx.where(keep, row, -mx.inf)


def top_k_filter(logprobs: mx.array, top_k: int) -> mx.array:
    """Top-k filter; ``k <= 0`` or ``k >= vocab`` keeps every token (no truncation)."""
    if top_k <= 0 or top_k >= logprobs.shape[-1]:
        return logprobs
    from mlx_lm.sample_utils import apply_top_k

    return apply_top_k(logprobs, int(top_k))


def filter_logprobs(logprobs: mx.array, params: Any) -> mx.array:
    """top-p / min-p / top-k then temperature, in RowSampler's order (``[N, V]``)."""
    from mlx_lm.sample_utils import apply_min_p

    row = logprobs
    if params.top_p < 1.0:
        row = top_p_filter(row, max(float(params.top_p), 0.0))
    if params.min_p:
        row = apply_min_p(row, params.min_p)
    row = top_k_filter(row, params.top_k)
    return row * (1 / params.temperature)


def supports(params: Any) -> bool:
    """XTC and friends stay on the stateful sampler."""
    return not getattr(params, "xtc_probability", 0.0)


def tree_supported(sampler: Any, greedy: bool) -> bool:
    """Trees can verify greedy rows or explicitly position-keyed draws only."""
    if isinstance(sampler, KeyedSampler):
        return supports(sampler.params) and sampler.params.temperature > 0
    return bool(greedy)


def tree_penalty_counts(penalty, shape, window, vocab: int):
    """``[W, V]`` generated-token counts seen by each window row: the committed counts plus
    the window tokens on the row's root-to-row path (``GeneratedTokenPenalty.window_row``'s
    ``fed``), built on the GPU so a tree round needs no host loop."""
    paths = shape.path_table()  # [W, M] ancestors, padded with the row itself
    width, steps = int(paths.shape[0]), int(paths.shape[1])
    valid = (mx.arange(steps)[None] <= shape.depth_array()[:, None]).astype(mx.float32)
    tokens = mx.take(window.reshape(-1).astype(mx.int32), paths)
    valid = valid * ((tokens >= 0) & (tokens < vocab)).astype(mx.float32)
    tokens = mx.clip(tokens, 0, vocab - 1)
    counts = mx.broadcast_to(penalty._counts(vocab)[None], (width, vocab))
    rows = mx.broadcast_to(mx.arange(width)[:, None], tokens.shape)
    return counts.at[rows, tokens].add(valid)


def tree_target_rows(
    logits, shape, committed: int, sampler, request=None, penalty=None, window=None
):
    """Draw after every tree node, using depth rather than storage row order.

    The pending root token has already been emitted: its distribution predicts token
    ``committed``. Siblings at a depth share the same noise. This function also handles
    copy windows, represented as chains. It never advances the sampler's serial cursor.
    Return unfiltered target logprobs for accepted-token reporting. ``penalty`` (a
    ``GeneratedTokenPenalty``) is applied per row over ``window``'s tokens, with the
    arithmetic of the serial ``window_row`` (same elementwise ops on the same rows).
    """
    from .constrained_spec import normalize_rows

    if request is not None and request.logprobs:
        logits = logits.astype(mx.float32)
    if penalty is not None:
        counts = tree_penalty_counts(penalty, shape, window, int(logits.shape[-1]))
        logits = penalty._apply(counts[None], logits)
    probs = normalize_rows(logits)
    target = (
        sampler.sample_positions(probs, shape.depth_array() + committed)
        if sampler is not None
        else mx.argmax(probs, axis=-1)
    )
    return target, probs


def coupled_scores(logprobs, sampler, positions):
    """Proposal scores sharing the target draw's noise; no sampler cursor mutation.

    Filtering precedes perturbation, exactly as in sample_rows. Ranking these scores
    gives distinct sibling candidates, with the serial keyed draw ranked first.
    """
    row = filter_logprobs(logprobs.astype(mx.float32), sampler.params)
    if sampler.position_base:
        positions = positions + sampler.position_base
    noise = gumbel(sampler.seed, positions, row.shape[-1])
    return mx.where(row == -mx.inf, -mx.inf, row + noise)


def coupled_candidates(logprobs, sampler, positions):
    """``coupled_scores`` restricted to the candidates that can win: ``(ids, scores)`` ``[N, K]``
    (``scores`` ``-inf`` where filtered out), or ``None`` for the dense ``coupled_scores``."""
    cand = candidate_rows(logprobs.astype(mx.float32), sampler.params)
    if cand is None:
        return None
    ids, scaled = cand
    if sampler.position_base:
        positions = positions + sampler.position_base
    bases = mx.array(
        np.full((int(positions.size),), seed_base(sampler.seed), dtype=np.uint64)
    )
    noise = _gumbel_at(bases, positions, ids.astype(mx.uint64))
    return ids, mx.where(scaled == -mx.inf, -mx.inf, scaled + noise)


class KeyedSampler:
    """Sampler for one request: ``sample_target`` draws for explicit generation indices."""

    def __init__(self, params: Any, seed: int):
        self.params = params
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        # A promoted lane counts locally from its pending token; keep its draws
        # at the original request's absolute generation indices.
        self.position_base = 0
        self._next = 0
        from .utils.hardware import is_paravirtual_metal

        if is_paravirtual_metal():
            # Bind once: physical GPUs retain the original per-draw method.
            self.sample_positions = self._sample_positions_virtual  # type: ignore[method-assign]

    def _sample_positions_virtual(self, logprobs, positions):
        """Bound each VM command buffer; the 600 x 248320 graph can GPU-hang."""
        pos = (
            positions if isinstance(positions, mx.array) else mx.array(list(positions))
        )
        draws = []
        for start in range(0, int(pos.size), 64):
            # type(self): a subclass's own sample_positions (e.g. the qwen4 greedy reports) keeps its behaviour
            draw = type(self).sample_positions(
                self, logprobs[start : start + 64], pos[start : start + 64]
            )
            mx.eval(draw)  # submit before building the next graph
            draws.append(draw)
        return mx.concatenate(draws) if draws else mx.array([], dtype=mx.uint32)

    def sample_positions(self, logprobs: mx.array, positions) -> mx.array:
        """Tokens ``[N]`` for logprob rows ``[N, V]`` at generation indices ``positions``."""
        pos = (
            positions if isinstance(positions, mx.array) else mx.array(list(positions))
        )
        if self.position_base:
            pos = pos + self.position_base
        bases = mx.array(
            np.full((int(pos.size),), seed_base(self.seed), dtype=np.uint64)
        )
        return sample_rows(logprobs, self.params, bases, pos)

    def sample_target(self, logprobs, row_ids=None, positions=None):
        if positions is None:
            return self(logprobs)
        self._next = max(self._next, int(max(positions)) + 1)
        return self.sample_positions(logprobs, positions)

    def __call__(self, logprobs: mx.array) -> mx.array:
        # Fallback for callers that pass no position: successive calls advance the index.
        n = int(logprobs.shape[0])
        pos = list(range(self._next, self._next + n))
        self._next += n
        return self.sample_positions(logprobs, pos)
