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


def gumbel(seed: int, positions: mx.array, vocab: int) -> mx.array:
    """Gumbel(0, 1) noise ``[N, vocab]`` keyed by (seed, positions[n], token id)."""
    pos = positions.astype(mx.uint64).reshape(-1, 1)
    ids = mx.arange(vocab, dtype=mx.uint64)[None, :]
    base = _mix(_u64((int(seed) & 0xFFFFFFFFFFFFFFFF) ^ _C0))
    x = _mix(base ^ (pos * _u64(_C3)))
    x = _mix(x ^ ids)
    # 23 random bits -> u in [2^-24, 1 - 2^-24], exactly representable in float32. (With 24
    # bits the top bucket is 1 - 2^-25, which float32 rounds to 1.0: -log(-log(1)) = +inf, a
    # token with infinite noise that wins even when filtered out, or NaN with its -inf row.)
    u = ((x >> _u64(41)).astype(mx.float32) + 0.5) * (1.0 / (1 << 23))
    return -mx.log(-mx.log(u))


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


class KeyedSampler:
    """Sampler for one request: ``sample_target`` draws for explicit generation indices."""

    def __init__(self, params: Any, seed: int):
        self.params = params
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self._next = 0

    def sample_positions(self, logprobs: mx.array, positions) -> mx.array:
        """Tokens ``[N]`` for logprob rows ``[N, V]`` at generation indices ``positions``."""
        pos = (
            positions if isinstance(positions, mx.array) else mx.array(list(positions))
        )
        row = filter_logprobs(logprobs.astype(mx.float32), self.params)
        noisy = mx.where(
            row == -mx.inf, -mx.inf, row + gumbel(self.seed, pos, row.shape[-1])
        )
        tokens = mx.argmax(noisy, axis=-1)
        return tokens

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
