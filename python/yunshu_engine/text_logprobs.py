"""Top-k log-probabilities for the text fast path without a full-vocab sort.

The loop it replaces did ``argsort(-log_probs)`` over the whole vocabulary (150-250k) and
then ``int(idx[j])`` + ``float(log_probs[idx])`` per rank: ~2k+2 device syncs per token.
Here the k best come from one ``argpartition`` and leave the device in one ``tolist()`` each.

Order contract (same as the stable argsort it replaces): descending log-prob, equal
values by ascending token id. ``argpartition`` may pick an arbitrary subset of values that
tie with the k-th one and is undefined with NaN, so those two cases (counted on-device in
the same evaluation) take the original full-sort path and stay exactly what they were.
"""

from __future__ import annotations

import mlx.core as mx


def top_logprobs(
    log_probs: mx.array, k: int, token=None
) -> tuple[float | None, list[int], list[float]]:
    """(log_probs[token] or None, top-k ids, top-k values) of a 1-D ``log_probs``."""
    n = log_probs.shape[-1]
    k = max(0, min(int(k), n))
    tok = None if token is None else log_probs[token]
    if k == 0:
        return (None if tok is None else float(tok)), [], []
    idx = mx.argpartition(-log_probs, k - 1)[:k]
    vals = log_probs[idx]
    # entries >= the k-th best: exactly k unless the k-th value ties with outsiders
    ambiguous = mx.logical_or(
        mx.sum(log_probs >= mx.min(vals)) != k, mx.any(mx.isnan(log_probs))
    )
    mx.eval(idx, vals, ambiguous, *(() if tok is None else (tok,)))
    tok_lp = None if tok is None else tok.item()
    if bool(ambiguous.item()):
        idx = mx.argsort(-log_probs)[:k]
        vals = log_probs[idx]
        mx.eval(idx, vals)
        return tok_lp, idx.tolist(), vals.tolist()
    ids, vs = idx.tolist(), vals.tolist()
    order = sorted(range(k), key=lambda j: (-vs[j], ids[j]))
    return tok_lp, [ids[j] for j in order], [vs[j] for j in order]
