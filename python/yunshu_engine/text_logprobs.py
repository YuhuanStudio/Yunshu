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


def top_logprobs_rows(
    log_probs: mx.array, k: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Top-k ids/values per row of ``[N, V]`` ``log_probs`` in the mlx-vlm ``BatchGenerator``
    order: ``argsort`` ascending, last k reversed, i.e. descending value and, for equal
    values, DESCENDING token id (the opposite tie order of :func:`top_logprobs`).

    One batched ``argpartition`` and one ``tolist`` per array replace the full-vocab sort.
    Rows whose k-th value ties with outsiders (``>= min`` count != k, which includes ``-inf``
    tails) or that contain NaN are re-done with the original stable argsort, so every row
    equals what the sort would have produced.
    """
    n, vocab = log_probs.shape
    k = max(0, min(int(k), vocab))
    if k == 0 or n == 0:
        return [[] for _ in range(n)], [[] for _ in range(n)]
    idx = mx.argpartition(-log_probs, k - 1, axis=-1)[:, :k]
    vals = mx.take_along_axis(log_probs, idx, axis=-1)
    ambiguous = mx.logical_or(
        mx.sum(log_probs >= mx.min(vals, axis=-1, keepdims=True), axis=-1) != k,
        mx.any(mx.isnan(log_probs), axis=-1),
    )
    mx.eval(idx, vals, ambiguous)
    amb: list = ambiguous.tolist()
    ids, vs = idx.tolist(), vals.tolist()
    if any(amb):
        rows = [i for i, a in enumerate(amb) if a]
        sub = log_probs[mx.array(rows)]
        full = mx.argsort(sub, axis=-1)[:, -k:][:, ::-1].astype(mx.int32)
        full_v = mx.take_along_axis(sub, full, axis=-1)
        mx.eval(full, full_v)
        for r, fi, fv in zip(rows, full.tolist(), full_v.tolist(), strict=True):
            ids[r], vs[r] = fi, fv
            amb[r] = None  # already in final order
    out_i: list[list[int]] = []
    out_v: list[list[float]] = []
    for r in range(n):
        if amb[r] is None:
            out_i.append(ids[r])
            out_v.append(vs[r])
            continue
        order = sorted(range(k), key=lambda j: (-vs[r][j], -ids[r][j]))
        out_i.append([ids[r][j] for j in order])
        out_v.append([vs[r][j] for j in order])
    return out_i, out_v
