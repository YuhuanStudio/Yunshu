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

from typing import cast

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
    tok_lp = None if tok is None else cast(float, tok.item())
    if bool(ambiguous.item()):
        idx = mx.argsort(-log_probs)[:k]
        vals = log_probs[idx]
        mx.eval(idx, vals)
        return tok_lp, cast(list[int], idx.tolist()), cast(list[float], vals.tolist())
    ids, vs = cast(list[int], idx.tolist()), cast(list[float], vals.tolist())
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
    amb = cast(list, ambiguous.tolist())
    ids = cast(list[list[int]], idx.tolist())
    vs = cast(list[list[float]], vals.tolist())
    if any(amb):
        rows = [i for i, a in enumerate(amb) if a]
        sub = log_probs[mx.array(rows)]
        full = mx.argsort(sub, axis=-1)[:, -k:][:, ::-1].astype(mx.int32)
        full_v = mx.take_along_axis(sub, full, axis=-1)
        mx.eval(full, full_v)
        for r, fi, fv in zip(
            rows,
            cast(list[list[int]], full.tolist()),
            cast(list[list[float]], full_v.tolist()),
            strict=True,
        ):
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


def _orderable_u32(x: mx.array) -> mx.array:
    """float32 -> uint32 whose unsigned order is the sort order ``mx.argsort`` gives:
    -inf < finite < +inf < NaN (every NaN equal), -0.0 == +0.0."""
    bits = (x + 0.0).view(mx.uint32)  # +0.0 folds -0.0 into +0.0
    key = mx.where((bits >> 31) == 1, ~bits, bits | mx.array(0x80000000, mx.uint32))
    return mx.where(mx.isnan(x), mx.array(0xFFFFFFFF, mx.uint32), key)


def top_logprobs_device(log_probs: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Top-k ``(ids int32 [N, k], values [N, k])`` of ``[N, V]`` ``log_probs``, lazily.

    Same result, bit for bit, as the mlx-vlm ``GenerationBatch`` code it replaces
    (``argsort`` ascending, last k reversed: descending value, equal values by DESCENDING
    token id; NaN ranks above everything, -inf tails and ties included) but without the
    full-vocab sort and without a host sync. Each element becomes one uint64 key
    ``order(value) << 32 | token_id``; the keys are unique, so a single ``argpartition``
    selects exactly the right k and a k-wide sort orders them -- no ambiguity to detect.
    Rows are independent, so a batched row equals the same row alone. Nothing is
    evaluated here: the caller keeps the double-buffered decode pipelining.
    """
    n, vocab = log_probs.shape
    k = max(1, min(int(k), vocab))
    key = (
        _orderable_u32(log_probs.astype(mx.float32)).astype(mx.uint64) << 32
    ) | mx.arange(vocab, dtype=mx.uint64)[None]
    inv = ~key  # smallest inverted key == largest key
    part = mx.argpartition(inv, k - 1, axis=-1)[:, :k]
    order = mx.argsort(mx.take_along_axis(inv, part, axis=-1), axis=-1)
    idx = mx.take_along_axis(part, order, axis=-1).astype(mx.int32)
    return idx, mx.take_along_axis(log_probs, idx, axis=-1)


def logprob_entry(log_probs: mx.array, token: int, top_k: int | None) -> dict:
    """One per-token logprob entry in the text path's shape, from a 1-D ``log_probs``
    that is already normalised (the engine loop gets it from the BatchGenerator).

    NaN is reported as -100.0 (as ``_generate_fast`` does); with ``top_k`` the entry also
    carries ``top_logprobs`` from :func:`top_logprobs` (no full-vocabulary sort).
    """
    lp = log_probs.astype(mx.float32)
    lp = mx.where(mx.isnan(lp), mx.array(-100.0, dtype=lp.dtype), lp)
    if top_k and top_k > 0:
        tok_lp, ids, vals = top_logprobs(lp, top_k, token)
        return {
            "token_id": int(token),
            "logprob": tok_lp,
            "top_logprobs": [
                {"token_id": i, "logprob": v} for i, v in zip(ids, vals, strict=True)
            ],
        }
    return {"token_id": int(token), "logprob": float(lp[token])}


def device_topk_supported() -> bool:
    """The uint64 argpartition/shift path works on the current MLX device."""
    try:
        i, v = top_logprobs_device(mx.array([[0.0, 1.0, 1.0, float("nan")]]), 2)
        mx.eval(i, v)
        return i.tolist() == [[3, 2]]
    except Exception:  # noqa: BLE001
        return False
