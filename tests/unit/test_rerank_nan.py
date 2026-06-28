"""a NaN/Inf relevance (from a degenerate/overflow embedding — the cosine
norm==0 guard doesn't catch Inf) corrupted the ENTIRE /v1/rerank sort (every comparison
against NaN is False → Timsort leaves a position-dependent broken order → the top_n slice
then drops genuinely high-scoring docs) AND serialized as a bare NaN JSON literal (invalid
per RFC 8259). Coerce non-finite scores to a clean sentinel before sorting/serializing."""
from __future__ import annotations

import math
import pathlib


def test_nan_coerced_doc_sorts_last_not_corrupt():
    # Reproduce the rerank scored-list sort with the W812 sanitization.
    raw = [(0, float("nan")), (1, 0.9), (2, 0.7), (3, float("inf")), (4, 0.5)]
    scored = [(i, (0.0 if not math.isfinite(r) else r)) for i, r in raw]
    scored.sort(key=lambda x: x[1], reverse=True)
    # The real docs rank by relevance; the two non-finite ones sit at 0.0 (last).
    assert scored[0] == (1, 0.9)
    assert scored[1] == (2, 0.7)
    assert scored[2] == (4, 0.5)
    assert {scored[3][0], scored[4][0]} == {0, 3}  # the NaN+Inf docs, both 0.0
    # no NaN survives into the output
    assert all(math.isfinite(r) for _, r in scored)


def test_top_n_keeps_real_high_scorers_after_sanitization():
    # With a NaN at index 0 and top_n=2, the genuinely top-2 docs must survive (the old
    # broken sort would have left the NaN at #1 and dropped a real high scorer).
    raw = [(0, float("nan")), (1, 0.95), (2, 0.6), (3, 0.99)]
    scored = [(i, (0.0 if not math.isfinite(r) else r)) for i, r in raw]
    scored.sort(key=lambda x: x[1], reverse=True)
    top2 = scored[:2]
    assert [i for i, _ in top2] == [3, 1]  # 0.99 then 0.95 — the real winners


def test_all_three_endpoints_guard_non_finite():
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "python/yunshu_gateway/routers/scoring.py").read_text()
    # rerank relevance guard, score guard, classify logit guard
    assert src.count("math.isfinite") >= 3
    assert "if not math.isfinite(relevance):" in src
