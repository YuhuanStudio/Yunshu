"""Spec-lane top_logprobs (SpecRequest.reports) equals the full-vocab argsort order."""

import mlx.core as mx
import pytest

from yunshu_engine.constrained_spec import SpecRequest
from yunshu_engine.text_logprobs import top_logprobs_rows


def _lsm(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _ref(lp, k):
    """Descending value, equal values by ascending id (NaN first)."""
    import numpy as np

    a = np.array(lp)
    ids = []
    for row in a:
        nan = np.isnan(row)
        order = np.lexsort((np.arange(row.size), -np.where(nan, 0, row), ~nan))
        ids.append(order[:k].astype(int).tolist())
    top = mx.array(ids)
    return ids, mx.take_along_axis(lp, top, -1).tolist()


@pytest.mark.parametrize("k", [1, 3, 5, 20])
def test_random_rows(k):
    mx.random.seed(1)
    lp = _lsm(mx.random.normal((6, 2048)) * 3)
    got, want = top_logprobs_rows(lp, k), _ref(lp, k)
    assert got == want


def test_ties_inf_and_nan_rows():
    mx.random.seed(5)
    rows = _lsm(mx.random.normal((5, 300))).tolist()
    rows[0][10] = rows[0][200] = rows[0][3] = 5.0  # tie at the top
    rows[1][7] = rows[1][8] = rows[1][9] = rows[1][11] = (
        4.0  # tie across the k boundary
    )
    rows[2] = [float("-inf")] * 300  # constrained to one token
    rows[2][42] = 0.0
    lp = mx.array(rows)
    for k in (1, 2, 4, 6):
        assert top_logprobs_rows(lp, k) == _ref(lp, k)
    # NaN rows take the same sort the old code ran (NaN != NaN, so compare ids only).
    rows[3][17] = float("nan")
    lp = mx.array(rows)
    assert top_logprobs_rows(lp, 4)[0] == _ref(lp, 4)[0]


def test_spec_request_reports_unchanged():
    mx.random.seed(2)
    probs = _lsm(mx.random.normal((4, 512)))
    req = SpecRequest(logprobs=True, top_logprobs=5)
    toks = [3, 9, 100, 7]
    got = req.reports(probs, toks)
    ids, vals = _ref(probs, 5)
    lps = mx.take_along_axis(probs, mx.array(toks)[:, None], -1).reshape(-1).tolist()
    for g, t, lp, i, v in zip(got, toks, lps, ids, vals, strict=True):
        assert g == (t, lp, list(zip(i, v, strict=True)))
