"""VLM shared-batch top_logprobs without the upstream full-vocab argsort.

``top_logprobs_device`` orders like the text path: descending value, equal values by ASCENDING
token id (mlx-vlm's own ``argsort`` + reverse slice breaks ties by descending id), NaN above
everything. The VALUES must equal the upstream sort bit for bit; only the tie order differs.
The round driver reports through the same function (``driver._report``); the GenerationBatch hook
that once did it for the stock runner is gone.
"""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import text_logprobs
from yunshu_engine.text_logprobs import top_logprobs_device


def upstream_top(lp, k):
    """mlx_vlm/generate/ar.py GenerationBatch._step, verbatim."""
    sort_idx = mx.argsort(lp, axis=-1)
    top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
    return top_idx, mx.take_along_axis(lp, top_idx, axis=-1)


def canonical_top(lp, k):
    """Descending value, equal values by ascending id, NaN first (numpy, whole vocabulary)."""
    a = np.array(lp.astype(mx.float32))
    ids = np.empty((a.shape[0], min(k, a.shape[1])), dtype=np.int32)
    for r, row in enumerate(a):
        nan = np.isnan(row)
        order = np.lexsort((np.arange(row.size), -np.where(nan, 0, row), ~nan))
        ids[r] = order[: ids.shape[1]]
    return ids


def _same(lp, k):
    want_i, want_v = upstream_top(lp, k)
    got_i, got_v = top_logprobs_device(lp, k)
    mx.eval(want_i, want_v, got_i, got_v)
    assert got_i.dtype == want_i.dtype and got_v.dtype == want_v.dtype
    assert np.array_equal(np.array(got_i), canonical_top(lp, k))
    wv, gv = np.array(want_v.astype(mx.float32)), np.array(got_v.astype(mx.float32))
    # same values (NaN equal to NaN; -0.0 == 0.0 -- which of the two zeros sits at a tie
    # position follows the id order, so their sign bit may differ)
    assert np.array_equal(wv, gv, equal_nan=True)
    # within a run of equal values the ids ascend
    gi = np.array(got_i)
    for r in range(gi.shape[0]):
        v = np.array(got_v.astype(mx.float32))[r]
        for j in range(1, gi.shape[1]):
            if v[j] == v[j - 1] or (np.isnan(v[j]) and np.isnan(v[j - 1])):
                assert gi[r, j] > gi[r, j - 1]


def _rows(vocab, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, vocab)).astype(np.float32) * 3
    a[0, ::7] = np.round(a[0, ::7])  # many ties, also at the top-k boundary
    a[1, :100] = -np.inf  # constrained tail
    a[2, :] = -np.inf  # fully masked row
    a[2, 42] = 0.0
    a[3, 5] = a[3, 9] = np.nan  # NaN row
    a[4, :] = 1.0  # everything tied
    a[5, 11] = a[5, vocab // 2] = a[5, 3] = a[5].max() + 1  # tie at the very top
    a[5, 20] = -0.0
    a[5, 21] = 0.0
    return mx.array(a)


@pytest.mark.parametrize("vocab", [3001, 152064, 248320])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_bit_identical_to_the_upstream_sort(vocab, k):
    _same(_rows(vocab, vocab + k), k)


def test_small_and_oversized_k():
    lp = _rows(50, 3)
    _same(lp, 1)
    _same(lp, 50)
    got_i, _ = top_logprobs_device(
        lp, 99
    )  # clipped to the vocabulary like the sort slice
    assert got_i.shape == (6, 50)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_low_precision_logprobs(dtype):
    lp = (_rows(4096, 9) * 0.5).astype(dtype)
    want_i, want_v = upstream_top(lp, 5)
    got_i, got_v = top_logprobs_device(lp, 5)
    assert np.array_equal(np.array(got_i), canonical_top(lp, 5))
    assert got_v.dtype == dtype
    assert np.array_equal(
        np.array(want_v.astype(mx.float32)).view(np.uint32),
        np.array(got_v.astype(mx.float32)).view(np.uint32),
    )


def test_batched_row_equals_single_row():
    lp = _rows(20000, 4)
    batch_i, batch_v = top_logprobs_device(lp, 7)
    for r in range(lp.shape[0]):
        one_i, one_v = top_logprobs_device(lp[r : r + 1], 7)
        mx.eval(one_i, one_v, batch_i, batch_v)
        assert np.array_equal(np.array(one_i[0]), np.array(batch_i[r]))
        assert np.array_equal(
            np.array(one_v[0]).view(np.uint32), np.array(batch_v[r]).view(np.uint32)
        )


def test_tie_order_is_ascending_id_and_values_match_upstream():
    lp = mx.array([[0.0, 1.0, 1.0, 1.0, -2.0, 1.0]])
    ids, vals = top_logprobs_device(lp, 3)
    up_i, up_v = upstream_top(lp, 3)
    assert ids.tolist() == [[1, 2, 3]]  # upstream: [5, 3, 2]
    assert up_i.tolist() == [[5, 3, 2]]
    assert vals.tolist() == up_v.tolist()


def test_probe_reports_support():
    assert text_logprobs.device_topk_supported()
