"""VLM shared-batch top_logprobs without the upstream full-vocab argsort.

``top_logprobs_device`` orders like the text path: descending value, equal values by ASCENDING
token id (mlx-vlm's own ``argsort`` + reverse slice breaks ties by descending id), NaN above
everything. The VALUES must equal the upstream sort bit for bit; only the tie order differs.
The GenerationBatch hook in ``vlm_batch_runner`` must report that order.
"""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import text_logprobs, vlm_batch_runner
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
    gi, f = np.array(got_i), np.array(lp.astype(mx.float32))
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


# ---- the hook on the real upstream GenerationBatch --------------------------------------


class _Table:
    """Language model whose next-token logits are a fixed function of the input token."""

    def __init__(self, table):
        self.table = table

    def __call__(self, inputs, cache=None, **_):
        return self.table[inputs[:, 0]][:, None, :]


def _run(table, first, k, steps, rows_sampler=None):
    from mlx_vlm.generate import ar

    vlm_batch_runner._install_row_context()
    n = len(first)
    batch = ar.GenerationBatch(
        model=_Table(table),
        uids=list(range(n)),
        inputs=mx.array(first),
        prompt_cache=[],
        sampler=rows_sampler or vlm_batch_runner.RowSampler(),
        stop_criteria=lambda tok: False,
        max_tokens=[10**6] * n,
        top_logprobs_k=k,
    )
    batch.compute_logprobs = True
    out = []
    for _ in range(steps):
        out.append(batch.next())
    return out


@pytest.mark.parametrize("k", [1, 5])
def test_generation_batch_reports_the_upstream_top_logprobs(k):
    vocab, n = 2048, 3
    table = _rows(vocab, 21)[:3]  # rows 0..2: ties, -inf tail, fully masked
    table = mx.concatenate([table, mx.zeros((vocab - 3, vocab))], axis=0)
    table = table - mx.logsumexp(table, axis=-1, keepdims=True)
    first = [0, 1, 2]
    responses = _run(table, first, k, 4)
    # step s reports the tops of the distribution that produced the token emitted at s
    # (token 0 is the given input, so its tops are the prime buffers = None here).
    for s in range(1, 4):
        for i in range(n):
            tok_prev = responses[s - 1][i].token
            row = table[tok_prev][None]
            norm = row - mx.logsumexp(row, axis=-1, keepdims=True)
            want_i, want_v = upstream_top(norm, k)
            got = responses[s][i].top_logprobs
            assert got is not None
            assert [t for t, _ in got] == canonical_top(norm, k)[0].tolist()
            assert [v for _, v in got] == want_v[0].tolist()


def test_hook_does_not_touch_rows_without_top_logprobs(monkeypatch):
    vlm_batch_runner._install_row_context()  # its capability probe uses the helper once
    called = []
    monkeypatch.setattr(
        text_logprobs,
        "top_logprobs_device",
        lambda *a, **kw: called.append(1) or (_ for _ in ()).throw(AssertionError),
    )
    table = mx.zeros((64, 64))
    out = _run(table, [0, 1], 0, 2)
    assert not called
    assert all(r.top_logprobs is None for step in out for r in step)
