"""The candidate-set keyed sampler equals the dense one (same draws, same coupled scores)."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import keyed_sampling as ks
from yunshu_engine.keyed_sampling import KeyedSampler
from yunshu_engine.vlm_batch_runner import RowParams

V = 5000


def _params(**kw):
    base = dict(temperature=0.7, top_p=1.0, top_k=0, min_p=0.0)
    base.update(kw)
    return RowParams(**base)


def _lp(seed, rows=6, scale=4.0):
    x = mx.array(
        np.random.default_rng(seed)
        .normal(scale=scale, size=(rows, V))
        .astype(np.float32)
    )
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _dense(lp, params, bases, pos):
    row = ks.filter_logprobs(lp, params)
    noisy = mx.where(row == -mx.inf, -mx.inf, row + ks.gumbel_rows(bases, pos, V))
    return mx.argmax(noisy, axis=-1)


def _bases(n, seed=1234):
    return mx.array(np.full((n,), ks.seed_base(seed), dtype=np.uint64))


CASES = [
    dict(top_p=0.95),
    dict(top_p=0.5, temperature=1.3),
    dict(top_k=40),
    dict(top_k=40, top_p=0.9),
    dict(min_p=0.05),
    dict(min_p=0.02, top_p=0.95, top_k=200),
    dict(top_p=0.0),
]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("seed", range(3))
def test_sample_rows_matches_dense(kw, seed):
    params = _params(**kw)
    lp = _lp(seed)
    pos = mx.arange(lp.shape[0]) + 17 * seed
    assert ks.candidate_rows(lp, params) is not None
    got = ks.sample_rows(lp, params, _bases(lp.shape[0]), pos)
    assert got.tolist() == _dense(lp, params, _bases(lp.shape[0]), pos).tolist()


def test_wide_nucleus_falls_back_to_dense():
    # a near-flat row: the nucleus holds far more than CAND_K tokens
    params = _params(top_p=0.95, temperature=1.0)
    lp = _lp(3, scale=0.05)
    assert ks.candidate_rows(lp, params) is None
    pos = mx.arange(lp.shape[0])
    got = ks.sample_rows(lp, params, _bases(lp.shape[0]), pos)
    assert got.tolist() == _dense(lp, params, _bases(lp.shape[0]), pos).tolist()


def test_no_truncating_filter_is_dense():
    assert ks.candidate_rows(_lp(0), _params()) is None


def test_batch_rows_are_independent():
    params = _params(top_p=0.95)
    lp = _lp(5, rows=8)
    pos = mx.arange(8) + 3
    both = ks.sample_rows(lp, params, _bases(8), pos).tolist()
    one = [
        int(ks.sample_rows(lp[i : i + 1], params, _bases(1), pos[i : i + 1])[0])
        for i in range(8)
    ]
    assert both == one


@pytest.mark.parametrize("kw", [dict(top_p=0.95), dict(top_k=40), dict(min_p=0.05)])
def test_coupled_candidates_match_coupled_scores(kw):
    sampler = KeyedSampler(_params(**kw), 99)
    lp = _lp(7, rows=5)
    pos = mx.arange(5) + 11
    ids, sc = ks.coupled_candidates(lp, sampler, pos)
    dense = ks.coupled_scores(lp, sampler, pos)
    assert mx.array_equal(mx.take_along_axis(dense, ids, axis=-1), sc).item()
    # every finite dense score is a candidate: the top of both agree, and counts match
    assert (
        (mx.sum(dense > -mx.inf, axis=-1) == mx.sum(sc > -mx.inf, axis=-1)).all().item()
    )
    assert (
        mx.argmax(dense, axis=-1).tolist()
        == mx.take_along_axis(ids, mx.argmax(sc, axis=-1, keepdims=True), axis=-1)[
            :, 0
        ].tolist()
    )


def test_keyed_draw_is_position_keyed():
    sampler = KeyedSampler(_params(top_p=0.95), 5)
    lp = _lp(9, rows=4)
    a = sampler.sample_positions(lp, [3, 4, 5, 6]).tolist()
    b = sampler.sample_positions(lp[1:3], [4, 5]).tolist()
    assert a[1:3] == b
