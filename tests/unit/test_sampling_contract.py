"""Sampling contract: top_p / top_k edge cases and path-independent seeded draws."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import keyed_sampling
from yunshu_engine import vlm_batch_runner as vbr
from yunshu_engine.keyed_sampling import KeyedSampler, filter_logprobs
from yunshu_engine.vlm_batch_runner import RowParams


def _params(**kw):
    base = dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0)
    base.update(kw)
    return RowParams(**base)


def _lp(logits, dtype=mx.float32):
    x = mx.array(logits, dtype=mx.float32)
    return (x - mx.logsumexp(x, axis=-1, keepdims=True)).astype(dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("top_p", [0.0, 1e-10, 1e-6, 0.5, 0.95, 1.0])
def test_top_p_always_keeps_a_finite_candidate(top_p, dtype):
    lp = _lp([[0.7, 0.2, 0.1], [1.0, 1.0, 1.0]], dtype)
    out = np.array(filter_logprobs(lp.astype(mx.float32), _params(top_p=top_p)))
    assert np.isfinite(out).any(axis=-1).all()
    assert np.isfinite(out[0, 0])  # the most probable token survives


def test_top_p_tiny_keeps_only_the_top_token():
    lp = _lp([[0.1, 3.0, 0.5, 0.2]])
    out = np.array(filter_logprobs(lp, _params(top_p=1e-10)))
    assert np.isfinite(out).tolist() == [[False, True, False, False]]


def test_top_p_unnormalized_logprobs():
    lp = _lp([[0.1, 3.0, 0.5, 0.2]]) - 5.0  # sums to far less than 1
    out = np.array(filter_logprobs(lp, _params(top_p=0.5)))
    assert np.isfinite(out[0, 1]) and np.isfinite(out).sum() >= 1


def test_top_p_statistics_match_nucleus():
    logits = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = np.exp(np.array(_lp(logits)))
    order = np.argsort(-p)
    keep = set(order[: int(np.searchsorted(np.cumsum(p[order]), 0.8)) + 1].tolist())
    n = 4000
    s = KeyedSampler(_params(top_p=0.8), seed=3)
    tok = np.array(s.sample_positions(mx.broadcast_to(_lp(logits), (n, 5)), range(n)))
    assert set(tok.tolist()) == keep


@pytest.mark.parametrize("k", [0, 1, 31, 32, 33, 10**9])
def test_top_k_at_or_above_vocab_is_a_noop(k):
    lp = _lp(np.linspace(0, 3, 32)[None])
    out = np.array(filter_logprobs(lp, _params(top_k=k)))
    assert np.isfinite(out).sum() == (1 if k == 1 else min(k, 32) if k else 32)


def test_row_sampler_seeded_stream_is_the_keyed_stream():
    """Same seed, same tokens whether a row is sampled by the shared batch or the lane."""
    rng = np.random.default_rng(0)
    steps = [_lp(rng.normal(size=(1, 64)).astype(np.float32)) for _ in range(12)]
    params = _params(top_p=0.9, top_k=20, temperature=0.8)
    row = vbr.RowSampler()
    row.add(5, RowParams(0.8, 0.9, 20, 0.0, 7))
    vbr._STEP_UIDS = [5]
    try:
        got = [int(row(lp)[0]) for lp in steps]
    finally:
        vbr._STEP_UIDS = None
    keyed = KeyedSampler(params, seed=7)
    want = [int(keyed.sample_positions(lp, [i])[0]) for i, lp in enumerate(steps)]
    assert got == want


def test_row_sampler_stream_ignores_batch_neighbours():
    rng = np.random.default_rng(1)
    a = [_lp(rng.normal(size=(1, 64)).astype(np.float32)) for _ in range(8)]
    b = [_lp(rng.normal(size=(1, 64)).astype(np.float32)) for _ in range(8)]

    def solo():
        s = vbr.RowSampler()
        s.add(1, RowParams(1.0, 1.0, 0, 0.0, 9))
        vbr._STEP_UIDS = [1]
        try:
            return [int(s(x)[0]) for x in a]
        finally:
            vbr._STEP_UIDS = None

    def mixed():
        s = vbr.RowSampler()
        s.add(1, RowParams(1.0, 1.0, 0, 0.0, 9))
        s.add(2, RowParams(1.0, 1.0, 0, 0.0, 4))
        out = []
        for x, y in zip(a, b, strict=True):
            vbr._STEP_UIDS = [2, 1]
            try:
                out.append(int(s(mx.concatenate([y, x]))[1]))
            finally:
                vbr._STEP_UIDS = None
        return out

    assert solo() == mixed()


def test_xtc_stays_on_the_stateful_sampler():
    assert not keyed_sampling.supports(RowParams(1.0, 1.0, 0, 0.0, 1, 0.5, 0.1))


# ── R23: one per-choice seed derivation for every route ──────────────────────────────
import pathlib  # noqa: E402

from yunshu_gateway.routers.chat import _per_choice_seed  # noqa: E402

_ROUTERS = pathlib.Path(__file__).resolve().parents[2] / "python/yunshu_gateway/routers"


@pytest.mark.parametrize("seed", [0, 7, -1, -(2**63), 2**63 - 1])
def test_choice_zero_keeps_the_callers_seed(seed):
    # n=1 and choice 0 of an n>1 request must draw the same stream.
    assert _per_choice_seed(seed, 0) == seed


@pytest.mark.parametrize("seed", [0, -1, 2**63 - 2, 2**63 - 1])
def test_choice_seeds_stay_signed_64_and_distinct(seed):
    seeds = [_per_choice_seed(seed, i) for i in range(8)]
    assert len(set(seeds)) == 8
    assert all(-(2**63) <= s < 2**63 for s in seeds)


def test_every_route_derives_choice_seeds_through_the_shared_helper():
    for name in ("chat.py", "completions.py", "responses.py"):
        src = (_ROUTERS / name).read_text()
        assert "req.seed +" not in src, name
