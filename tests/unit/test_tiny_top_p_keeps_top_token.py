"""A tiny top_p (or tiny temperature) must never empty the candidate set."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import engine_sampling
from yunshu_engine.batch_sampler import BatchSampler
from yunshu_engine.keyed_sampling import top_p_filter

TINY = [1e-12, 1e-9, 1e-6, 1e-3]


def _logits(seed=0, vocab=257, rows=1):
    return (
        np.random.default_rng(seed)
        .normal(scale=3, size=(rows, vocab))
        .astype(np.float32)
    )


@pytest.mark.parametrize("top_p", TINY)
@pytest.mark.parametrize("seed", range(6))
def test_batch_sampler_top_p_keeps_top_token(top_p, seed):
    raw = _logits(seed, rows=2)
    lp = raw - np.log(np.exp(raw).sum(-1, keepdims=True))
    out = BatchSampler()._apply_batch_top_p(mx.array(lp), [top_p, top_p])
    out = np.array(out)
    for r in range(2):
        assert np.isfinite(out[r]).any()
        assert np.isfinite(out[r, lp[r].argmax()])


@pytest.mark.parametrize("top_p", TINY)
def test_batch_sampler_top_p_peaked_row(top_p):
    row = np.full((1, 64), -30.0, dtype=np.float32)
    row[0, 5] = 0.0
    out = np.array(BatchSampler()._apply_batch_top_p(mx.array(row), [top_p]))
    assert np.isfinite(out[0, 5])


@pytest.mark.parametrize("top_p", TINY)
def test_keyed_top_p_keeps_top_token(top_p):
    lp = mx.array(_logits(1))
    out = np.array(top_p_filter(lp, top_p))
    assert np.isfinite(out[0, int(np.argmax(np.array(lp)))])


@pytest.mark.parametrize("top_p", TINY)
@pytest.mark.parametrize("temp", [1e-3, 0.05, 1.0])
def test_numpy_and_gpu_samplers_pick_valid_token(top_p, temp):
    raw = _logits(2, rows=1)
    lp = mx.array(raw - np.log(np.exp(raw).sum(-1, keepdims=True)))
    for build in (
        engine_sampling._build_noncached_sampler_text,
        engine_sampling._build_gpu_sampler_text,
    ):
        s = build(temp, top_p, 0, 0.0, 3)
        tok = int(np.array(s(lp)).reshape(-1)[0])
        assert 0 <= tok < raw.shape[-1]
        if top_p <= 1e-6:
            assert tok == int(raw.argmax())


@pytest.mark.parametrize("top_p", TINY)
def test_top_p_with_grammar_mask_keeps_allowed_token(top_p):
    """top_p after a grammar mask: the best ALLOWED token must survive."""

    raw = _logits(4, vocab=128)
    masked = raw.copy()
    allowed = [10, 20, 30]
    keep = np.zeros(128, dtype=bool)
    keep[allowed] = True
    masked[0, ~keep] = -np.inf
    lp = mx.array(masked) - mx.logsumexp(mx.array(masked), axis=-1, keepdims=True)
    for build in (
        engine_sampling._build_noncached_sampler_text,
        engine_sampling._build_gpu_sampler_text,
    ):
        tok = int(np.array(build(0.7, top_p, 0, 0.0, 1)(lp)).reshape(-1)[0])
        assert tok in allowed
    out = np.array(BatchSampler()._apply_batch_top_p(mx.array(masked), [top_p]))
    assert np.isfinite(out[0]).any()
    out = np.array(top_p_filter(lp, top_p))
    assert np.isfinite(out[0]).any()
