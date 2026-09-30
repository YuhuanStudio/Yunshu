import mlx.core as mx
import numpy as np

from yunshu_engine.keyed_sampling import KeyedSampler, gumbel
from yunshu_engine.vlm_batch_runner import RowParams


def _params(**kw):
    base = dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0)
    base.update(kw)
    return RowParams(**base)


def _logprobs(logits):
    x = mx.array(logits, dtype=mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def test_noise_is_a_pure_function_of_seed_position_id():
    a = gumbel(7, mx.array([3, 4]), 16)
    b = gumbel(7, mx.array([4]), 16)
    assert np.array_equal(np.array(a[1]), np.array(b[0]))
    assert not np.array_equal(np.array(a[0]), np.array(a[1]))
    c = gumbel(8, mx.array([3]), 16)
    assert not np.array_equal(np.array(a[0]), np.array(c[0]))


def test_draws_follow_the_distribution():
    logits = [2.0, 1.0, 0.0, -1.0, 3.0]
    n = 6000
    s = KeyedSampler(_params(), seed=11)
    lp = mx.broadcast_to(_logprobs(logits), (n, len(logits)))
    tok = np.array(s.sample_positions(lp, list(range(n))))
    freq = np.bincount(tok, minlength=len(logits)) / n
    p = np.exp(np.array(_logprobs(logits)))
    assert np.abs(freq - p).max() < 0.03


def test_same_token_alone_or_in_a_block():
    logits = np.random.default_rng(0).normal(size=(6, 64)).astype(np.float32)
    lp = _logprobs(logits)
    s = KeyedSampler(_params(top_k=8, top_p=0.9), seed=5)
    block = np.array(s.sample_positions(lp, [10, 11, 12, 13, 14, 15]))
    one = [int(s.sample_positions(lp[i : i + 1], [10 + i])[0]) for i in range(6)]
    assert block.tolist() == one


def test_top_k_is_respected():
    logits = np.linspace(0, 5, 32).astype(np.float32)
    lp = mx.broadcast_to(_logprobs(logits), (400, 32))
    s = KeyedSampler(_params(top_k=3), seed=1)
    tok = np.array(s.sample_positions(lp, list(range(400))))
    assert set(tok.tolist()) <= {29, 30, 31}


def test_fallback_call_advances_the_position():
    s = KeyedSampler(_params(), seed=3)
    lp = mx.broadcast_to(_logprobs([0.0] * 50), (1, 50))
    draws = {int(s(lp)[0]) for _ in range(30)}
    assert len(draws) > 5


def test_noise_is_finite_over_the_real_vocab():
    # Regression: 24-bit uniforms reached 1 - 2^-25, which float32 rounds to 1.0, so about
    # 1.5% of positions had a token with +inf noise (a stray token outside the filtered
    # support, or NaN against its -inf row). These seeds/positions hit it 6 times in 600.
    vocab = 248320
    for seed in (0, 1, 2):
        g = np.array(gumbel(seed, mx.arange(200), vocab))
        assert np.isfinite(g).all()


def test_draws_never_leave_the_filtered_support():
    vocab = 248320
    row = np.full((1, vocab), -np.inf, dtype=np.float32)
    row[0, :5] = 0.0
    lp = mx.broadcast_to(mx.array(row), (600, vocab))
    for seed in (0, 1, 2):
        s = KeyedSampler(_params(), seed=seed)
        tok = np.array(s.sample_positions(lp, list(range(600))))
        assert tok.max() < 5
