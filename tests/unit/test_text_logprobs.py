"""top_logprobs via argpartition vs the full-vocab argsort loop it replaced."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.text_logprobs import top_logprobs


def old_top(log_probs, k, token):
    """engine_fast.py before this change."""
    tok_lp = float(log_probs[token])
    k = min(k, log_probs.shape[0])
    top_k_idx = mx.argsort(-log_probs)[:k]
    return (
        tok_lp,
        [int(top_k_idx[j]) for j in range(k)],
        [float(log_probs[int(top_k_idx[j])]) for j in range(k)],
    )


def _log_softmax(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


@pytest.mark.parametrize("n", [37, 1000, 4099])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_random_logits_match_the_full_sort(n, k):
    rng = np.random.default_rng(n + k)
    for _ in range(10):
        lp = _log_softmax(mx.array(rng.standard_normal(n).astype(np.float32) * 3))
        tok = mx.array(int(rng.integers(0, n)))
        assert top_logprobs(lp, k, tok) == old_top(lp, k, tok)


@pytest.mark.parametrize("n,k", [(500, 20), (64, 10), (50, 1)])
def test_ties_keep_the_stable_argsort_order(n, k):
    rng = np.random.default_rng(1)
    for _ in range(20):
        # few distinct values -> ties inside the top-k and across its boundary
        raw = rng.integers(0, 6, size=n).astype(np.float32)
        lp = _log_softmax(mx.array(raw))
        tok = mx.array(int(rng.integers(0, n)))
        assert top_logprobs(lp, k, tok) == old_top(lp, k, tok)


def test_inf_nan_and_short_vocab():
    base = np.full(40, -np.inf, dtype=np.float32)
    base[[3, 9, 11]] = [-0.5, -0.5, -2.0]  # a grammar mask: most of the vocab is -inf
    lp = mx.array(base)
    tok = mx.array(9)
    assert top_logprobs(lp, 8, tok) == old_top(lp, 8, tok)
    assert top_logprobs(lp, 100, tok) == old_top(lp, 100, tok)  # k > vocab
    nan = mx.array(np.array([0.0, np.nan, -1.0, -2.0, -3.0], dtype=np.float32))
    got, want = top_logprobs(nan, 3, mx.array(2)), old_top(nan, 3, mx.array(2))
    assert got[1] == want[1] and [str(v) for v in got[2]] == [str(v) for v in want[2]]


def test_without_a_token_only_the_top_k_come_back():
    lp = _log_softmax(mx.array([0.1, 3.0, 2.0, -1.0]))
    tok_lp, ids, vals = top_logprobs(lp, 2)
    assert tok_lp is None and ids == [1, 2]
    assert vals == [float(lp[1]), float(lp[2])]


async def test_both_fast_paths_return_the_same_top_logprobs():
    from .tiny_text_engine import collect_stream, make_engine

    kw = dict(prompt="the quick brown fox ", max_tokens=6, temperature=0.0)
    gen = await make_engine()._generate_fast(logprobs=True, top_logprobs=5, **kw)
    _, last = await collect_stream(make_engine(), logprobs=True, top_logprobs=5, **kw)
    assert gen.error is None and len(gen.logprobs) == 6
    assert last.logprobs is not None
    for a, b in zip(gen.logprobs, last.logprobs, strict=True):
        assert a["token_id"] == b["token_id"]
        assert [t["token_id"] for t in a["top_logprobs"]] == [
            t["token_id"] for t in b["top_logprobs"]
        ]
        # greedy: the sampled token is the best of the top-k
        assert a["top_logprobs"][0]["token_id"] == a["token_id"]
