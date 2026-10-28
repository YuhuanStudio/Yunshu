"""Dense-vector logits processors vs the per-token-id loops they replaced: same bits."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.text_logit_processors import (
    FrequencyPresencePenalty,
    LogitBias,
    TokenBan,
)

# ── the old implementations, verbatim (engine_fast.py before this change) ──


def old_freq_pres(fp, pp, n_prompt=1):
    st = {"counts": {}, "last_len": -1}

    def proc(tokens, logits):
        counts = st["counts"]
        last_len = int(st["last_len"])
        cur_len = len(tokens)
        if cur_len <= n_prompt:
            st["last_len"] = cur_len
            return logits
        if cur_len < last_len or last_len < n_prompt:
            counts = {}
            for t in tokens[n_prompt:]:
                counts[int(t)] = counts.get(int(t), 0) + 1
            st["counts"] = counts
        else:
            start = max(last_len, n_prompt)
            for t in tokens[start:]:
                counts[int(t)] = counts.get(int(t), 0) + 1
        st["last_len"] = cur_len
        for tid, cnt in counts.items():
            if fp != 0.0:
                logits[..., tid] = logits[..., tid] - fp * cnt
            if pp != 0.0 and cnt > 0:
                logits[..., tid] = logits[..., tid] - pp
        return logits

    return proc


def old_bias(biases):
    def proc(_tokens, logits):
        vocab = logits.shape[-1]
        for tid, bias in biases.items():
            if 0 <= tid < vocab:
                logits[..., tid] = logits[..., tid] + bias
        return logits

    return proc


def old_ban(ids, floor=None):
    def proc(tokens, logits):
        if floor is None or (len(tokens) - 1) < floor:
            vocab = logits.shape[-1]
            for tid in ids:
                if 0 <= tid < vocab:
                    logits[..., tid] = -float("inf")
        return logits

    return proc


V = 257
DTYPES = [mx.float32, mx.bfloat16, mx.float16]


def _logits(rng, dtype, shape):
    x = mx.array(rng.standard_normal(shape).astype(np.float32) * 4.0).astype(dtype)
    mx.eval(x)
    return x


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    an = np.array(a.astype(mx.float32))
    bn = np.array(b.astype(mx.float32))
    assert an.view(np.uint32).tolist() == bn.view(np.uint32).tolist()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, V), (V,)])
@pytest.mark.parametrize(
    "fp,pp",
    [(0.5, 0.0), (0.0, 1.5), (0.37, 0.81), (-2.0, 0.0), (0.0, -1.3), (1.9, -0.7)],
)
def test_frequency_presence_matches_the_loop(dtype, shape, fp, pp):
    rng = np.random.default_rng(7)
    old, new = old_freq_pres(fp, pp), FrequencyPresencePenalty(fp, pp)
    toks = [5]  # mlx-lm: [last_prompt_token, gen1, ...]
    for _step in range(60):
        base = _logits(rng, dtype, shape)
        a = old(mx.array(toks), mx.array(base))
        b = new(mx.array(toks), base)
        _same(a, b)
        # repeats are common so counts > 1 occur; ids near the ends of the vocab
        toks.append(int(rng.choice([3, 3, 17, 17, 17, V - 1, int(rng.integers(0, V))])))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_frequency_presence_rebuilds_after_a_rollback(dtype):
    rng = np.random.default_rng(3)
    old, new = old_freq_pres(0.6, 0.9), FrequencyPresencePenalty(0.6, 0.9)
    toks = [2, 4, 4, 9, 9, 9, 1, 4]
    for seq in (toks, toks[:5], toks[:5] + [4, 4, 6], toks[:2]):
        base = _logits(rng, dtype, (1, V))
        _same(old(mx.array(seq), mx.array(base)), new(mx.array(seq), base))


def test_frequency_presence_accepts_python_lists_and_ignores_the_prompt_token():
    new = FrequencyPresencePenalty(1.0, 1.0)
    logits = mx.zeros((1, 8))
    assert new([3], logits) is logits  # nothing generated yet
    out = new([3, 5, 5], logits)
    assert out[0, 5].item() == -3.0 and out[0, 3].item() == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, V), (V,)])
def test_logit_bias_matches_the_loop(dtype, shape):
    rng = np.random.default_rng(11)
    biases = {
        0: 2.5,
        7: -3.1,
        100: 0.3333333,
        V - 1: 100.0,
        V: 5.0,
        -1: 5.0,
        99999: 1.0,
    }
    old, new = old_bias(biases), LogitBias(biases)
    for _ in range(8):
        base = _logits(rng, dtype, shape)
        _same(old(None, mx.array(base)), new(None, base))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("floor", [None, 4])
def test_token_ban_matches_the_loop(dtype, floor):
    rng = np.random.default_rng(5)
    ids = [1, 2, 200, V, -3, 2]
    old, new = old_ban(ids, floor), TokenBan(ids, floor)
    for n in range(9):
        toks = mx.array([0] * (n + 1))
        base = _logits(rng, dtype, (1, V))
        _same(old(toks, mx.array(base)), new(toks, base))


def test_the_callers_logits_are_not_modified():
    base = mx.arange(8, dtype=mx.float32).reshape(1, 8)
    keep = np.array(base)
    FrequencyPresencePenalty(1.0, 1.0)([0, 3, 3], base)
    LogitBias({1: 9.0})(None, base)
    TokenBan([2])(None, base)
    assert np.array(base).tolist() == keep.tolist()


# ── wired into both text fast paths ──


async def test_fast_paths_apply_them_and_agree():
    from .tiny_text_engine import collect_stream, make_engine

    kw = dict(
        prompt="the quick brown fox ",
        max_tokens=16,
        temperature=0.0,
        frequency_penalty=0.7,
        presence_penalty=0.5,
        suppress_tokens=[3, 4, 5],
        logit_bias={9: 2.0},
    )
    plain = await make_engine()._generate_fast(
        prompt=kw["prompt"], max_tokens=16, temperature=0.0
    )
    gen = await make_engine()._generate_fast(**kw)
    text, last = await collect_stream(make_engine(), **kw)
    assert gen.error is None and last.error is None
    assert gen.text == text and gen.completion_tokens == last.completion_tokens
    assert gen.text != plain.text  # the processors changed the greedy path
