import random

import pytest

from yunshu_engine.copy_drafter import CopyConfig, CopyDrafter


def _pair(ngram, **kw):
    cfg = dict(ngram=ngram, min_match=max(ngram, 6))
    cfg.update(kw)
    vec = CopyDrafter(CopyConfig(**cfg))
    ref = CopyDrafter(CopyConfig(**cfg))
    ref._vec = False
    return vec, ref


def _check(vec, ref, width):
    assert vec.lookup(width) == ref.lookup(width)
    assert vec.last_match == ref.last_match


@pytest.mark.parametrize("ngram", [1, 2, 3, 4])
def test_randomized_equivalence(ngram):
    rng = random.Random(ngram)
    for _ in range(60):
        vocab = rng.choice([2, 3, 5, 12])
        vec, ref = _pair(ngram, max_candidates=rng.choice([1, 4, 24, 256]))
        for _ in range(rng.randint(1, 12)):
            toks = [rng.randrange(vocab) for _ in range(rng.randint(0, 80))]
            vec.extend(toks)
            ref.extend(toks)
            _check(vec, ref, rng.choice([1, 4, 7, 16]))
        assert vec._vec == (ngram <= 3)


def test_capacity_growth_and_repeats():
    vec, ref = _pair(3)
    base = list(range(50))
    for _ in range(100):
        vec.extend(base)
        ref.extend(base)
    _check(vec, ref, 16)
    assert len(vec._tok) >= 5000
    assert vec.ctx == ref.ctx


def test_fallback_large_token_and_big_ngram():
    vec, ref = _pair(3)
    vec.extend([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3])
    ref.extend([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3])
    _check(vec, ref, 8)
    assert vec._vec
    big = [2**21 + 7, 9, 8] * 4
    vec.extend(big)
    ref.extend(big)
    _check(vec, ref, 8)
    assert not vec._vec
    vec.extend(big)
    ref.extend(big)
    _check(vec, ref, 8)

    d2 = CopyDrafter(CopyConfig(ngram=4, min_match=6))
    assert not d2._vec
    seq = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6]
    d2.extend(seq)
    assert d2.lookup(3) == [7, 1, 2]
