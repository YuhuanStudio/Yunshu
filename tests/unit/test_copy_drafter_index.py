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


def _longest_before(drafter, ends, size):
    """The pre-optimization ``CopyDrafter._longest`` (one [C, reach] comparison), verbatim."""
    import numpy as np

    reach = drafter.cfg.max_extension
    tok = drafter._tok
    steps = np.arange(reach)
    back = ends[:, None] - 1 - steps[None, :]  # [C, reach]
    tail = size - 1 - steps
    valid = (back >= 0) & (tail >= 0)[None, :]
    same = valid & (tok[np.maximum(back, 0)] == tok[np.maximum(tail, 0)][None, :])
    lengths = np.cumprod(same, axis=1).sum(axis=1)
    pick = len(ends) - 1 - int(np.argmax(lengths[::-1]))
    best = int(lengths[pick])
    return (int(ends[pick]), best) if best > 0 else (-1, 0)


@pytest.mark.parametrize("reach", [3, 8, 9, 20, 64])
def test_longest_matches_the_full_comparison(reach):
    """Comparing the first steps for every candidate and only the survivors further gives the
    same (end, length), ties to the most recent, as comparing the whole reach at once."""
    import numpy as np

    rng = random.Random(reach)
    for _ in range(80):
        drafter = CopyDrafter(CopyConfig(max_extension=reach, max_candidates=64))
        vocab = rng.choice([2, 3, 6])
        base = [rng.randrange(vocab) for _ in range(rng.randint(5, 40))]
        ctx = []
        for _ in range(rng.randint(2, 12)):
            ctx += base[: rng.randint(1, len(base))] + [rng.randrange(vocab)]
        ctx += base[: rng.randint(3, len(base))]
        drafter.extend(ctx)
        drafter._sync_index()
        size = len(ctx)
        n = drafter.cfg.ngram
        ends = np.flatnonzero(drafter._keys[n:size] == drafter._keys[size]) + n
        ends = ends[ends < size]
        if not len(ends):
            continue
        assert drafter._longest(ends, size) == _longest_before(drafter, ends, size)


def test_second_lookup_of_the_same_tail_reuses_the_search():
    """A graft lookup after the round's draft (another width, same context) must return what a
    fresh drafter returns, and any extend must drop the remembered search."""
    rng = random.Random(5)
    ctx = [rng.randrange(4) for _ in range(300)]
    kept, fresh = CopyDrafter(max_draft=31), CopyDrafter(max_draft=31)
    kept.extend(ctx)
    fresh.extend(ctx)
    for width in (15, 3, 31, 1, 15):
        fresh._memo_key = None  # never reuse
        assert kept.lookup(width) == fresh.lookup(width)
        assert kept.last_match == fresh.last_match
    searches = []
    original = kept._search
    kept._search = lambda size: searches.append(size) or original(size)
    kept.lookup(7)
    kept.lookup(9)
    assert searches == []  # already searched at this length
    kept.extend([1])
    fresh.extend([1])
    fresh._memo_key = None
    assert kept.lookup(7) == fresh.lookup(7) and kept.last_match == fresh.last_match
    kept.lookup(9)
    assert len(searches) == 1
