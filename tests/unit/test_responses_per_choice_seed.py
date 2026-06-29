"""Responses n>1 must use a per-choice seed. It passed the SAME req.seed to
every choice, so with an explicit seed + temperature>0 + n>1 all n choices got the same
RNG base key → IDENTICAL token streams (n>1 degraded to one response repeated n times,
still billed n×). chat/completions already offset by +idx; Responses was the outlier."""

from __future__ import annotations

import pathlib

from yunshu_gateway.routers.chat import _per_choice_seed


def test_per_choice_seed_distinct_for_explicit_seed():
    seeds = [_per_choice_seed(42, i) for i in range(5)]
    assert len(set(seeds)) == 5  # every choice independent
    assert seeds == [42, 43, 44, 45, 46]


def test_per_choice_seed_distinct_for_none():
    # No user seed → fresh per-choice (avoids the @mx.compile PRNG-trap collapse).
    assert _per_choice_seed(None, 0) != _per_choice_seed(None, 1)


def test_responses_n_loop_uses_per_choice_seed():
    """Both engine-call sites in the Responses n>1 loop must call _per_choice_seed,
    not pass req.seed verbatim."""
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "python/yunshu_gateway/routers/responses.py").read_text()
    # The 2 sites inside `for choice_idx in range(req.n)` use the helper.
    # ruff may split _per_choice_seed(req.seed, choice_idx) across lines,
    # so count all calls to the helper (must be exactly 2).
    assert src.count("_per_choice_seed(") == 2
