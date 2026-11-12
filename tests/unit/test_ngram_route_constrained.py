"""A constrained greedy request must not take the n-gram default route.

The n-gram loop does not drive the constraint state machine, so a json_schema / regex /
grammar request has to be served by the plain fast path, which does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx_lm")

from yunshu_engine.grammar_constraint import ConstraintFactory  # noqa: E402

from .test_reasoning_gate import SeqConstraint  # noqa: E402
from .test_spec_routes_behaviour import N, PROMPT, _spec_engine  # noqa: E402


async def test_greedy_constrained_request_on_ngram_default_is_not_speculated(monkeypatch):
    monkeypatch.setattr(
        ConstraintFactory, "create", staticmethod(lambda *a, **k: SeqConstraint())
    )
    eng = _spec_engine(0)
    eng._ngram_greedy_default = True
    ngram_calls = []
    orig_ngram = eng._generate_ngram_spec

    async def spy_ngram(**kw):
        ngram_calls.append(kw)
        return await orig_ngram(**kw)

    eng._generate_ngram_spec = spy_ngram
    out = await eng.generate(
        prompt=PROMPT,
        max_tokens=N,
        temperature=0.0,
        json_schema={"type": "choice", "choices": ["ab"]},
    )
    assert out.error is None
    assert out.text == "ab"  # the grammar's only valid output, then EOS
    assert ngram_calls == []  # the constrained request never enters the n-gram loop
