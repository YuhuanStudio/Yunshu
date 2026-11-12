"""A greedy request with a logits processor must obey it on the n-gram default route.

The n-gram loop never applies request logits processors; the plain fast path does. The
processor here forces one token id, so every generated character must be that token.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402

from .test_spec_routes_behaviour import PROMPT, N, _spec_engine  # noqa: E402

FORCED = 7  # decodes to chr(97 + 7) == "h" in the tiny tokenizer


def _force(tokens, logits):
    mask = mx.arange(logits.shape[-1]) == FORCED
    return mx.where(mask, logits, -float("inf"))


async def test_ngram_default_control_engages_speculation_without_processor():
    eng = _spec_engine(0)
    eng._ngram_greedy_default = True
    out = await eng.generate(prompt=PROMPT, max_tokens=N, temperature=0.0)
    assert out.error is None
    assert eng._ngram_stats["proposals"] > 0  # the n-gram route really runs here
    assert out.text != "h" * N


async def test_greedy_ngram_default_request_obeys_logits_processor():
    eng = _spec_engine(0)
    eng._ngram_greedy_default = True
    out = await eng.generate(
        prompt=PROMPT,
        max_tokens=N,
        temperature=0.0,
        logits_processors=[_force],
    )
    assert out.error is None
    assert out.text == "h" * N
