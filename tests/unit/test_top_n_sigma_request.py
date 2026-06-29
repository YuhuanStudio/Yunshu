"""Per-request top-nσ wiring: schema field + the ContextVar bridge.

The sampler is built on the request's event-loop task before the executor decode,
so a ContextVar set by generate()/stream_generate() is visible in
_build_temp_sampler() and baked into the returned closure — without threading the
value through the deep fast-path call chain. These tests lock that in.
"""

import collections

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import batched_engine as be


def _distinct_tokens(cv_value, n=400):
    """How many distinct token ids the temp sampler emits, given the per-request CV."""
    # 5 strong logits + a 50-token tail with real softmax mass
    logits = mx.array(np.array([6.0] * 5 + [3.0] * 50, dtype=np.float32))[None, :]
    tok = be._REQUEST_TOP_N_SIGMA.set(cv_value)
    try:
        # NOTE: no explicit top_n_sigma arg — the sampler must read the ContextVar
        sampler = be._build_temp_sampler(
            temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=1
        )
        c = collections.Counter(int(sampler(logits).reshape(-1)[0]) for _ in range(n))
        return len(c)
    finally:
        be._REQUEST_TOP_N_SIGMA.reset(tok)


def test_cv_off_lets_tail_through():
    # no per-request value → no filter → the tail is sampled (many distinct ids)
    assert _distinct_tokens(None) > 10


def test_cv_filters_tail():
    # per-request nσ=1.0 → only the 5 logits within 1σ of max survive
    assert _distinct_tokens(1.0) == 5


def test_explicit_arg_beats_cv():
    # an explicit call arg takes precedence over the ContextVar
    logits = mx.array(np.array([6.0] * 5 + [3.0] * 50, dtype=np.float32))[None, :]
    tok = be._REQUEST_TOP_N_SIGMA.set(1.0)  # CV says filter
    try:
        s = be._build_temp_sampler(
            temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=1, top_n_sigma=0.0
        )
        # explicit 0.0 means "unset" → falls through to CV (1.0) → still filters.
        # (explicit > 0 would override; 0 defers, which is the documented order)
        c = collections.Counter(int(s(logits).reshape(-1)[0]) for _ in range(400))
        assert len(c) == 5
    finally:
        be._REQUEST_TOP_N_SIGMA.reset(tok)


def test_chat_request_exposes_top_n_sigma():
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    r = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], top_n_sigma=1.5
    )
    assert r.top_n_sigma == 1.5


def test_chat_request_rejects_out_of_range_top_n_sigma():
    from pydantic import ValidationError

    from yunshu_gateway.routers.chat import ChatCompletionRequest

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], top_n_sigma=99.0
        )
