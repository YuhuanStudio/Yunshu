"""the engine-loop (scheduler._make_sampler) is the concurrency path,
so it must NOT route temp>0 through mlx-lm's PRNG-trapped make_sampler. Each
running sequence must sample from an independent per-request RNG (no @mx.compile
PRNG-state collapse), reproducibly when a seed is given, while greedy (temp==0)
stays on argmax. Mirrors the fast-path fix, swept into the engine-loop."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from python.yunshu_engine.request import SamplingParams
from python.yunshu_engine.scheduler import Scheduler


class _Stub:
    """Minimal carrier for the bound _make_sampler — it only reads self.running
    (len, for the legacy global-seed guard) and self.tokenizer (constrained path,
    unused here)."""
    running: list = []
    tokenizer = None


_make_sampler = Scheduler._make_sampler

# generate_step feeds logprobs (logits - logsumexp) to the sampler.
_LOGITS = mx.array([[3.0, 2.5, 2.0, 1.0, 0.0, -1.0, -3.0, -5.0]], dtype=mx.float32)
_LOGPROBS = _LOGITS - mx.logsumexp(_LOGITS, axis=-1, keepdims=True)


def _stream(sampler, n=16):
    return [int(np.asarray(sampler(_LOGPROBS)).reshape(-1)[0]) for _ in range(n)]


def test_engine_loop_temp_pos_independent_per_seed():
    """Two concurrent temp>0 requests with different seeds must NOT collapse to
    the same token stream (the @mx.compile PRNG-trap symptom)."""
    s1 = _make_sampler(_Stub(), SamplingParams(temperature=0.9, seed=1))
    s2 = _make_sampler(_Stub(), SamplingParams(temperature=0.9, seed=2))
    assert _stream(s1) != _stream(s2)


def test_engine_loop_temp_pos_reproducible_same_seed():
    s1 = _make_sampler(_Stub(), SamplingParams(temperature=0.9, seed=7))
    s2 = _make_sampler(_Stub(), SamplingParams(temperature=0.9, seed=7))
    assert _stream(s1) == _stream(s2)


def test_engine_loop_greedy_is_argmax_deterministic():
    s = _make_sampler(_Stub(), SamplingParams(temperature=0.0))
    # token 0 is the argmax of _LOGITS; greedy must always pick it.
    assert set(_stream(s)) == {0}
