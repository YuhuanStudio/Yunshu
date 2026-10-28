"""on-GPU Gumbel-max temp>0 sampler is distributionally equivalent to the streaming
make_sampler (mlx-lm filters + categorical), reproducible via explicit key, and doesn't
collapse for n>1."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from python.yunshu_engine.batched_engine import (
    _build_gpu_sampler_text,
    _build_temp_sampler,
)

# generate_step passes LOGPROBS (logits - logsumexp) to the sampler.
_LOGITS = mx.array([[3.0, 2.5, 2.0, 1.0, 0.0, -1.0, -3.0, -5.0]], dtype=mx.float32)
_LOGPROBS = _LOGITS - mx.logsumexp(_LOGITS, axis=-1, keepdims=True)


def _dist(sampler, n=20000):
    c = np.zeros(_LOGITS.shape[-1])
    for _ in range(n):
        c[int(np.asarray(sampler(_LOGPROBS)).reshape(-1)[0])] += 1
    return c / c.sum()


def test_gpu_matches_mlxlm_distribution():
    from mlx_lm.sample_utils import apply_top_p

    gpu = _dist(_build_gpu_sampler_text(1.2, 0.9, 0, 0.0, seed=123))
    lp = apply_top_p(_LOGPROBS, 0.9) * (1.0 / 1.2)
    ref = np.asarray(mx.softmax(lp, axis=-1)).reshape(-1)
    assert float(np.max(np.abs(gpu - ref))) < 0.03


def test_gpu_reproducible_same_seed():
    s1 = _build_gpu_sampler_text(1.0, 1.0, 0, 0.0, seed=42)
    s2 = _build_gpu_sampler_text(1.0, 1.0, 0, 0.0, seed=42)
    q1 = [int(np.asarray(s1(_LOGPROBS)).reshape(-1)[0]) for _ in range(12)]
    q2 = [int(np.asarray(s2(_LOGPROBS)).reshape(-1)[0]) for _ in range(12)]
    assert q1 == q2


def test_gpu_differs_diff_seed_no_collapse():
    s1 = _build_gpu_sampler_text(1.0, 1.0, 0, 0.0, seed=1)
    s2 = _build_gpu_sampler_text(1.0, 1.0, 0, 0.0, seed=2)
    q1 = [int(np.asarray(s1(_LOGPROBS)).reshape(-1)[0]) for _ in range(12)]
    q2 = [int(np.asarray(s2(_LOGPROBS)).reshape(-1)[0]) for _ in range(12)]
    assert q1 != q2  # the @mx.compile PRNG-trap collapse must NOT happen


def test_selector_returns_gpu_sampler():
    s = _build_temp_sampler(0.8, 1.0, 0, 0.0, seed=1)
    assert int(np.asarray(s(_LOGPROBS)).reshape(-1)[0]) in range(_LOGITS.shape[-1])


def test_gpu_returns_mx_array_no_sync():
    s = _build_gpu_sampler_text(0.8, 0.9, 40, 0.0, seed=5)
    out = s(_LOGPROBS)
    assert isinstance(out, mx.array)  # stays on GPU (no .item()/np conversion)
