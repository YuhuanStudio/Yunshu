"""the opt-in GPU sampler (YUNSHU_GPU_SAMPLER=1) applied top_k BEFORE XTC, so
XTC computed its cutoff over a top_k-truncated distribution — masking a different token
set than mlx-lm and the default numpy sampler (top_p→min_p→XTC→top_k), breaking the
sampler's own "identical to the numpy path" invariant. top_k now applies LAST (after XTC).

(The default served sampler is the numpy one and was already correct; this only affected
the opt-in GPU sampler.)"""
from __future__ import annotations

import inspect

import mlx.core as mx

from yunshu_engine.batched_engine import _build_gpu_sampler_text


def test_top_k_applied_after_xtc_in_source():
    src = inspect.getsource(_build_gpu_sampler_text)
    # the XTC coin draw and the top_k application both exist...
    assert "coin = mx.random.uniform" in src
    assert "apply_top_k(lp, _top_k_n)" in src
    # ...and top_k comes AFTER the XTC block (mlx-lm order)
    assert src.index("coin = mx.random.uniform") < src.index("apply_top_k(lp, _top_k_n)")
    # top_k is no longer pushed into the pre-XTC `methods` list
    assert "methods.append(lambda x: apply_top_k" not in src


def test_gpu_sampler_runs_with_xtc_and_top_k():
    s = _build_gpu_sampler_text(temperature=0.8, top_p=0.9, top_k=5, min_p=0.0,
                                xtc_probability=0.5, xtc_threshold=0.1, seed=42)
    lp = mx.log(mx.softmax(mx.array([[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001]]), axis=-1))
    tok = s(lp)
    tid = int(tok.reshape(-1)[0])
    assert 0 <= tid < 10


def test_gpu_sampler_top_k_alone_still_constrains():
    # top_k=1 (no xtc) at low temperature must pick the argmax token (index 0 here)
    s = _build_gpu_sampler_text(temperature=0.01, top_p=1.0, top_k=1, min_p=0.0,
                                xtc_probability=0.0, xtc_threshold=0.1, seed=7)
    lp = mx.log(mx.softmax(mx.array([[5.0, 1.0, 0.5, 0.1]]), axis=-1))
    assert int(s(lp).reshape(-1)[0]) == 0


def test_gpu_sampler_seed_reproducible():
    def _run(seed):
        s = _build_gpu_sampler_text(temperature=0.9, top_p=1.0, top_k=0, min_p=0.0,
                                    xtc_probability=0.0, xtc_threshold=0.1, seed=seed)
        lp = mx.log(mx.softmax(mx.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), axis=-1))
        return int(s(lp).reshape(-1)[0])
    assert _run(123) == _run(123)  # same seed → identical
