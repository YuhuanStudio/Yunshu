"""the custom non-streaming temp>0 sampler must produce the SAME
distribution as the streaming path (mlx-lm make_sampler) for the same params.

mlx-lm applies the top_p/min_p/xtc/top_k filters on the UN-tempered logprobs and
applies temperature LAST (categorical_sampling: logprobs * 1/temp). The custom
sampler previously divided by temperature FIRST, so for temperature != 1 combined
with any filter the surviving token set — and thus the sampled distribution —
diverged depending on the `stream` flag.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from python.yunshu_engine.batched_engine import _build_noncached_sampler_text


def _empirical_dist(sampler, logits, vocab, n=20000):
    """Sample many times and return the empirical token distribution."""
    counts = np.zeros(vocab)
    batched = mx.array(np.tile(logits, (n, 1)).astype(np.float32))
    out = np.asarray(sampler(batched)).reshape(-1)
    for t in out:
        counts[int(t)] += 1
    return counts / counts.sum()


def _reference_dist(logits, temp, top_p, vocab):
    """mlx-lm semantics: filter un-tempered logprobs, temp last."""
    from mlx_lm.sample_utils import apply_top_p
    lp = mx.array(logits.astype(np.float32))
    lp = lp - mx.logsumexp(lp, keepdims=True)
    if 0 < top_p < 1:
        lp = apply_top_p(lp, top_p)
    rl = np.array(lp) * (1.0 / temp)
    rl = rl - rl.max()
    p = np.exp(rl)
    return p / p.sum()


@pytest.mark.parametrize("temp,top_p", [(2.0, 0.8), (0.5, 0.9), (1.5, 0.7)])
def test_sampler_matches_mlxlm_with_temp_and_top_p(temp, top_p):
    logits = np.array([3.0, 2.0, 1.0, 0.5, -1.0, -2.0, -3.0, -4.0], dtype=np.float64)
    vocab = len(logits)
    sampler = _build_noncached_sampler_text(temp, top_p, 0, 0.0, seed=1234)
    emp = _empirical_dist(sampler, logits, vocab)
    ref = _reference_dist(logits, temp, top_p, vocab)
    # Empirical should be close to the mlx-lm reference distribution.
    assert np.allclose(emp, ref, atol=0.02), f"emp={np.round(emp,3)} ref={np.round(ref,3)}"
    # The crossing token that the OLD (temp-first) sampler wrongly kept must stay
    # at zero probability under both (top_p nucleus is the same set as mlx-lm).
    assert (emp[ref == 0] < 0.01).all()


def test_sampler_temp_changes_within_nucleus_weighting():
    """Temperature must still reshape the weighting AMONG surviving tokens
    (proves temp is applied, not dropped)."""
    logits = np.array([2.0, 1.0, 0.0, -5.0, -6.0], dtype=np.float64)
    hot = _empirical_dist(_build_noncached_sampler_text(4.0, 1.0, 0, 0.0, seed=7), logits, len(logits))
    cold = _empirical_dist(_build_noncached_sampler_text(0.3, 1.0, 0, 0.0, seed=7), logits, len(logits))
    # Colder temp concentrates mass on the top token; hotter spreads it.
    assert cold[0] > hot[0]
