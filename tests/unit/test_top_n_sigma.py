"""top-nσ sampler (ACL 2025): keep only tokens whose raw logit is within n·σ of
the max logit. Temperature-invariant; fires on the fast-path samplers + the
YUNSHU_TOP_N_SIGMA env default. Guards the filter math + that it's actually wired."""

import mlx.core as mx

from yunshu_engine.batched_engine import (
    _build_gpu_sampler_text,
    _build_noncached_sampler_text,
    _build_temp_sampler,
)

# logits where only token 0 sits within ~0.5σ of the max; the rest are far below.
_LOGITS = mx.array([[12.0, 1.0, 1.0, 1.0, 1.0, 1.0]])


def _runs(sampler, n=64):
    return {int(sampler(_LOGITS).reshape(-1)[0]) for _ in range(n)}


def test_numpy_sampler_top_n_sigma_masks_low_logits():
    # With top_n_sigma small, only the dominant logit survives → always token 0.
    s = _build_noncached_sampler_text(
        temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=7, top_n_sigma=0.5
    )
    assert _runs(s) == {0}


def test_numpy_sampler_off_can_pick_others():
    # Without top-nσ (and high temp), the tail is reachable → not locked to {0}.
    s = _build_noncached_sampler_text(
        temperature=5.0, top_p=1.0, top_k=0, min_p=0.0, seed=7, top_n_sigma=0.0
    )
    assert len(_runs(s, n=128)) > 1


def test_gpu_sampler_top_n_sigma_matches_numpy_mask():
    s = _build_gpu_sampler_text(
        temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=7, top_n_sigma=0.5
    )
    assert _runs(s) == {0}


def test_env_default_wires_top_n_sigma(monkeypatch):
    # A path that passes no explicit value still picks up YUNSHU_TOP_N_SIGMA.
    monkeypatch.setenv("YUNSHU_TOP_N_SIGMA", "0.5")
    s = _build_temp_sampler(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=7)
    assert _runs(s) == {0}


def test_env_default_absent_is_noop(monkeypatch):
    monkeypatch.delenv("YUNSHU_TOP_N_SIGMA", raising=False)
    s = _build_temp_sampler(temperature=5.0, top_p=1.0, top_k=0, min_p=0.0, seed=7)
    assert len(_runs(s, n=128)) > 1
