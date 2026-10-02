"""Experiment statistics: validity (A/A), power, CUPED variance reduction, mSPRT, fail-closed."""

from __future__ import annotations

import math
import random

import pytest

from yunshu_engine.serving import stats


def _blocks(rng, n, effect=0.0, sd=1.0):
    labels = [0] * (n // 2) + [1] * (n // 2)
    rng.shuffle(labels)
    vals = [rng.gauss(0, sd) + effect * lab for lab in labels]
    return vals, labels


def test_permutation_exact_small():
    # 3 vs 3 blocks, perfectly separated: p = 2 / C(6,3) = 0.1 (two-sided)
    res = stats.switchback_test([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1])
    assert res.exact and res.n_assignments == 20
    assert res.p_value == pytest.approx(2 / 20)
    assert res.effect == pytest.approx(9.0)


def test_permutation_deterministic_with_seed():
    rng = random.Random(1)
    vals, labels = _blocks(rng, 40, 0.5)
    a = stats.switchback_test(vals, labels, n_perm=500, seed=3)
    b = stats.switchback_test(vals, labels, n_perm=500, seed=3)
    assert a == b and not a.exact


def test_max_run_restriction_shrinks_reference_set():
    labels = [0, 1, 0, 1, 1, 0, 0, 1]
    vals = [1.0, 2.0, 1.1, 2.1, 1.9, 0.9, 1.0, 2.2]
    free = stats.switchback_test(vals, labels)
    cons = stats.switchback_test(vals, labels, max_run=2)
    assert cons.n_assignments < free.n_assignments
    with pytest.raises(stats.StatsError):
        stats.switchback_test(vals, [0, 0, 0, 1, 1, 1, 0, 1], max_run=2)


def test_aa_false_positive_rate_at_alpha():
    """1000 simulated A/A experiments (seeded): rejection rate within tolerance of 5%."""
    rng = random.Random(20260102)
    rejections = 0
    for i in range(1000):
        vals, labels = _blocks(rng, 16)
        p = stats.switchback_test(
            vals, labels, n_perm=199, seed=i, exact_limit=10
        ).p_value
        rejections += p <= 0.05
    rate = rejections / 1000
    assert 0.03 <= rate <= 0.07, rate


def test_aa_with_run_restriction_still_valid():
    rng = random.Random(7)
    rejections = n = 0
    while n < 400:
        labels = [0, 1] * 8
        rng.shuffle(labels)
        if not stats._runs_ok(labels, 2):
            continue
        n += 1
        vals = [rng.gauss(0, 1) for _ in labels]
        p = stats.switchback_test(
            vals, labels, max_run=2, n_perm=199, seed=n, exact_limit=10
        ).p_value
        rejections += p <= 0.05
    assert rejections / n <= 0.09


def test_power_under_known_effect():
    rng = random.Random(11)
    hits = 0
    for i in range(200):
        vals, labels = _blocks(rng, 24, effect=1.5)
        hits += (
            stats.switchback_test(
                vals, labels, n_perm=299, seed=i, exact_limit=10
            ).p_value
            <= 0.05
        )
    assert hits / 200 >= 0.8


def test_ci_covers_true_effect_and_excludes_zero_for_big_effect():
    rng = random.Random(5)
    vals, labels = _blocks(rng, 20, effect=3.0, sd=0.5)
    lo, hi = stats.switchback_ci(vals, labels, n_perm=300, grid=41, exact_limit=10)
    assert lo < 3.0 < hi and lo > 0


@pytest.mark.parametrize(
    "vals,labels",
    [
        ([1, 2, math.nan, 4], [0, 0, 1, 1]),
        ([1, 2, 3], [0, 0, 1, 1]),
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([1, 2, 3, 4], [0, 1, 1, 1]),
    ],
)
def test_permutation_fails_closed(vals, labels):
    with pytest.raises(stats.StatsError):
        stats.switchback_test(vals, labels)


def test_cuped_reduces_variance_when_correlated():
    rng = random.Random(3)
    rho = 0.7
    x = [rng.gauss(0, 1) for _ in range(4000)]
    y = [rho * a + math.sqrt(1 - rho**2) * rng.gauss(0, 1) + 10 for a in x]
    res = stats.cuped_adjust(y, x)
    assert res.variance_ratio == pytest.approx(1 - rho**2, abs=0.04)
    assert sum(res.adjusted) / len(y) == pytest.approx(sum(y) / len(y), abs=1e-9)


def test_cuped_uncorrelated_does_not_inflate():
    rng = random.Random(4)
    x = [rng.gauss(0, 1) for _ in range(2000)]
    y = [rng.gauss(0, 1) for _ in range(2000)]
    assert stats.cuped_adjust(y, x).variance_ratio <= 1.0 + 1e-9


def test_cuped_keeps_effect_estimate_and_tightens_it():
    """Randomized treatment effect 0.5 on y; x pre-treatment. Both estimates are unbiased,
    the adjusted one has smaller spread across simulations."""
    rng = random.Random(8)
    raw_est, adj_est = [], []
    for _ in range(300):
        n = 200
        x = [rng.gauss(0, 1) for _ in range(n)]
        arm = [i % 2 for i in range(n)]
        rng.shuffle(arm)
        y = [0.8 * a + 0.5 * t + rng.gauss(0, 0.6) for a, t in zip(x, arm, strict=True)]
        adj = stats.cuped_adjust(y, x).adjusted

        def eff(v, arm=arm):
            t = [a for a, k in zip(v, arm, strict=True) if k]
            c = [a for a, k in zip(v, arm, strict=True) if not k]
            return sum(t) / len(t) - sum(c) / len(c)

        raw_est.append(eff(y))
        adj_est.append(eff(adj))
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    var = lambda v: sum((a - mean(v)) ** 2 for a in v) / len(v)  # noqa: E731
    assert mean(adj_est) == pytest.approx(0.5, abs=0.03)
    assert var(adj_est) < 0.6 * var(raw_est)


def test_cuped_fails_closed():
    with pytest.raises(stats.StatsError):
        stats.cuped_adjust([1, 2, 3], [1, 1, 1])
    with pytest.raises(stats.StatsError):
        stats.cuped_adjust([1, 2, 3], [1, 2])
    with pytest.raises(stats.StatsError):
        stats.cuped_adjust([1, math.nan, 3], [1, 2, 3])


def test_covariate_must_be_pre_treatment():
    stats.check_covariate("prompt_tokens")
    stats.check_covariate("prev_decode_tps")
    for bad in ("spec_acceptance", "decode_tps", "ttft_ms", "unknown_thing"):
        with pytest.raises(stats.StatsError):
            stats.check_covariate(bad)


def test_msprt_aa_false_positive_with_continuous_peeking():
    rng = random.Random(99)
    sigma2, tau2 = 1.0, 0.25
    rejected = naive = 0
    for _ in range(1000):
        m = stats.MSPRT(sigma2, tau2, alpha=0.05)
        total, hit_naive = 0.0, False
        for n in range(1, 301):
            x = rng.gauss(0, 1)
            m.update(x)
            total += x
            if abs(total) / math.sqrt(n) > 1.96:
                hit_naive = True
        rejected += m.rejected
        naive += hit_naive
    assert rejected / 1000 <= 0.06  # always valid
    assert naive / 1000 > 0.2  # the peeking z-test would have been badly inflated


def test_msprt_detects_effect_and_ci_excludes_zero():
    rng = random.Random(2)
    m = stats.MSPRT(1.0, 0.25)
    stopped = None
    for n in range(1, 2000):
        m.update(rng.gauss(0.3, 1.0))
        if m.rejected:
            stopped = n
            break
    assert stopped is not None and stopped < 600
    lo, hi = m.ci()
    assert lo > 0 and hi > lo


def test_msprt_p_monotone_and_fail_closed():
    m = stats.MSPRT(1.0, 1.0)
    ps = [m.update(x) for x in (0.1, 2.0, -0.5, 1.0)]
    assert all(b <= a for a, b in zip(ps, ps[1:], strict=False))
    assert m.ci() is not None and stats.MSPRT(1, 1).ci() is None
    with pytest.raises(stats.StatsError):
        m.update(math.nan)
    with pytest.raises(stats.StatsError):
        stats.MSPRT(0.0, 1.0)


def test_sample_size_matches_design_formula():
    # n = 15.7 (sigma/delta)^2 unpaired, 7.85 paired
    assert stats.sample_size_per_arm(0.15, 0.10) in (35, 36)
    assert stats.sample_size_per_arm(0.15, 0.10, paired=True) in (17, 18)
    with pytest.raises(stats.StatsError):
        stats.sample_size_per_arm(0.0, 0.1)
