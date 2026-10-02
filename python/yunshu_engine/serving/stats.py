"""Statistics for online experiments on the serving path (pure Python, CPU only).

- :func:`switchback_test`: randomization test for switchback (time-block) experiments. The
  unit is a time block, the p value is exact (full enumeration) when the permutation space is
  small and a seeded Monte Carlo otherwise; the design restriction "no more than ``max_run``
  consecutive blocks of one arm" is honoured by only drawing assignments the design could
  have produced. :func:`switchback_ci` inverts the test for a constant additive effect.
- :func:`cuped_adjust`: CUPED variance reduction with a **pre-treatment** covariate only
  (theta is fitted on both arms pooled, which is valid because the covariate cannot depend on
  the arm). :func:`check_covariate` refuses names of post-treatment quantities.
- :class:`MSPRT`: an always-valid sequential test (normal mixture, Johari-Pekelis-Walsh) for
  guardrails and early stopping; it may be looked at after every observation without
  inflating the false-positive rate.
- :func:`sample_size_per_arm`: n for a two-sided test (alpha 0.05, power 0.8).

Measurement code fails closed: NaN, mismatched lengths, a missing arm or a constant
covariate raise :class:`StatsError`; nothing is imputed or dropped silently.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb


class StatsError(ValueError):
    """The data cannot support the requested inference."""


# Z quantiles (two-sided alpha -> z_{1 - alpha/2}, power -> z_power) without scipy.
_NORM = statistics.NormalDist()


def _finite(values: Sequence[float], what: str) -> list[float]:
    out = [float(v) for v in values]
    if any(v != v or v in (math.inf, -math.inf) for v in out):
        raise StatsError(f"{what} contains a missing or non-finite value")
    return out


# ── sample size ─────────────────────────────────────────────────────────


def sample_size_per_arm(
    sigma: float,
    delta: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = False,
) -> int:
    """Observations per arm to detect an absolute change ``delta`` (same unit as ``sigma``).
    Paired designs pass the sd of the paired differences as ``sigma`` and get one n for all
    pairs; unpaired uses ``2 (z_a + z_b)^2 sigma^2 / delta^2``."""
    if not (sigma > 0 and delta > 0):
        raise StatsError("sigma and delta must be positive")
    z = _NORM.inv_cdf(1 - alpha / 2) + _NORM.inv_cdf(power)
    n = (z * sigma / delta) ** 2 * (1 if paired else 2)
    return math.ceil(n)


# ── randomization test for switchback blocks ─────────────────────────────


def _runs_ok(labels: Sequence[int], max_run: int | None) -> bool:
    if max_run is None:
        return True
    run = 1
    for a, b in itertools.pairwise(labels):
        run = run + 1 if a == b else 1
        if run > max_run:
            return False
    return True


def _diff(values: Sequence[float], labels: Sequence[int]) -> float:
    a = [v for v, lab in zip(values, labels, strict=True) if lab == 0]
    b = [v for v, lab in zip(values, labels, strict=True) if lab == 1]
    return statistics.fmean(b) - statistics.fmean(a)


@dataclass(frozen=True)
class PermutationResult:
    effect: float  # mean(treatment) - mean(control) over blocks
    p_value: float  # two-sided
    n_blocks: int
    n_assignments: int  # permutations used (all of them when exact)
    exact: bool


def _assignments(
    labels: Sequence[int],
    max_run: int | None,
    n_perm: int,
    rng: random.Random,
    exact_limit: int,
):
    """Yield (assignments, exact). Exact enumeration of the label arrangements when the space
    is at most ``exact_limit``; otherwise ``n_perm`` seeded random arrangements."""
    n = len(labels)
    k = sum(labels)
    if comb(n, k) <= exact_limit:
        found = []
        for ones in itertools.combinations(range(n), k):
            arr = [0] * n
            for i in ones:
                arr[i] = 1
            if _runs_ok(arr, max_run):
                found.append(arr)
        return found, True
    out: list[list[int]] = []
    base = list(labels)
    tries = 0
    while len(out) < n_perm:
        tries += 1
        if tries > n_perm * 1000:
            raise StatsError("design restriction leaves too few admissible assignments")
        rng.shuffle(base)
        if _runs_ok(base, max_run):
            out.append(list(base))
    return out, False


def switchback_test(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    max_run: int | None = None,
    n_perm: int = 5000,
    seed: int = 0,
    exact_limit: int = 20000,
) -> PermutationResult:
    """Randomization test of the sharp null of no effect on any block.

    ``values``: one number per time block (e.g. the mean log decode tok/s of the block's
    requests); ``labels``: 0 = control, 1 = treatment. ``max_run`` restricts the reference
    distribution to arrangements with at most that many consecutive blocks of one arm, as the
    design did. The observed arrangement is always counted, so the p value is never 0."""
    vals = _finite(values, "values")
    lab = [int(v) for v in labels]
    if len(vals) != len(lab):
        raise StatsError("values and labels differ in length")
    if set(lab) != {0, 1}:
        raise StatsError("both arms need at least one block")
    if min(lab.count(0), lab.count(1)) < 2:
        raise StatsError("each arm needs at least 2 blocks")
    if not _runs_ok(lab, max_run):
        raise StatsError("the observed assignment violates max_run")
    obs = _diff(vals, lab)
    arrangements, exact = _assignments(
        lab, max_run, n_perm, random.Random(seed), exact_limit
    )
    tol = 1e-12 * (1 + abs(obs))
    extreme = sum(1 for arr in arrangements if abs(_diff(vals, arr)) >= abs(obs) - tol)
    if exact:
        p = extreme / len(arrangements)
    else:
        p = (extreme + 1) / (len(arrangements) + 1)
    return PermutationResult(obs, min(p, 1.0), len(vals), len(arrangements), exact)


def switchback_ci(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    alpha: float = 0.05,
    max_run: int | None = None,
    n_perm: int = 2000,
    seed: int = 0,
    grid: int = 81,
    exact_limit: int = 20000,
) -> tuple[float, float]:
    """Confidence interval for a constant additive effect by test inversion: the effects
    tau whose adjusted outcomes (treatment blocks minus tau) the test does not reject. The
    grid spans the observed effect +- 4 standard errors; a bound that touches the grid edge
    is reported as +-inf rather than as a finite number."""
    vals = _finite(values, "values")
    lab = [int(v) for v in labels]
    obs = _diff(vals, lab)
    sd = statistics.pstdev(vals)
    se = sd * math.sqrt(1 / lab.count(0) + 1 / lab.count(1)) or 1e-9
    taus = [obs + se * 4 * (2 * i / (grid - 1) - 1) for i in range(grid)]
    kept = []
    for tau in taus:
        adj = [v - tau * lb for v, lb in zip(vals, lab, strict=True)]
        p = switchback_test(
            adj, lab, max_run=max_run, n_perm=n_perm, seed=seed, exact_limit=exact_limit
        ).p_value
        if p > alpha:
            kept.append(tau)
    if not kept:
        return (obs, obs)
    lo = -math.inf if kept[0] == taus[0] else kept[0]
    hi = math.inf if kept[-1] == taus[-1] else kept[-1]
    return (lo, hi)


# ── CUPED ───────────────────────────────────────────────────────────────

# Quantities that exist only after the request ran with its arm; using one as a covariate
# would bias the estimate. The allowed set is what is known before the arm is applied.
POST_TREATMENT = frozenset(
    {
        "decode_tps",
        "ttft_ms",
        "ttft_s",
        "prefill_tps",
        "decode_ms",
        "prefill_ms",
        "queue_wait_ms",
        "total_ms",
        "completion_tokens",
        "acceptance_rate",
        "spec_acceptance",
        "spec_accepted",
        "spec_drafted",
        "finish_reason",
        "cancelled",
        "concurrency_end",
        "cached_tokens",
    }
)
PRE_TREATMENT = frozenset(
    {
        "prompt_tokens",
        "ctx_bucket",
        "has_tools",
        "has_media",
        "stream",
        "dialect",
        "prev_decode_tps",  # the same session's previous request, finished before this arm
        "prev_ttft_ms",
        "hour_of_day",
        "baseline_decode_tps",  # trailing baseline of the block's workload class
    }
)


def check_covariate(name: str) -> None:
    """Raise for a covariate that is not known to be pre-treatment (fail closed: an unknown
    name is refused until it is added to :data:`PRE_TREATMENT` on purpose)."""
    if name in POST_TREATMENT:
        raise StatsError(
            f"covariate {name!r} is measured after treatment; CUPED would bias"
        )
    if name not in PRE_TREATMENT:
        raise StatsError(f"covariate {name!r} is not registered as pre-treatment")


@dataclass(frozen=True)
class CupedResult:
    adjusted: list[float]
    theta: float
    variance_ratio: float  # var(adjusted) / var(y): 1 - rho^2 in expectation


def cuped_adjust(
    y: Sequence[float], x: Sequence[float], *, theta: float | None = None
) -> CupedResult:
    """``y - theta (x - mean(x))`` with ``theta = cov(x, y) / var(x)`` fitted on all units
    pooled (both arms). Pass ``theta`` to reuse a coefficient fitted on pre-experiment data.
    The mean of the adjusted metric equals the mean of ``y``, so effect estimates are
    unchanged in expectation while their variance drops by about ``rho^2``."""
    ys, xs = _finite(y, "y"), _finite(x, "x")
    if len(ys) != len(xs):
        raise StatsError("y and x differ in length")
    if len(ys) < 3:
        raise StatsError("need at least 3 units")
    mx = statistics.fmean(xs)
    if theta is None:
        vx = statistics.pvariance(xs)
        if vx <= 0:
            raise StatsError("covariate is constant")
        my = statistics.fmean(ys)
        theta = (
            sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / len(xs) / vx
        )
    adj = [b - theta * (a - mx) for a, b in zip(xs, ys, strict=True)]
    vy = statistics.pvariance(ys)
    ratio = statistics.pvariance(adj) / vy if vy > 0 else 1.0
    return CupedResult(adj, theta, ratio)


# ── mSPRT ───────────────────────────────────────────────────────────────


class MSPRT:
    """Always-valid sequential test of H0: mean = 0 for a stream of (paired) differences with
    known variance ``sigma2`` (pre-registered from a baseline, never re-estimated from the
    stream) and a N(0, ``tau2``) mixing prior on the effect.

    ``p_value`` is non-increasing and valid at every stopping time:
    ``P_H0(exists n: p_n <= a) <= a``. ``ci()`` is the matching always-valid interval."""

    def __init__(self, sigma2: float, tau2: float, alpha: float = 0.05):
        if not (sigma2 > 0 and tau2 > 0 and 0 < alpha < 1):
            raise StatsError("sigma2, tau2 must be positive and 0 < alpha < 1")
        self.sigma2 = sigma2
        self.tau2 = tau2
        self.alpha = alpha
        self.n = 0
        self._sum = 0.0
        self._p = 1.0

    def update(self, x: float) -> float:
        """Add one observation (a paired difference); returns the always-valid p value."""
        x = float(x)
        if x != x or x in (math.inf, -math.inf):
            raise StatsError("observation is missing or non-finite")
        self.n += 1
        self._sum += x
        n, s2, t2 = self.n, self.sigma2, self.tau2
        mean = self._sum / n
        log_lam = 0.5 * math.log(s2 / (s2 + n * t2)) + (n * n * t2 * mean * mean) / (
            2 * s2 * (s2 + n * t2)
        )
        self._p = min(self._p, math.exp(-log_lam) if log_lam < 700 else 0.0)
        return self._p

    def update_pair(self, control: float, treatment: float) -> float:
        return self.update(treatment - control)

    @property
    def p_value(self) -> float:
        return self._p

    @property
    def rejected(self) -> bool:
        return self._p <= self.alpha

    @property
    def mean(self) -> float | None:
        return self._sum / self.n if self.n else None

    def ci(self) -> tuple[float, float] | None:
        """Always-valid (1 - alpha) interval for the mean difference, or None before data."""
        if not self.n:
            return None
        n, s2, t2 = self.n, self.sigma2, self.tau2
        radius = math.sqrt(
            s2
            * (s2 + n * t2)
            / (n * n * t2)
            * (math.log((s2 + n * t2) / s2) + 2 * math.log(1 / self.alpha))
        )
        m = self._sum / n
        return (m - radius, m + radius)
