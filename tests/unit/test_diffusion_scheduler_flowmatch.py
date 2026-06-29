"""the opt-in YUNSHU_DIFFUSION_SCHEDULER swapped a DDPM/Karras-convention
scheduler (DiffusionScheduler, sigmas sqrt((1-abar)/abar) in range ~[0.03, 14.6]) into
image_engine's FLOW-MATCHING Euler loop (x_t = (1-s)*x0 + s*eps, s in [0,1], timestep =
1 - s). Raw Karras sigmas made timestep = 1 - 14.6 = -13.6 (garbage embedding) and s=14.6
wildly off-manifold for the Z-Image transformer → noise output. The opt-in path silently
produced garbage (a 'fake success' opt-in path, the class).

Fix: _resolve_sigmas converts via SNR matching — flow-matching noise/signal ratio is
s/(1-s), Karras is sigma_k, so s = sigma_k/(1+sigma_k) maps [0,inf)->[0,1) preserving the
relative spacing. This test exercises _resolve_sigmas directly (no model load) and asserts
the resulting schedule is flow-matching-valid.
"""

from __future__ import annotations

import mlx.core as mx

from yunshu_engine.diffusion_infra import SchedulerType
from yunshu_engine.image_engine import ImageGenEngine, _compute_sigmas


def _resolve(scheduler_type, num_steps=8, w=512, h=512):
    eng = ImageGenEngine.__new__(ImageGenEngine)  # no model load
    eng._diffusion_scheduler_type = scheduler_type
    return eng._resolve_sigmas(num_steps, w, h)


def test_opt_in_sigmas_are_flow_matching_valid():
    for st in (SchedulerType.EULER, SchedulerType.DDIM, SchedulerType.DPM_PLUS_PLUS):
        sig = _resolve(st)
        vals = [float(x) for x in sig.tolist()]
        # trailing 0 sentinel, like the default schedule
        assert vals[-1] == 0.0
        body = vals[:-1]
        # every sigma strictly inside (0, 1) — NOT the raw [0.03, 14.6] Karras range
        assert all(0.0 < s < 1.0 for s in body), f"{st}: sigma out of [0,1]: {body}"
        # monotonically descending (Euler dt = s[i+1]-s[i] must stay negative)
        assert all(body[i] > body[i + 1] for i in range(len(body) - 1)), (
            f"{st}: not descending"
        )
        # timestep = 1 - sigma must never go negative (the original bug)
        assert all((1.0 - s) >= 0.0 for s in body)


def test_default_path_unchanged_when_no_override():
    eng = ImageGenEngine.__new__(ImageGenEngine)
    eng._diffusion_scheduler_type = None
    got = eng._resolve_sigmas(8, 512, 512)
    want = _compute_sigmas(8, 512, 512)
    assert mx.allclose(got, want).item()


def test_snr_conversion_is_monotone_in_karras_sigma():
    # s = sk/(sk+1) is strictly increasing in sk, so larger Karras sigma → larger (but <1)
    # flow-matching sigma; the ordering of the schedule is preserved, not scrambled.
    sig = _resolve(SchedulerType.EULER)
    body = [float(x) for x in sig.tolist()][:-1]
    # first (highest-noise) step is the largest sigma and below 1
    assert body[0] == max(body) < 1.0
