"""(MED): denoise_strength=0.0 did NOT keep the source — it injected noise.

img2img/inpaint compute start_step = int(num_steps*(1-strength)) then clamp
max(0, min(start_step, num_steps-1)). For strength=0 that clamp forces
start_step=num_steps-1, so the loop runs ONE denoise step with the blend
(1-sigma_start)*source + sigma_start*noise where sigma_start=sigmas[num_steps-1] is
NON-zero (≈0.51 at the Turbo 4-step default) — the source is heavily re-rendered,
contradicting the docstring + /images router "0.0 = keep source" contract.

special-cases strength<=0 → start_step=num_steps BEFORE the clamp, so the loop
is empty and the blend reads sigma_start=sigmas[num_steps]=0.0 → pure source. Applied
to BOTH img2img and inpaint (consistent siblings).
"""

from __future__ import annotations

import inspect

from yunshu_engine.image_engine import ImageGenEngine


def _start_step(num_steps: int, strength: float) -> int:
    """Mirror of the fixed pipeline start_step formula."""
    s = int(num_steps * (1.0 - strength))
    if strength <= 0.0:
        return num_steps
    s = max(0, min(s, num_steps - 1))
    if strength < 1.0 and num_steps >= 2:
        s = max(1, s)
    return s


def test_sigmas_have_trailing_zero_sentinel():
    # the fix's correctness relies on sigmas[num_steps] == 0.0 (empty-loop blend = pure source)
    eng = ImageGenEngine.__new__(ImageGenEngine)
    eng._diffusion_scheduler_type = None
    for n in (4, 8, 20):
        sig = eng._resolve_sigmas(n, 1024, 1024)
        assert len(sig) == n + 1
        assert float(sig[n]) == 0.0, f"sigmas[{n}] must be the trailing 0 sentinel"
        assert float(sig[n - 1]) > 0.0  # the wrongly-used noise level (the old bug)


def test_start_step_formula_keeps_source_at_strength_zero():
    # strength=0 → start_step == num_steps → empty loop range(num_steps, num_steps)
    assert _start_step(4, 0.0) == 4
    assert _start_step(20, 0.0) == 20
    # strength=1 → full denoise from pure noise
    assert _start_step(4, 1.0) == 0
    # the low-step-count floor still holds for 0 < strength < 1
    assert _start_step(4, 0.8) == 1
    assert _start_step(20, 0.5) == 10


def test_img2img_pipeline_special_cases_strength_zero():
    src = inspect.getsource(ImageGenEngine._run_img2img_pipeline)
    i = src.index("if denoise_strength <= 0.0:")
    # the special-case sets start_step=num_steps and comes BEFORE the min-clamp
    assert "start_step = num_steps" in src[i : i + 120]
    clamp = src.index("min(start_step, num_steps - 1)")
    assert i < clamp, "strength<=0 special-case must precede the min-clamp"


def test_inpaint_pipeline_special_cases_strength_zero():
    src = inspect.getsource(ImageGenEngine._run_inpaint_pipeline)
    i = src.index("if denoise_strength <= 0.0:")
    assert "start_step = num_steps" in src[i : i + 120]
    clamp = src.index("min(start_step, num_steps - 1)")
    assert i < clamp
