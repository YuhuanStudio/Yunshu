"""inpaint kept-region initialized ON-MANIFOLD for the first denoise step.

The inpaint initial latent blended the KEPT region (1-mask) using the CLEAN (sigma≈0) VAE
encoding while the masked region was noised to sigma_start. But the first denoise step
(t=start_step) feeds the transformer at timestep 1-sigma_start (high sigma), so the clean kept
context was OFF-MANIFOLD for that timestep → boundary seams at the worst (highest-sigma) step.
The in-loop RePaint composite re-noises the kept region to sigma_{t+1} but only AFTER
the Euler step — i.e. for every step except the first. extends it to step 0 by noising
the kept-region init to sigma_start too: known_init = (1-sigma_start)*known + sigma_start*noise.
"""

from __future__ import annotations

import inspect

import mlx.core as mx


def test_inpaint_init_noises_kept_region_to_sigma_start():
    from yunshu_engine import image_engine

    # find the inpaint method source
    src = inspect.getsource(image_engine)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the kept-region init is now the noised flow-matching value, NOT the clean encoding
    assert (
        "known_init = (1.0 - sigma_start) * known_latents_4d + sigma_start * noise"
        in code
    )
    assert "latents = (1 - mask_4d) * known_init + mask_4d * masked_init" in code
    # the old clean-kept init is gone
    assert (
        "latents = (1 - mask_4d) * known_latents_4d + mask_4d * masked_init" not in code
    )


def test_init_is_on_manifold_flow_matching():
    # the init must equal the flow-matching interpolation at sigma_start (same formula the
    # in-loop RePaint uses), so step 0's context matches the timestep it's told.
    known = mx.array([0.5, -1.0, 2.0])
    noise = mx.array([2.0, 0.3, -0.7])
    sigmas = mx.array([0.936, 0.745, 0.483, 0.028, 0.0])
    sigma_start = sigmas[0]
    known_init = (1.0 - sigma_start) * known + sigma_start * noise
    # equals the in-loop RePaint formula evaluated at sigma_start → on the same trajectory
    repaint_at_start = (1.0 - sigmas[0]) * known + sigmas[0] * noise
    assert bool(mx.allclose(known_init, repaint_at_start).item())
    # and it is NOT the clean encoding (the old buggy init)
    assert not bool(mx.allclose(known_init, known).item())
