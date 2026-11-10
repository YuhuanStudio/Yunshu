"""Speculation off must install the same verify/decode kernels as speculation on."""

import inspect

from yunshu_engine import vlm_engine


def test_kernels_install_for_spec_family_even_without_a_drafter():
    """Regression: with the driver serving spec-off rows, a `not use_driver` clause skipped the
    invariant kernels, so spec off ran different arithmetic from spec on (12/12 digests differ)."""
    from yunshu_engine import serving_kernels

    # the kernel setup moved into serving_kernels.install (M0); the runner build passes spec_family
    src = inspect.getsource(serving_kernels.install)
    assert "(drafter is not None or spec_family)" in src
    assert "spec_family and not use_driver" not in src
    build = inspect.getsource(vlm_engine.VLMEngine._build_batch_runner)
    assert "serving_kernels.install(" in build and "spec_family=spec_family" in build


def test_spec_off_driver_decodes_through_the_verify_forward_without_a_drafter():
    """Regression: the spec-off driver used DecodeBatch (a different forward from the DFlash
    verify window), so spec off != spec on (first divergence at token 0..49 on 27B)."""
    from yunshu_engine.round_driver.dflash import DFlashBatch

    batch = object.__new__(DFlashBatch)
    batch.can_draft = False
    batch.rows, batch.hidden = [], {}
    assert batch.absorb([], []) is None
    src = inspect.getsource(
        __import__(
            "yunshu_engine.round_driver.driver", fromlist=["x"]
        ).RoundDriver.__init__
    )
    assert "plain_verify and drafter is None" in src
