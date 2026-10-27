"""Speculation off must install the same verify/decode kernels as speculation on."""

import inspect

from yunshu_engine import vlm_engine


def test_kernels_install_for_spec_family_even_without_a_drafter():
    """Regression: with the driver serving spec-off rows, a `not use_driver` clause skipped the
    invariant kernels, so spec off ran different arithmetic from spec on (12/12 digests differ)."""
    src = inspect.getsource(vlm_engine.VLMEngine._build_batch_runner)
    assert "(drafter is not None or spec_family)" in src
    assert "spec_family and not use_driver" not in src
