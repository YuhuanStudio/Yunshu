"""Conformance entrypoint; CPU generic runs on M3, Metal families via gpuq."""

import os

import pytest
from scripts.research.round_driver_golden import capture, compare


@pytest.mark.parametrize("layer", ["generic", "qwen3_5", "qwen3_5_moe", "flash_next"])
def test_layer_golden(layer, tmp_path):
    if layer != "generic" and os.environ.get("GPUQ_OWNER") != "m0-golden":
        pytest.skip("Metal conformance must run through gpuq")
    receipt = capture(layer, device="cpu" if layer == "generic" else "gpu")
    compare(receipt, receipt)


def test_prefill_scopes_restore_after_nested_exception():
    from scripts.research.round_driver_golden import assert_globals, global_state

    from yunshu_engine.kernels import gdn_prefill

    baseline = global_state()
    with pytest.raises(RuntimeError, match="scope probe"):
        with gdn_prefill.prefill_scope():
            assert gdn_prefill.in_prefill()
            with gdn_prefill.tail_scope():
                assert gdn_prefill.in_tail() and gdn_prefill.in_prefill_math()
                raise RuntimeError("scope probe")
    assert_globals(baseline)
