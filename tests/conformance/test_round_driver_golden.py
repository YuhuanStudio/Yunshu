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
