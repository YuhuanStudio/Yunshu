"""No test module leaves the dense-lane attention route on for the files that follow it in the same process."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dense_lane_route_is_off_for_a_new_module():
    ragged_kv = sys.modules.get("yunshu_engine.kernels.ragged_kv")
    assert ragged_kv is None or ragged_kv._STATE["dense_lane"] is False


@pytest.mark.m5_gpu
def test_int8_dflash_module_does_not_leak_the_route_into_the_next_file():
    """The pair that failed test_apc_manager_e2e on a shared worker: the DFlash int8 file, then any later file."""
    if not (ROOT / "tests/unit/test_round_driver_int8_dflash.py").exists():
        pytest.skip("the round driver's int8 DFlash tests land with the Phase B merge")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "tests/unit/test_round_driver_int8_dflash.py",
            "tests/unit/test_dense_lane_state_probe.py::test_dense_lane_route_is_off_for_a_new_module",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout[-2000:]
