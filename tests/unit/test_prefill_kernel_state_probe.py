"""No test module leaves the GDN prefill kernel switch on for the files that follow it in the same process."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gdn_prefill_switch_is_off_for_a_new_module():
    gp = sys.modules.get("yunshu_engine.kernels.gdn_prefill")
    assert gp is None or (gp._STATE["enabled"] is False and gp._STATE["split"] is False)


def test_installing_module_does_not_leak_the_switch_into_the_next_file():
    """The pair that failed test_apc_manager_e2e on a shared worker: a file that installs the kernel, then any later file."""
    probe = "tests/unit/test_prefill_kernel_state_probe.py::test_gdn_prefill_switch_is_off_for_a_new_module"
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
            "tests/unit/leak_helper_gdn_install.py",
            probe,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout[-2000:]
