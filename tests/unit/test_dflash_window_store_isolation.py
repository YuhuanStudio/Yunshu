"""A test that runs DFlash with APC does not leave captured-layer windows in the process-wide store for the next file."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_window_store_is_empty_for_a_new_test():
    mod = sys.modules.get("yunshu_engine.dflash_apc_window")
    assert mod is None or mod.resident_bytes() == 0


def test_the_pair_that_failed_the_memory_ledger_test():
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
            "tests/unit/test_round_driver_dflash.py",
            "tests/unit/test_round_driver_int8_dflash.py",
            "tests/unit/test_memory_ledger_counters.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2500:]
