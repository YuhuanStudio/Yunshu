"""HARN 311 / 312 / 71: where the conc cell and round-golden run, and what stays out of LADDER."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from verify import suites  # noqa: E402

PER_CHANGE = ("tiny", "prefill", "scheduler", "memory", "speed", "identity", "quick")


def test_conc_runs_in_the_long_gate_only():
    assert "conc" in suites.SUITES["long-gate"]["stages"]
    for name in PER_CHANGE:
        if name in suites.SUITES:
            assert "conc" not in suites.SUITES[name]["stages"], name
    assert "conc" not in suites.LADDER


def test_round_golden_runs_in_the_per_change_tiny_suite_without_the_checkpoint():
    tiny = suites.SUITES["tiny"]
    assert "round-golden" in tiny["stages"]
    assert tiny["round_golden_tiny_only"] is True


def test_sub_8k_cells_stay_out_of_the_ladder():
    # HARN 71 decision: 1K/8K/32K+ already bracket 4096 and 20480; no GPU cost for no new axis
    for name in ("full", "tiny"):
        ctx = suites.SUITES[name]["ctx"]
        assert 4096 not in ctx and 20480 not in ctx, name
