"""CPU-only tests for the sampler A/B driver (scripts/research/fullcmp_sampler_ab.py).

No server, no GPU: the config table, the footprint-free probe set and the DFlash round profile
parser are checked against the real modules they depend on.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_sampler_ab as ab  # noqa: E402
import fullcmp_telemetry as tm  # noqa: E402


def test_config_table_matches_the_experiment():
    assert ab.CONFIGS["A"] is None  # no sampler
    assert ab.CONFIGS["B"] == tm.ALL_PROBES  # harness default, every probe
    assert "footprint" not in ab.CONFIGS["C"]  # the only difference from B
    assert set(ab.CONFIGS["C"]) == set(tm.ALL_PROBES) - {"footprint"}


def test_footprint_probe_is_the_only_slow_probe_removed_from_c():
    assert "footprint" in tm.SLOW_PROBES
    assert [p for p in ab.CONFIGS["B"] if p not in ab.CONFIGS["C"]] == ["footprint"]


def test_profile_line_parses_verify_and_draft_ms():
    line = (
        "2026-11-03 05:42:02,438 INFO yunshu_engine.dflash_fast: "
        "DFlash round profile: 37 rounds, 15.2 nodes/round, draft 6.10 ms, verify 44.50 ms, "
        "commit+host 1.20 ms"
    )
    m = ab.PROFILE_RE.search(line)
    assert m is not None
    rounds, nodes, draft, verify = m.groups()
    assert (rounds, nodes, draft, verify) == ("37", "15.2", "6.10", "44.50")


def test_profile_line_slow_run_parses_as_slow_verify():
    line = "DFlash round profile: 37 rounds, 15.2 nodes/round, draft 6.10 ms, verify 138.90 ms"
    assert float(ab.PROFILE_RE.search(line).group(4)) == 138.9


def test_non_profile_lines_do_not_match():
    assert ab.PROFILE_RE.search("DFlash suffix graft engagement: copy_rounds=0") is None


def test_unknown_config_letter_is_rejected_by_the_sequence_check():
    assert [c for c in "ABX" if c not in ab.CONFIGS] == ["X"]
