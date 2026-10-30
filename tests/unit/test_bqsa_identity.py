"""Identity probe rejects unmatched and divergent concurrent requests."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "bqsa_identity",
    Path(__file__).resolve().parents[2] / "scripts/research/bqsa_identity.py",
)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_burst_and_stagger_match_rows_and_lengths():
    assert len(probe.plans("burst4", 128)) == 4
    assert probe.plans("stagger", 128)[1] == ("B", 42, 102)
    with pytest.raises(KeyError):
        probe.plans("unknown", 128)


def test_digest_comparison_fails_closed():
    same = {"digest": "same", "finish": "length"}
    other = {"digest": "other", "finish": "length"}
    probe.compare({"A": same}, {"A": same})
    with pytest.raises(RuntimeError):
        probe.compare({"A": same}, {"A": other})
    with pytest.raises(RuntimeError):
        probe.compare({"A": same, "B": same}, {"A": same})


def test_compare_modes_requires_every_scenario_and_equal_digests():
    same = {"digest": "same", "finish": "length"}
    rows = [
        {
            "scenario": scenario,
            "shape": shape,
            "solo": {"A": same},
            "concurrent": {"A": same},
        }
        for scenario in ("burst2", "burst4", "stagger")
        for shape in ("greedy", "seeded")
    ]
    off = [*rows, {"complete": True, "draft": "off"}]
    probe.compare_modes(off, rows)
    with pytest.raises(RuntimeError):
        probe.compare_modes(off, rows[:-1])
    bad = [*rows[:-1], {**rows[-1], "concurrent": {"A": {**same, "digest": "changed"}}}]
    with pytest.raises(RuntimeError):
        probe.compare_modes(off, bad)
