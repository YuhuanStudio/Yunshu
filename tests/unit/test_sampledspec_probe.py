"""CPU-only contracts for the sampled decode timing driver."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.research.sampledspec_speed import arm_order, validate_row


def test_interleaved_order():
    assert arm_order(3) == ["base", "cand", "cand", "base", "base", "cand"]


def test_timing_rows_fail_closed():
    row = dict(cached=0, ct=65, dec_tps=40, xy={"speculative": {"rounds": 16}})
    assert validate_row(row) == 4
    for bad in (
        {**row, "cached": 1},
        {**row, "cached": None},
        {**row, "ct": 2},
        {**row, "xy": {}},
    ):
        with pytest.raises(RuntimeError):
            validate_row(bad)


def test_dry_run_never_imports_mlx_or_starts_server(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/research/sampledspec_speed.py"
    result = subprocess.run(
        [
            sys.executable,
            str(path),
            "--base-src",
            str(tmp_path / "base"),
            "--cand-src",
            str(tmp_path / "cand"),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["complete"] is True
    assert not (tmp_path / "out").exists()
