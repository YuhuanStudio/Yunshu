"""SMOKE batch dispatch, unknown-engine refusal, and the 2-rep default of the real matrix."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))

import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402


@pytest.fixture
def smoke_manifest(tmp_path):
    p = tmp_path / "smoke-manifest.jsonl"
    p.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in fm.build_smoke_rows())
    )
    return p


def test_smoke_batch_is_a_valid_batch_choice_and_dry_runs(smoke_manifest, capsys):
    rc = fr.main(
        [
            "--batch",
            "SMOKE",
            "--manifest",
            str(smoke_manifest),
            "--engines",
            "yunshu",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in out
    assert "SMOKE" in out


def test_smoke_run_audits_only_the_smoke_batch():
    assert fr.audited_batches("SMOKE") == ["SMOKE"]
    assert fr.audited_batches("B1") == ["B1"]
    assert fr.audited_batches("B2") == ["B1", "B2"]
    assert fr.audited_batches("B5") == ["B1", "B2", "B4", "B5"]


def test_unknown_engine_id_is_refused_not_silently_empty(smoke_manifest, capsys):
    # The runbook's bench id "yunshu-main" is not a manifest id ("yunshu"): it must fail closed.
    rc = fr.main(
        [
            "--batch",
            "SMOKE",
            "--manifest",
            str(smoke_manifest),
            "--engines",
            "yunshu-main",
            "--dry-run",
        ]
    )
    assert rc == 2
    assert "yunshu-main" in capsys.readouterr().err


def test_unknown_engine_helper_passes_known_ids():
    rows = fm.build_smoke_rows()
    assert fr.unknown_engines(rows, "SMOKE", ["yunshu"]) == []
    assert fr.unknown_engines(rows, "SMOKE", ["yunshu", "nope"]) == ["nope"]


def test_real_matrix_b1_b2_cells_default_to_two_reps():
    assert fm.REPS == 2
    spec = importlib.util.spec_from_file_location(
        "fullcmp_manifest_reps", SCRIPTS / "fullcmp_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows, _meta = mod.build_rows()
    b12 = [
        r for r in rows if r["batch"] in ("B1", "B2") and r["status"] == "unverified"
    ]
    assert b12, "expected runnable B1/B2 rows"
    assert {r["reps"] for r in b12} == {2}


def test_smoke_expected_records_are_per_rep_file():
    # the audit checks each rep's file on its own: the count does not multiply by reps
    row = dict(fm.build_smoke_rows()[0])
    row["reps"] = 2
    assert fr.row_reps(row) == 2
    assert fr.expected_records(row) == len(row["phases"])
