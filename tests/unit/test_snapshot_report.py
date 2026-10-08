"""CPU tests: incomplete comparisons and agent sets must not be ranked."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import snapshot_report as report  # noqa: E402


def test_partial_agent_set_is_unknown_against_the_best():
    rows = [{"type": "run", "engine": "yunshu-new", "passed": True, "wall_s": 10}]
    rows += [{"type": "run", "engine": "mlxlm", "passed": True, "wall_s": 12}] * 20
    markdown = report.agent_table(rows, ["yunshu-new", "mlxlm"])
    assert "| yunshu-new | 1/20 | 1/1 | unknown |" in markdown
    assert "| mlxlm | 20/20 | 20/20 | 0.0 pp |" in markdown


def test_agent_pass_rate_gap_uses_equal_complete_task_sets():
    rows = [{"type": "run", "engine": "yunshu-new", "passed": True}] * 20
    rows += [{"type": "run", "engine": "mlxlm", "passed": True}] * 19
    rows += [{"type": "run", "engine": "mlxlm", "passed": False, "api_errors": 1}]
    markdown = report.agent_table(rows, ["yunshu-new", "mlxlm"])
    assert "| mlxlm | 20/20 | 19/20 | 5.0 pp | 1/0/0 |" in markdown


def test_finished_run_retains_failed_job_and_unknown_cells(tmp_path):
    (tmp_path / "cells").mkdir()
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "status": "finished",
                "base": {"commit": "release"},
                "cand": {"commit": "harness"},
            }
        )
    )
    event = {
        "ev": "cell_done",
        "cell": "mlxlm-smoke-pilot",
        "ok": False,
        "job": "fixture-job",
        "rc": 1,
        "reason": "fixture failure",
    }
    (tmp_path / "snapshot.jsonl").write_text(json.dumps(event) + "\n")
    result = report.summarize(tmp_path, ["mlxlm"])
    assert result["complete"]
    assert result["failures"] == [event]
    assert (
        "fixture-job" in result["markdown"] and "fixture failure" in result["markdown"]
    )
    assert "unknown" in result["markdown"]


def test_missing_run_is_incomplete(tmp_path):
    result = report.summarize(tmp_path, ["mlxlm"])
    assert not result["complete"]
    assert result["agent_runs"] == [] and result["jobs"] == []


def test_recall_gaps_require_the_same_ten_item_set():
    cells = {
        ("yunshu-new", "needle", 131072, "prose", "correct"): [1] * 10,
        ("mlxlm", "needle", 131072, "prose", "correct"): [1] * 9 + [0],
    }
    markdown = report.recall_table(cells, ["yunshu-new", "mlxlm", "omlx"])
    assert "| 128K | 10/10; gap 0.0 pp | 9/10; gap 10.0 pp | unknown |" in markdown


def test_combined_report_refuses_mixed_release_commits(tmp_path):
    sources = {}
    for engine, release in (("mlxlm", "release-a"), ("splash", "release-b")):
        run = tmp_path / engine
        run.mkdir()
        (run / "state.json").write_text(
            json.dumps(
                {
                    "status": "finished",
                    "base": {"commit": release},
                    "cand": {"commit": "harness"},
                }
            )
        )
        sources[engine] = run
    with pytest.raises(ValueError, match="different released engine commits"):
        report.combined(sources)


def test_combined_report_keeps_per_engine_harness_provenance(tmp_path):
    sources = {}
    for engine, harness in (("mlxlm", "harness-one"), ("splash", "startup-fix")):
        run = tmp_path / engine
        run.mkdir()
        (run / "state.json").write_text(
            json.dumps(
                {
                    "status": "finished",
                    "base": {"commit": "release"},
                    "cand": {"commit": harness},
                }
            )
        )
        sources[engine] = run
    result = report.combined(sources)
    assert result["complete"]
    assert result["provenance"]["splash"]["harness"] == "startup-fix"
    assert "harness-one" in result["markdown"] and "startup-fix" in result["markdown"]
