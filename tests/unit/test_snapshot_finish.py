"""The completion collector must refuse early publication and append once."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import snapshot_finish as finish  # noqa: E402


def test_waits_for_every_controller(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "state.json").write_text('{"status":"finished"}')
    assert not finish.all_finished({"mlxlm": first, "splash": second})
    (second / "state.json").write_text('{"status":"running"}')
    assert not finish.all_finished({"mlxlm": first, "splash": second})
    (second / "state.json").write_text('{"status":"finished"}')
    assert finish.all_finished({"mlxlm": first, "splash": second})


def test_early_publication_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        finish.snapshot_report, "combined", lambda *_: {"complete": False}
    )
    with pytest.raises(RuntimeError, match="have not finished"):
        finish.write_results(tmp_path, {}, tmp_path / "last.md")
    assert not (tmp_path / "last.md").exists()


def test_completed_outputs_and_trend_are_written_once(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "reports").mkdir(parents=True)
    (docs / "BENCHMARKS.md").write_text("intro\n# Metric tables\nold")
    (docs / "reports/PERF_TREND.md").write_text("history\n")
    result = {
        "complete": True,
        "release": "release-sha",
        "markdown": "# Snapshot tables\nmetric 42\n",
        "jobs": [{"job": "fixture-job", "ok": True}],
        "failures": [],
        "provenance": {"mlxlm": {"harness": "harness-sha"}},
        "identity_mismatches": [],
        "agent_runs": [],
    }
    monkeypatch.setattr(finish.snapshot_report, "combined", lambda *_: result)
    last = tmp_path / "last.md"
    finish.write_results(tmp_path, {}, last)
    finish.write_results(tmp_path, {}, last)
    assert "metric 42" in (docs / "BENCHMARKS.md").read_text()
    trend = (docs / "reports/PERF_TREND.md").read_text()
    assert trend.startswith("history\n") and trend.count("## snapshot014 —") == 1
    assert "fixture-job" in trend
    assert (
        "own ideas" in last.read_text()
        and "release-notes paragraph" in last.read_text()
    )
    assert (
        json.loads((docs / "research/snapshot014/final_evidence.json").read_text())[
            "complete"
        ]
        is True
    )
