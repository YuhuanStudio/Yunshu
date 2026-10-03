"""Legacy replays must never turn partial/CPU results into GPU evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "legacy_audit",
    Path(__file__).resolve().parents[2] / "scripts/research/audit_legacy_replay.py",
)
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def test_final_record_requires_boolean_completion_at_end(tmp_path):
    path = tmp_path / "arm.jsonl"
    path.write_text('{"complete":true}\n{"task":"unfinished"}\n')
    assert not replay.final_record(path)
    path.write_text('{"complete":"dry-run"}\n')
    assert not replay.final_record(path)
    path.write_text('{"complete":true}\n')
    assert replay.final_record(path)
    path.write_text("Traceback\n{invalid}\n")
    assert not replay.final_record(path)


@pytest.mark.parametrize(
    "change", ["source_sha", "dry_run", "smoke", "complete", "arms", "rc"]
)
def test_smoke_gate_rejects_invalid_evidence(change):
    job = {"state": "done", "rc": 0}
    receipt = {
        "source_sha": "sha",
        "dry_run": False,
        "smoke": True,
        "complete": True,
        "arms": [{"rc": 0, "complete": True}],
    }
    assert replay.eligible_smoke(job, receipt, "sha")
    if change == "rc":
        job["rc"] = 1
    else:
        receipt[change] = {
            "source_sha": "other",
            "dry_run": True,
            "smoke": False,
            "complete": False,
            "arms": [],
        }[change]
    assert not replay.eligible_smoke(job, receipt, "sha")


def test_arm_failure_is_recorded_and_later_arm_runs(monkeypatch, tmp_path):
    commands = []

    def run(argv, **kw):
        commands.append(argv)
        kw["stdout"].write('{"complete":true}\n')
        return type("Result", (), {"returncode": 2 if argv[0] == "bad" else 0})()

    monkeypatch.setattr(replay.subprocess, "run", run)
    arms = [
        {
            "name": name,
            "argv": [name],
            "log": str(tmp_path / (name + ".log")),
            "out": str(tmp_path / (name + ".jsonl")),
            "stdout_result": True,
        }
        for name in ["bad", "good"]
    ]
    rows = replay.run_arms(arms)
    assert [row["rc"] for row in rows] == [2, 0]
    assert len(commands) == 2
    assert rows[0]["complete"]  # Completion alone cannot override a failed rc.


def test_dry_run_never_copies_stdout_into_real_result(monkeypatch, tmp_path):
    def run(argv, **kw):
        assert argv[-1] == "--dry-run"
        kw["stdout"].write(json.dumps({"complete": "dry-run"}))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(replay.subprocess, "run", run)
    result = tmp_path / "real.jsonl"
    rows = replay.run_arms(
        [
            {
                "name": "dry",
                "argv": ["script"],
                "log": str(tmp_path / "dry.log"),
                "out": str(result),
                "stdout_result": True,
            }
        ],
        dry_run=True,
    )
    assert rows[0]["rc"] == 0 and not result.exists()
