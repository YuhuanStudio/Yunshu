"""Release orchestration contract, no GPU or live queue needed."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dev/release_check"
loader = importlib.machinery.SourceFileLoader("release_check", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
rc = importlib.util.module_from_spec(spec)
loader.exec_module(rc)


def test_plan_pins_refs_and_keeps_benchmark_nonblocking(tmp_path):
    sha = "a" * 40
    commands = rc.plan(sha, tmp_path, tmp_path, "releng015")
    assert set(commands) == {"gate", "m3sweep", "agentcompat", "agentbench"}
    assert sha in commands["m3sweep"] and sha in commands["agentbench"]
    assert "--no-wait" in commands["agentbench"]
    assert (
        commands["agentbench"][commands["agentbench"].index("--priority") + 1] == "-3"
    )


def test_verdict_fails_closed(tmp_path):
    p = tmp_path / "verdict.json"
    assert rc.judge(0, p) == "FAIL"
    for value in (
        [],
        {},
        {"overall": "FAIL"},
        {"ok": False},
        {"overall": "FAIL", "ok": True},
    ):
        p.write_text(json.dumps(value))
        assert rc.judge(0, p) == "FAIL"
    for value in ({"overall": "PASS"}, {"verdict": "PASS"}, {"ok": True}):
        p.write_text(json.dumps(value))
        assert rc.judge(0, p) == "PASS"
        assert rc.judge(1, p) == "FAIL"


def test_dry_run_never_launches_checks(monkeypatch, capsys):
    monkeypatch.setattr(rc.subprocess, "check_output", lambda *a, **k: "a" * 40)

    def forbidden(*a, **k):
        raise AssertionError("dry run launched a check")

    monkeypatch.setattr(rc.subprocess, "run", forbidden)
    monkeypatch.setattr(rc.subprocess, "Popen", forbidden)
    assert rc.main(["main", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Release checklist" in output
    assert output.count("PLANNED") == 5


def test_ci_failure_submits_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.subprocess, "check_output", lambda *a, **k: "a" * 40)

    class Failure:
        returncode = 1

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: Failure())
    monkeypatch.setattr(
        rc.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("submitted after CI failed")
        ),
    )
    assert rc.main(["--out", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "verdict.json").read_text())
    assert result["checks"]["gate"]["status"] == "NOT_RUN"


def test_all_verifiers_start_before_wait(monkeypatch, tmp_path):
    sha = "a" * 40
    monkeypatch.setattr(rc.subprocess, "check_output", lambda *a, **k: sha)
    events = []

    class Process:
        returncode = 0

        def wait(self):
            assert events == ["start"] * 4
            return 1

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: Process())

    def launch(*a, **k):
        events.append("start")
        return Process()

    monkeypatch.setattr(rc.subprocess, "Popen", launch)
    assert rc.main(["--out", str(tmp_path)]) == 1
    assert events == ["start"] * 4


def test_missing_executable_still_reports_all_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.subprocess, "check_output", lambda *a, **k: "a" * 40)

    class Success:
        returncode = 0

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: Success())
    launches = []
    monkeypatch.setattr(rc.time, "time_ns", lambda: 123)

    def missing(cmd, **kwargs):
        launches.append(cmd)
        if "--label-prefix" in cmd:
            assert "-123-" in cmd[cmd.index("--label-prefix") + 1]
        raise FileNotFoundError("missing verifier")

    monkeypatch.setattr(rc.subprocess, "Popen", missing)
    assert rc.main(["--out", str(tmp_path)]) == 1
    assert len(launches) == 4
    checks = json.loads((tmp_path / "verdict.json").read_text())["checks"]
    assert checks["ci-local"]["status"] == "PASS"
    assert all(checks[n]["status"] == "FAIL" for n in checks if n != "ci-local")


def test_success_requires_artifacts_but_not_finished_agentbench(monkeypatch, tmp_path):
    sha = "a" * 40
    monkeypatch.setattr(rc.subprocess, "check_output", lambda *a, **k: sha)

    class Success:
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: Success())

    def launch(cmd, **kwargs):
        name = Path(cmd[3]).name
        if name == "yv":
            path = tmp_path / "verify/runs" / f"gate-{sha[:12]}" / "verdict.json"
        elif name == "agentcompat":
            path = tmp_path / "agentcompat/verdict.json"
        elif name == "m3sweep":
            path = tmp_path / "m3/verdict.json"
            kwargs["stdout"].write(f"m3sweep PASS ({path})\n")
        else:
            kwargs["stdout"].write("collect later: agentbench --collect /bench/run\n")
            return Success()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"verdict": "PASS"}')
        return Success()

    monkeypatch.setattr(rc.subprocess, "Popen", launch)
    assert rc.main(["--out", str(tmp_path)]) == 0
    checks = json.loads((tmp_path / "verdict.json").read_text())["checks"]
    assert checks["agentbench"]["status"] == "PENDING"
    assert "--collect /bench/run" in checks["agentbench"]["evidence"]
    # Same SHA/output keeps the gate's successful stage evidence on a retry.
    assert rc.main(["--out", str(tmp_path)]) == 0
