"""No launchd job may be killed merely because it has the product label."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "deploy_service",
    Path(__file__).resolve().parents[2] / "scripts/research/deploy27b_service.py",
)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def test_owned_pid_requires_our_executable():
    assert (
        service.require_owned(
            "program = /Volumes/P5Plus/test/bin/python\n pid = 4321\n",
            Path("/Volumes/P5Plus/test/bin/python"),
        )
        == 4321
    )
    with pytest.raises(RuntimeError):
        service.require_owned(
            "program = /usr/bin/python\n pid = 4321\n",
            Path("/Volumes/P5Plus/test/bin/python"),
        )
    with pytest.raises(RuntimeError):
        service.require_owned(
            "program = /Volumes/P5Plus/test/bin/python\n",
            Path("/Volumes/P5Plus/test/bin/python"),
        )


def test_deployment_stage_is_opt_in_and_rejects_false_complete(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify.stages import _deployment_valid
    from verify.suites import LADDER, STAGES

    assert "deployment" in STAGES and "service_deploy" in STAGES
    assert "deployment" not in LADDER and "service_deploy" not in LADDER
    assert "deployment-idle" in STAGES and "deployment-idle" not in LADDER
    path = tmp_path / "result.jsonl"
    path.write_text('{"complete":true,"passed":false}\n')
    assert _deployment_valid(path)[0] is False
    path.write_text('{"complete":true,"passed":true}\n')
    assert _deployment_valid(path)[0] is True


def test_submitted_job_uses_isolated_plist_without_changing_user_home():
    job = {
        "Label": service.LABEL,
        "ProgramArguments": ["/test/python", "-m", "yunshu_cli", "serve"],
        "WorkingDirectory": "/Volumes/P5Plus/test/home",
        "EnvironmentVariables": {"HOME": "/Volumes/P5Plus/test/home", "PATH": "/bin"},
        "StandardOutPath": "/Volumes/P5Plus/test/server.log",
        "StandardErrorPath": "/Volumes/P5Plus/test/server.log",
    }
    args = service.submit_args(job)
    assert args[:3] == ["submit", "-l", service.LABEL]
    assert args[args.index("--") + 1 :][:2] == ["/test/python", "-c"]
    code = args[-1]
    assert "os.chdir" in code and "os.execve" in code
    assert "/Volumes/P5Plus/test/home" in code
    assert "'-m', 'yunshu_cli', 'serve'" in code
    assert "os.environ" not in code
    assert "-o" not in args and "-e" not in args
    assert "os.open" in code and "os.dup2" in code


def test_owned_inactive_job_can_be_cleaned_up_without_signalling_a_pid():
    state = "program = /Volumes/P5Plus/isolated/bin/python\n state = spawn scheduled\n"
    assert (
        service.require_owned_job(state, Path("/Volumes/P5Plus/isolated/bin/python"))
        is None
    )
    with pytest.raises(RuntimeError):
        service.require_owned_job(state, Path("/somebody/else/bin/python"))


def test_deployment_runs_exactly_one_hour_then_resident_load(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify import stages

    calls = []
    output = tmp_path / "complete.jsonl"
    output.write_text('{"complete":true,"passed":true}\n')

    def cells(batch):
        cell = batch[0]
        calls.append(cell)
        return {cell.key: SimpleNamespace(ok=True, evidence=output)}

    ctx = SimpleNamespace(
        py="python",
        cand=SimpleNamespace(path=tmp_path),
        model="27B",
        exe=SimpleNamespace(run_cells=cells),
    )
    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    monkeypatch.setattr(stages, "_failed_cells", lambda results: [])
    result = stages.stage_deployment(ctx)
    assert result.passed
    assert [cell.key for cell in calls] == ["pilot", "hour-1", "resident-load"]
