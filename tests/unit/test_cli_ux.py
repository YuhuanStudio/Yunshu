"""CPU-only UX regression tests: no downloads, launchd or model execution."""

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from yunshu_cli import app
from yunshu_engine import settings

runner = CliRunner()
model = importlib.import_module("yunshu_cli.model")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    settings.clear_overrides()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YUNSHU_MODELS_DIR", raising=False)
    monkeypatch.delenv("YUNSHU_MODEL", raising=False)
    monkeypatch.delenv("YUNSHU_GATEWAY_URL", raising=False)
    monkeypatch.setattr(model, "scan_hf_cache", lambda: [])
    monkeypatch.setattr(model, "_detect_model_type", lambda p: "LLM")
    yield
    settings.clear_overrides()


def checkpoint(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"llama"}')
    (path / "model.safetensors").write_bytes(b"weights")
    return path


@pytest.mark.parametrize("shell", ["zsh", "bash", "fish"])
def test_completions(shell):
    r = runner.invoke(app, ["--json", "completion", shell])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["shell"] == shell
    assert "yunshu" in data["script"]


def test_completion_invalid():
    r = runner.invoke(app, ["--json", "completion", "invalid"])
    assert r.exit_code == 2
    assert "zsh" in json.loads(r.stdout)["error"]


def test_models_list_show_pull_and_rm(tmp_path):
    p = checkpoint(tmp_path / ".yunshu/models/org/test")
    for cmd in (
        ["models", "list", "--no-hf-cache"],
        ["models", "show", "org/test"],
        ["models", "pull", "org/test"],
    ):
        r = runner.invoke(app, ["--json", *cmd])
        assert r.exit_code == 0, r.output
        assert isinstance(json.loads(r.stdout), dict)
    r = runner.invoke(app, ["--json", "models", "rm", "org/test"])
    assert r.exit_code == 2
    assert p.exists()
    r = runner.invoke(app, ["--json", "models", "rm", "org/test", "--yes"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["removed"]
    assert not p.exists()


def test_rm_keeps_external_symlink(tmp_path):
    external = checkpoint(tmp_path / "external")
    base = tmp_path / ".yunshu/models"
    base.mkdir(parents=True)
    (base / "linked").symlink_to(external, target_is_directory=True)
    r = runner.invoke(app, ["--json", "models", "rm", "linked", "--yes"])
    assert r.exit_code == 2
    assert external.exists()


@pytest.mark.parametrize("repo", ["../test", "org/..", "org/.", "org\\bad/test"])
def test_pull_rejects_unsafe_repo(repo):
    r = runner.invoke(app, ["--json", "models", "pull", repo])
    assert r.exit_code == 2
    assert "repo id" in json.loads(r.stdout)["error"]


def test_setup_noninteractive_has_next_steps():
    r = runner.invoke(app, ["--json", "setup"])
    assert r.exit_code == 2
    assert len(json.loads(r.stdout)["next_steps"]) == 3


def test_setup_selects_complete_local_model(tmp_path, monkeypatch):
    p = checkpoint(tmp_path / ".yunshu/models/org/test")
    setup = importlib.import_module("yunshu_cli.setup")
    monkeypatch.setattr(setup, "_interactive", lambda: True)
    monkeypatch.setattr(setup.typer, "prompt", lambda *a, **kw: 1)
    r = runner.invoke(app, ["setup"])
    assert r.exit_code == 0, r.output
    assert str(p) in r.stdout


def test_service_logs_json(tmp_path, monkeypatch):
    service = importlib.import_module("yunshu_cli.service")
    log = tmp_path / "server.log"
    log.write_text("one\ntwo\nthree\n")
    monkeypatch.setattr(service, "log_file", lambda: log)
    r = runner.invoke(app, ["--json", "service", "logs", "-n", "2"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["lines"] == ["two", "three"]
    r = runner.invoke(app, ["--json", "service", "logs", "-f"])
    assert r.exit_code == 2
    assert "snapshot" in json.loads(r.stdout)["error"]


def test_top_json_uses_snapshot(monkeypatch):
    top = importlib.import_module("yunshu_cli.top")
    from yunshu_cli._output import emit

    calls = []
    monkeypatch.setattr(
        top, "status", lambda **kw: (calls.append(kw), emit({"healthy": True}))
    )
    r = runner.invoke(app, ["--json", "top", "--url", "http://localhost:18999"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["healthy"]
    assert calls == [{"url": "http://localhost:18999"}]


def test_json_output_does_not_leak_between_invocations():
    r = runner.invoke(app, ["--json", "models", "list", "--no-hf-cache"])
    assert json.loads(r.stdout)["models"] == []
    r = runner.invoke(app, ["models", "list", "--no-hf-cache"])
    assert "No models found" in r.stdout


def test_serve_without_model_is_actionable_json(monkeypatch):
    serve = importlib.import_module("yunshu_cli.serve")
    monkeypatch.setattr(serve, "_check_bind_address", lambda *a: None)
    r = runner.invoke(app, ["--json", "serve", "--port", "18999"])
    assert r.exit_code == 2
    assert json.loads(r.stdout)["next_steps"]


def test_json_chat_rejected_before_network():
    r = runner.invoke(app, ["--json", "chat"])
    assert r.exit_code == 2
    assert "complete" in json.loads(r.stdout)["error"]


def test_json_statusline(monkeypatch):
    module = importlib.import_module("yunshu_cli.statusline")
    monkeypatch.setattr(module, "fetch_status", lambda *a: None)
    monkeypatch.setattr(module, "_read_session", lambda: {})
    r = runner.invoke(app, ["--json", "statusline"])
    assert r.exit_code == 0, r.output
    assert "line" in json.loads(r.stdout)


def test_doctor_no_metal_and_new_storage_checks(monkeypatch):
    doctor = importlib.import_module("yunshu_cli.doctor")
    monkeypatch.setattr(doctor, "check_platform", lambda: [])
    monkeypatch.setattr(doctor, "check_mlx", lambda: pytest.fail("Metal probe ran"))
    r = runner.invoke(app, ["--json", "doctor", "--no-metal", "--port", "18999"])
    assert r.exit_code == 0, r.output
    checks = {c["name"]: c for c in json.loads(r.stdout)["checks"]}
    assert checks["mlx"]["status"] == "warn"
    assert checks["Evals storage"]["status"] == "ok"
    assert checks["stored completions"]["status"] == "ok"


def command_paths():
    from typer.main import get_command

    root = get_command(app)

    def walk(cmd, prefix):
        yield prefix
        for name, child in getattr(cmd, "commands", {}).items():
            yield from walk(child, [*prefix, name])

    return list(walk(root, []))


@pytest.mark.parametrize("command", command_paths())
def test_every_command_has_help(command):
    r = runner.invoke(app, [*command, "--help"])
    assert r.exit_code == 0, (command, r.output)
    assert "Usage" in r.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["config"],
        ["config", "path"],
        ["config", "set", "max_concurrent", "2"],
        ["config", "unset", "max_concurrent"],
    ],
)
def test_config_global_json(args):
    r = runner.invoke(app, ["--json", *args])
    assert r.exit_code == 0, r.output
    assert isinstance(json.loads(r.stdout), dict)


def test_config_unknown_setting_json():
    r = runner.invoke(app, ["--json", "config", "set", "made_up", "1"])
    assert r.exit_code == 2
    assert "Unknown setting" in json.loads(r.stdout)["error"]


def test_bare_serve_guides_first_run(monkeypatch):
    serve = importlib.import_module("yunshu_cli.serve")
    monkeypatch.setattr(serve, "_check_bind_address", lambda *a: None)
    r = runner.invoke(app, ["--json", "serve"])
    assert r.exit_code == 2
    assert "next_steps" in json.loads(r.stdout)


def test_top_live_refresh_and_interrupt(monkeypatch):
    module = importlib.import_module("yunshu_cli.top")
    calls = []
    monkeypatch.setattr(module.console, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(module, "status", lambda **kw: calls.append("status"))

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.time, "sleep", interrupt)
    r = runner.invoke(app, ["top"])
    assert r.exit_code == 0, r.output
    assert calls == ["clear", "status"]


def test_setup_downloads_selected_repo(tmp_path, monkeypatch):
    module = importlib.import_module("yunshu_cli.setup")
    monkeypatch.setattr(module, "_interactive", lambda: True)
    answers = iter([0, "org/test"])
    monkeypatch.setattr(module.typer, "prompt", lambda *a, **kw: next(answers))
    calls = []

    def fake_pull(repo, **kwargs):
        calls.append(repo)
        checkpoint(tmp_path / ".yunshu/models" / repo)

    monkeypatch.setattr(model, "pull", fake_pull)
    r = runner.invoke(app, ["setup"])
    assert r.exit_code == 0, r.output
    assert calls == ["org/test"]
    assert "yunshu serve -m" in r.stdout


def test_models_server_http_error_is_json(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: httpx.Response(
            401,
            json={"error": "unauthorized"},
            request=httpx.Request("GET", "http://localhost:18999/v1/models"),
        ),
    )
    r = runner.invoke(
        app, ["--json", "models", "list", "--url", "http://localhost:18999"]
    )
    assert r.exit_code == 1
    assert "401" in json.loads(r.stdout)["error"]


def test_doctor_no_metal_with_downloaded_model(tmp_path, monkeypatch):
    doctor = importlib.import_module("yunshu_cli.doctor")
    p = checkpoint(tmp_path / ".yunshu/models/org/test")
    monkeypatch.setattr(doctor, "check_platform", lambda: [])
    monkeypatch.setattr(doctor, "check_mlx", lambda: pytest.fail("Metal probe ran"))
    monkeypatch.setattr(
        model, "_detect_model_type", lambda p: pytest.fail("model type probe ran")
    )
    r = runner.invoke(
        app, ["--json", "doctor", "--no-metal", "-m", str(p), "--port", "18999"]
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["ok"]


def test_service_json_lifecycle(tmp_path, monkeypatch):
    import subprocess

    from yunshu_engine import paths

    service = importlib.import_module("yunshu_cli.service")
    p = checkpoint(tmp_path / ".yunshu/models/org/test")
    loaded = False
    calls = []

    def launchctl(*args):
        nonlocal loaded
        calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                args, 0 if loaded else 1, "state = running\npid = 123\n", ""
            )
        if args[0] == "bootstrap":
            loaded = True
        elif args[0] == "bootout":
            loaded = False
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(service, "_launchctl", launchctl)
    monkeypatch.setattr(service, "_healthy", lambda url: True)
    commands = [
        (["install", "-m", str(p), "--no-start"], "started", False),
        (["start"], "started", True),
        (["status"], "healthy", True),
        (["restart"], "restarted", True),
        (["stop"], "stopped", True),
        (["uninstall"], "removed", True),
    ]
    for args, key, expected in commands:
        r = runner.invoke(app, ["--json", "service", *args])
        assert r.exit_code == 0, (args, r.output)
        assert json.loads(r.stdout)[key] is expected
    assert p.exists()
    assert not paths.launch_agent_plist().exists()
    assert {args[0] for args in calls} >= {"print", "bootstrap", "kickstart", "bootout"}


def test_completion_protocol_lists_commands(monkeypatch):
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "_YUNSHU_COMPLETE": "complete_bash",
        "COMP_WORDS": "yunshu models ",
        "COMP_CWORD": "2",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "python"),
    }
    r = subprocess.run(
        [sys.executable, "-m", "yunshu_cli"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, r.stderr
    assert "list" in r.stdout and "pull" in r.stdout and "show" in r.stdout
