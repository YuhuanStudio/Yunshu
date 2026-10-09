"""How the console process is started: as a sibling of `yunshu serve`, standalone, and as its own
launchd job next to the engine's."""

from __future__ import annotations

import plistlib
import subprocess
import sys

from yunshu_cli import console_cmd, serve, service
from yunshu_engine import paths, settings


def test_the_engine_job_never_starts_a_console_and_the_console_job_is_its_own():
    engine = service.serve_args("org/m", None, "127.0.0.1", 8000, None, [])
    assert "--no-console" in engine and engine[3] == "serve"
    console = service.console_args("0.0.0.0", 8000, 8100, "/etc/y.toml")
    assert console[1:3] == ["-m", "yunshu_console"]
    assert console[console.index("--engine") + 1] == "http://127.0.0.1:8000"
    assert console[console.index("--host") + 1] == "0.0.0.0"
    assert console[console.index("--port") + 1] == "8100"
    assert console[console.index("--config") + 1].endswith("y.toml")


def test_the_console_job_is_kept_alive_always_and_logs_apart_from_the_engine():
    engine = service.build_plist(["x"])
    console = service.build_plist(
        ["y"],
        label=paths.CONSOLE_SERVICE_LABEL,
        log_path=service.console_log_file(),
        keep_alive_always=True,
    )
    assert (
        engine["Label"] == paths.SERVICE_LABEL
        and console["Label"] == paths.CONSOLE_SERVICE_LABEL
    )
    assert engine["KeepAlive"] == {"SuccessfulExit": False}
    assert console["KeepAlive"] is True
    assert console["StandardOutPath"].endswith("console.log")
    assert engine["StandardOutPath"].endswith("yunshu.log")
    plistlib.dumps(console)  # a valid plist
    assert (
        paths.console_launch_agent_plist().name
        == "com.yuhuanstudio.yunshu.console.plist"
    )


def test_install_dry_run_shows_both_agents_and_no_console_shows_one(monkeypatch):
    from typer.testing import CliRunner

    from yunshu_cli import app

    monkeypatch.setattr(sys, "platform", "darwin")
    runner = CliRunner()
    both = runner.invoke(
        app, ["--json", "service", "install", "--models-dir", "/tmp", "--dry-run"]
    )
    assert both.exit_code == 0, both.output
    import json

    out = json.loads(both.stdout)
    assert out["console"]["plist"]["Label"] == paths.CONSOLE_SERVICE_LABEL
    assert "--no-console" in out["plist"]["ProgramArguments"]
    one = runner.invoke(
        app,
        [
            "--json",
            "service",
            "install",
            "--models-dir",
            "/tmp",
            "--dry-run",
            "--no-console",
        ],
    )
    assert "console" not in json.loads(one.stdout)


class FakeChild:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_serve_starts_the_console_as_a_sibling_process(monkeypatch):
    started: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda args, **kw: started.append(args) or FakeChild()
    )
    monkeypatch.setattr(console_cmd, "console_port_busy", lambda host, port: False)
    monkeypatch.setattr("atexit.register", lambda fn: None)
    serve._start_console_sibling("127.0.0.1", 8000)
    assert len(started) == 1
    args = started[0]
    assert args[1:3] == ["-m", "yunshu_console"]
    assert args[args.index("--engine") + 1] == "http://127.0.0.1:8000"
    assert args[args.index("--port") + 1] == "8100"


def test_no_console_and_an_already_running_console_start_nothing(monkeypatch):
    started: list = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda args, **kw: started.append(args) or FakeChild()
    )
    monkeypatch.setattr("atexit.register", lambda fn: None)
    monkeypatch.setattr(console_cmd, "console_port_busy", lambda host, port: True)
    serve._start_console_sibling("127.0.0.1", 8000)
    assert started == [], "the service's console is already there"
    monkeypatch.setattr(console_cmd, "console_port_busy", lambda host, port: False)
    settings.set_override("YUNSHU_CONSOLE", False)
    try:
        serve._start_console_sibling("127.0.0.1", 8000)
    finally:
        settings.clear_overrides()
    assert started == []


def test_the_sibling_is_stopped_with_the_engine_and_a_failed_start_is_not_fatal(
    monkeypatch,
):
    child = FakeChild()
    hooks: list = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: child)
    monkeypatch.setattr(console_cmd, "console_port_busy", lambda host, port: False)
    monkeypatch.setattr("atexit.register", lambda fn: hooks.append(fn))
    serve._start_console_sibling("127.0.0.1", 8000)
    hooks[0]()
    assert child.terminated

    def boom(args, **kw):
        raise OSError("no exec")

    monkeypatch.setattr(subprocess, "Popen", boom)
    serve._start_console_sibling("127.0.0.1", 8000)  # warns, returns


def test_the_console_settings_have_the_documented_defaults():
    assert settings.get("YUNSHU_CONSOLE") is True
    assert settings.get("YUNSHU_CONSOLE_PORT") == 8100
    assert settings.get("YUNSHU_CONSOLE_POLL_S") == 1.0
    assert settings.get("YUNSHU_CONSOLE_HISTORY") is True
    assert settings.get("YUNSHU_CONSOLE_RETENTION_DAYS") == 30.0
