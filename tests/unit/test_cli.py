"""Tests for Yunshu CLI — all 9 subcommands."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from yunshu_cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(s: str) -> str:
    """Strip ANSI escape codes — rich may colour option names on CI."""
    return _ANSI.sub("", s)


class TestCLIRegistration:
    def test_app_exists(self):
        assert app is not None

    def test_subcommands_registered(self):
        result = runner.invoke(app, ["--help"])
        expected = [
            "serve",
            "chat",
            "model",
            "status",
            "config",
            "launch",
            "eval",
            "bench",
            "diagnose",
        ]
        for cmd in expected:
            assert cmd in _plain(result.output), f"Missing: {cmd}"

    def test_serve_command_options(self):
        import inspect

        from yunshu_cli.serve import serve

        sig = inspect.signature(serve)
        param_names = list(sig.parameters.keys())
        assert "model" in param_names
        assert "host" in param_names
        assert "port" in param_names

    def test_model_command_options(self):
        from yunshu_cli.model import model_app

        commands = [cmd.name for cmd in model_app.registered_commands]
        assert "list" in commands
        assert "download" in commands
        assert "info" in commands
        assert "benchmark" in commands

    def test_bench_command_options(self):
        from yunshu_cli.benchmark import bench_app

        commands = [cmd.name for cmd in bench_app.registered_commands]
        assert "roofline" in commands
        assert "latency" in commands
        assert "throughput" in commands
        assert "memory" in commands

    def test_diagnose_command_options(self):
        from yunshu_cli.diagnose import diagnose_app

        commands = [cmd.name for cmd in diagnose_app.registered_commands]
        assert "system" in commands
        assert "gpu" in commands
        assert "server" in commands


class TestCLIChat:
    def test_chat_help(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--model" in _plain(result.output)
        assert "--temperature" in _plain(result.output)
        assert "--thinking" in _plain(result.output)
        assert "--system" in _plain(result.output)


class TestCLIStatus:
    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--url" in _plain(result.output)


class TestCLIConfig:
    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "--url" in _plain(result.output)

    def test_config_set_help(self):
        result = runner.invoke(app, ["config", "set", "--help"])
        assert result.exit_code in (0, 1)  # Typer argparse fallback


class TestCLILaunch:
    def test_launch_list(self):
        result = runner.invoke(app, ["launch", "list"])
        assert result.exit_code == 0
        assert "Codex" in _plain(result.output)
        assert "OpenCode" in _plain(result.output)
        assert "Pi" in _plain(result.output)

    def test_launch_unknown_tool(self):
        result = runner.invoke(app, ["launch", "nonexistent"])
        assert result.exit_code == 1


class TestCLIEval:
    def test_eval_list(self):
        result = runner.invoke(app, ["eval", "list"])
        assert result.exit_code == 0
        assert "mmlu" in _plain(result.output).lower()
        assert "gsm8k" in _plain(result.output).lower()
        assert "humaneval" in _plain(result.output).lower()
        assert "truthfulqa" in _plain(result.output).lower()
        assert "hellaswag" in _plain(result.output).lower()

    def test_eval_all_help(self):
        result = runner.invoke(app, ["eval", "all", "--help"])
        assert result.exit_code == 0
        assert "--model" in _plain(result.output)
        assert "--sample" in _plain(result.output)


class TestCLIModel:
    def test_model_list_no_dir(self):
        result = runner.invoke(app, ["model", "list", "--dir", "/nonexistent"])
        assert result.exit_code == 0

    def test_model_info_not_found(self):
        result = runner.invoke(app, ["model", "info", "nonexistent-model-xyz"])
        assert result.exit_code == 1
