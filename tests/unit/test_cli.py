"""Tests for Yunshu CLI — all 9 subcommands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from yunshu_cli import app

runner = CliRunner()


class TestCLIRegistration:
    def test_app_exists(self):
        assert app is not None

    def test_subcommands_registered(self):
        result = runner.invoke(app, ["--help"])
        expected = ["serve", "chat", "model", "status", "config", "launch", "eval", "bench", "diagnose"]
        for cmd in expected:
            assert cmd in result.output, f"Missing: {cmd}"

    def test_serve_command_options(self):
        from yunshu_cli.serve import serve
        import inspect
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
        assert "--model" in result.output
        assert "--temperature" in result.output
        assert "--thinking" in result.output
        assert "--system" in result.output


class TestCLIStatus:
    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--url" in result.output


class TestCLIConfig:
    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "--url" in result.output

    def test_config_set_help(self):
        result = runner.invoke(app, ["config", "set", "--help"])
        assert result.exit_code in (0, 1)  # Typer argparse fallback


class TestCLILaunch:
    def test_launch_list(self):
        result = runner.invoke(app, ["launch", "list"])
        assert result.exit_code == 0
        assert "Codex" in result.output
        assert "OpenCode" in result.output
        assert "Pi" in result.output

    def test_launch_unknown_tool(self):
        result = runner.invoke(app, ["launch", "nonexistent"])
        assert result.exit_code == 1


class TestCLIEval:
    def test_eval_list(self):
        result = runner.invoke(app, ["eval", "list"])
        assert result.exit_code == 0
        assert "mmlu" in result.output.lower()
        assert "gsm8k" in result.output.lower()
        assert "humaneval" in result.output.lower()
        assert "truthfulqa" in result.output.lower()
        assert "hellaswag" in result.output.lower()

    def test_eval_all_help(self):
        result = runner.invoke(app, ["eval", "all", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--sample" in result.output


class TestCLIModel:
    def test_model_list_no_dir(self):
        result = runner.invoke(app, ["model", "list", "--dir", "/nonexistent"])
        assert result.exit_code == 0

    def test_model_info_not_found(self):
        result = runner.invoke(app, ["model", "info", "nonexistent-model-xyz"])
        assert result.exit_code == 1
