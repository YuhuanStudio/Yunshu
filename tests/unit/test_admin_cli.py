"""Unit tests for admin CLI."""
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from yunshu_cli.admin import admin_app

runner = CliRunner()


class TestAdminCLI:
    def test_help(self):
        result = runner.invoke(admin_app, ["--help"])
        assert result.exit_code == 0
        assert "models" in result.output
        assert "keys" in result.output
        assert "mesh" in result.output

    def test_models_list_no_server(self):
        result = runner.invoke(admin_app, ["models", "list", "--url", "http://localhost:1"])
        assert result.exit_code == 1

    def test_keys_list_no_server(self):
        result = runner.invoke(admin_app, ["keys", "list", "--url", "http://localhost:1"])
        assert result.exit_code == 1

    def test_mesh_status_no_server(self):
        result = runner.invoke(admin_app, ["mesh", "status", "--url", "http://localhost:1"])
        assert result.exit_code == 1

    def test_models_subcommands(self):
        result = runner.invoke(admin_app, ["models", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "discover" in result.output
        assert "load" in result.output
        assert "unload" in result.output

    def test_keys_subcommands(self):
        result = runner.invoke(admin_app, ["keys", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "create" in result.output
        assert "revoke" in result.output

    def test_config_no_server(self):
        result = runner.invoke(admin_app, ["config", "--url", "http://localhost:1"])
        assert result.exit_code == 1

    def test_models_list_with_mock(self):
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": [
                {"id": "qwen-7b", "owned_by": "yunshu", "object": "model"},
            ]}
            mock_get.return_value = mock_resp
            result = runner.invoke(admin_app, ["models", "list"])
            assert result.exit_code == 0
            assert "qwen-7b" in result.output

    def test_keys_create_no_server(self):
        result = runner.invoke(admin_app, ["keys", "create", "test-key", "--url", "http://localhost:1"])
        assert result.exit_code == 1
