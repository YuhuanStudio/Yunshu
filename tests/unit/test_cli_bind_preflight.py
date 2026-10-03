"""Occupied endpoints fail before checkpoint resolution or GPU loading."""

import importlib
import socket

import pytest
from typer.testing import CliRunner

from yunshu_cli import app
from yunshu_engine import settings

serve_module = importlib.import_module("yunshu_cli.serve")


@pytest.fixture(autouse=True)
def clean_settings():
    settings.clear_overrides()
    yield
    settings.clear_overrides()


def test_occupied_port_refuses_before_model_preflight(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        def must_not_load(*args):
            pytest.fail("checkpoint preflight ran before bind check")

        monkeypatch.setattr(serve_module, "_preflight_model", must_not_load)
        result = CliRunner().invoke(
            app, ["serve", "--model", "/missing/model", "--port", str(port)]
        )
        assert result.exit_code == 2
        assert "Cannot listen" in result.output
        assert "--port 8001" in result.output
        # The existing listener still belongs to its owner and remains usable.
        with socket.create_connection(("127.0.0.1", port)):
            pass


def test_free_address_and_ephemeral_port():
    serve_module._check_bind_address("127.0.0.1", 0)


@pytest.mark.parametrize("port", [-1, 65536])
def test_invalid_port_has_actionable_error(port):
    result = CliRunner().invoke(app, ["serve", "--port", str(port)])
    assert result.exit_code == 2
    assert "between 0 and 65535" in result.output


def test_uds_skips_tcp_probe(monkeypatch, tmp_path):
    def must_not_probe(*args):
        pytest.fail("TCP bind was checked for a Unix socket")

    monkeypatch.setattr(serve_module, "_check_bind_address", must_not_probe)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_module, "_rotate_service_log", lambda: None)
    result = CliRunner().invoke(app, ["serve", "--uds", str(tmp_path / "server.sock")])
    assert result.exit_code == 0, result.output
