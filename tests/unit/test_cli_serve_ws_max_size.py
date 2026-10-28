"""`yunshu serve` sizes uvicorn's WebSocket frame limit from YUNSHU_MAX_REQUEST_SIZE.

uvicorn's ws_max_size defaults to 16 MiB, below the 32 MiB HTTP request default, so a
17-32 MiB response.create over `wss /v1/responses` (or a realtime / stream frame) was closed
with 1009 before the app saw it, while the same body over HTTP was accepted.
"""

import importlib

import pytest
from typer.testing import CliRunner

from yunshu_cli import app
from yunshu_engine import settings

serve = importlib.import_module("yunshu_cli.serve")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    settings.clear_overrides()
    monkeypatch.delenv("YUNSHU_MODEL", raising=False)
    monkeypatch.delenv("YUNSHU_MODELS_DIR", raising=False)
    monkeypatch.delenv("YUNSHU_MAX_REQUEST_SIZE", raising=False)
    monkeypatch.setattr(serve, "_check_bind_address", lambda *args: None)
    monkeypatch.setattr(serve, "_preflight_model", lambda *args: None)
    monkeypatch.setattr(serve, "_rotate_service_log", lambda: None)
    yield
    settings.clear_overrides()


def _serve_kwargs(monkeypatch, tmp_path, *extra):
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    path = tmp_path / "m"
    path.mkdir()
    result = CliRunner().invoke(app, ["serve", str(path), *extra])
    assert result.exit_code == 0, result.output
    return calls[0][1]


def test_ws_max_size_follows_the_request_size_default(monkeypatch, tmp_path):
    kwargs = _serve_kwargs(monkeypatch, tmp_path)
    assert kwargs["ws_max_size"] == 32 * 1024 * 1024


def test_ws_max_size_follows_an_explicit_request_size(monkeypatch, tmp_path):
    monkeypatch.setenv("YUNSHU_MAX_REQUEST_SIZE", str(64 * 1024 * 1024))
    kwargs = _serve_kwargs(monkeypatch, tmp_path)
    assert kwargs["ws_max_size"] == 64 * 1024 * 1024
