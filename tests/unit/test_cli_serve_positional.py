"""Positional model references preserve the established option/config path."""

import importlib
import json

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
    monkeypatch.setattr(serve, "_check_bind_address", lambda *args: None)
    monkeypatch.setattr(serve, "_preflight_model", lambda *args: None)
    monkeypatch.setattr(serve, "_rotate_service_log", lambda: None)
    yield
    settings.clear_overrides()


@pytest.mark.parametrize("style", ["positional", "option", "both"])
def test_model_reference_reaches_existing_server_settings(tmp_path, monkeypatch, style):
    path = tmp_path / "local-model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    args = ["serve"]
    if style in ("positional", "both"):
        args.append(str(path))
    if style in ("option", "both"):
        args += ["-m", str(path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert settings.get("YUNSHU_MODEL") == str(path)
    assert len(calls) == 1


def test_conflicting_models_fail_before_preflight(monkeypatch):
    def must_not_run(*args):
        pytest.fail("ambiguous model started server work")

    monkeypatch.setattr(serve, "_preflight_model", must_not_run)
    result = CliRunner().invoke(app, ["serve", "org/first", "-m", "org/second"])
    assert result.exit_code == 2
    assert "disagree" in result.output
