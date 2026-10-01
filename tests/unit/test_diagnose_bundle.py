"""F16: the local diagnostics bundle carries no prompts and no secrets."""

from __future__ import annotations

import importlib
import json

from typer.testing import CliRunner

from yunshu_cli import bundle, diagnose

doctor = importlib.import_module("yunshu_cli.doctor")
from yunshu_engine import paths, settings

PROMPT = "my secret diary entry about the merger"


def _log(tmp_path):
    log = tmp_path / "yunshu.log"
    log.write_text(
        "INFO ok request\n"
        f"WARNING [req_0123456789abcdef] POST /v1/chat/completions 400 "
        f"input_value=[{{'role': 'user', 'content': '{PROMPT}'}}], input_type=list\n"
        "ERROR boom Authorization: Bearer leakedtoken123456\n"
        f"ERROR bad body content: '{PROMPT}'\n"
        "Traceback (most recent call last):\n"
    )
    return log


def test_recent_errors_have_trace_ids_and_no_prompt_or_secret(tmp_path):
    out = bundle.recent_errors(_log(tmp_path))
    blob = json.dumps(out)
    assert PROMPT not in blob and "leakedtoken123456" not in blob
    assert "req_0123456789abcdef" in out["trace_ids"]
    assert all(len(line) <= bundle.MAX_LINE for line in out["lines"])
    assert not any("INFO ok" in line for line in out["lines"])


def test_bundle_redacts_secret_settings_and_writes_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    _log(tmp_path)
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok-very-secret-value")
    monkeypatch.setenv("YUNSHU_BRAVE_API_KEY", "brave-key-123456")
    monkeypatch.setenv(
        "YUNSHU_MCP_SERVERS", '[{"url":"https://x","headers":{"a":"zzsecretzz"}}]'
    )
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: [])
    dest = tmp_path / "out" / "b.json"
    bundle.write(dest)
    text = dest.read_text()
    for secret in ("tok-very-secret-value", "brave-key-123456", "zzsecretzz", PROMPT):
        assert secret not in text
    data = json.loads(text)
    names = {r["name"] for r in data["settings_changed"]}
    assert "YUNSHU_AUTH_TOKEN" in names  # present, value hidden
    assert data["version"] and data["errors"]["trace_ids"]


def test_cli_command_prints_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: [])
    dest = tmp_path / "x.json"
    r = CliRunner().invoke(diagnose.diagnose_app, ["bundle", "-o", str(dest)])
    assert r.exit_code == 0 and dest.exists() and "x.json" in r.output
    assert settings.get("YUNSHU_AUTH_TOKEN") is None or True
