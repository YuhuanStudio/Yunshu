"""The console_admin_config route check runs against the real admin routers + live CORS
middleware on a throwaway HOME (CPU only), so a bug in the check itself shows up before a
real-server run. launchctl is faked: nothing here can reach the operator's launchd."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import route_checks as rc  # noqa: E402

from yunshu_engine import settings  # noqa: E402
from yunshu_gateway.middleware.live_cors import LiveCORSMiddleware  # noqa: E402
from yunshu_gateway.routers import admin_settings as adm  # noqa: E402
from yunshu_gateway.routers import yunshu as yunshu_router  # noqa: E402


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # a shared conftest may redirect the user config; the real server resolves it under HOME
    monkeypatch.setattr(
        settings, "user_config_path", lambda: tmp_path / ".yunshu" / "config.toml"
    )
    for name in settings.REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    settings.clear_overrides()
    calls: list = []

    def fake_launchctl(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 113, "", "no such service")

    monkeypatch.setattr(adm, "_launchctl", fake_launchctl)
    app = FastAPI()
    app.add_middleware(LiveCORSMiddleware)
    app.include_router(adm.router, prefix="/v1")
    app.include_router(yunshu_router.router, prefix="/v1")
    app.add_api_route("/health/live", lambda: {"status": "ok"})
    http = TestClient(app, base_url="http://test")
    yield (
        rc.Ctx(
            url="http://test",
            token="",
            model="m",
            kind="text",
            http=http,
            home=str(tmp_path),
        ),
        calls,
    )
    settings.clear_overrides()


def test_admin_config_check_passes_and_restores_the_throwaway_config(ctx, tmp_path):
    c, calls = ctx
    rc.REGISTRY["console_admin_config"].fn(c)
    cfg = tmp_path / ".yunshu" / "config.toml"
    assert "YUNSHU_BATCH_MAX_ITEMS" not in (cfg.read_text() if cfg.exists() else "")
    assert not any(a and a[0] == "kickstart" for a in calls), "restart must never fire"


def test_admin_config_check_refuses_a_server_on_another_home(ctx, tmp_path):
    c, _ = ctx
    c.home = str(tmp_path / "somewhere-else")
    with pytest.raises(rc.Fail, match="throwaway HOME"):
        rc.REGISTRY["console_admin_config"].fn(c)
    assert not (tmp_path / ".yunshu").exists(), (
        "nothing may be written before the guard"
    )


def test_downloads_check_skips_offline(ctx):
    c, _ = ctx
    c.online = False
    with pytest.raises(rc.Skip, match="offline"):
        rc.REGISTRY["console_downloads"].fn(c)


def test_exempt_list_is_only_the_documented_501():
    # Only routes that need a model checkpoint or a push channel are exempt, each with a reason.
    assert set(rc.EXEMPT) == {
        "POST /api/push",
        "POST /v1/decisions",
        "POST /v1/systemone",
    }
    assert all(reason.strip() for reason in rc.EXEMPT.values())
