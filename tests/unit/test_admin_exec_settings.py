"""Settings that run host code or touch arbitrary host paths are read-only over HTTP
unless the host operator sets YUNSHU_ADMIN_WRITE_EXEC_SETTINGS (admin scope != host RCE)."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from yunshu_engine import settings
from yunshu_gateway.routers import admin_settings as adm
from yunshu_gateway.routers import yunshu as yunshu_router


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    cfg = tmp_path / "user-config.toml"
    monkeypatch.setattr(settings, "user_config_path", lambda: cfg)
    for name in settings.REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    settings.clear_overrides()
    yield
    settings.clear_overrides()


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(adm.router, prefix="/v1")
    app.include_router(yunshu_router.router, prefix="/v1")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value",
    [
        ("YUNSHU_MCP_SERVERS", '[{"name": "x", "command": "sh", "args": ["-c", "id"]}]'),
        ("YUNSHU_TRUST_REMOTE_CODE", True),
        ("YUNSHU_AUDIT_LOG_FILE", "/etc/cron.d/x"),
        ("YUNSHU_SERVE_LOG_DIR", "/tmp/x"),
        ("YUNSHU_CODE_EXECUTION", True),
        ("YUNSHU_ALLOW_LOCAL_FILES", True),
    ],
)
async def test_patch_refuses_host_access_settings(name, value):
    async with _client() as c:
        r = await c.patch("/v1/yunshu/config", json={"settings": {name: value}})
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "exec_setting_read_only"
        assert name in r.json()["detail"]["settings"]
        # a reset only removes a value, so it stays allowed
        ok = await c.patch("/v1/yunshu/config", json={"settings": {name: None}})
        assert ok.status_code == 200


@pytest.mark.asyncio
async def test_patch_opt_in_exists_and_is_not_self_writable(monkeypatch):
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_ADMIN_WRITE_EXEC_SETTINGS": True}},
        )
        assert r.status_code == 422  # only the host operator may enable it
        monkeypatch.setenv("YUNSHU_ADMIN_WRITE_EXEC_SETTINGS", "true")
        ok = await c.patch(
            "/v1/yunshu/config", json={"settings": {"YUNSHU_TRUST_REMOTE_CODE": True}}
        )
        assert ok.status_code == 200


@pytest.mark.asyncio
async def test_get_flags_host_access_rows_read_only():
    async with _client() as c:
        rows = {
            r["name"]: r for r in (await c.get("/v1/yunshu/config")).json()["settings"]
        }
    assert rows["YUNSHU_MCP_SERVERS"]["read_only"] is True
    assert rows["YUNSHU_AUDIT_LOG_FILE"]["read_only"] is True
    assert rows["YUNSHU_QUEUE_LIMIT"]["read_only"] is False
