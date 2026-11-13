"""The console on this machine can change things: no static token (with or without stored keys)
does not lock the local operator out of the admin routes, a remote caller is still refused, and the
settings list shows exactly what PATCH accepts for the caller."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from yunshu_engine import settings
from yunshu_gateway import api_keys
from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers import admin_keys
from yunshu_gateway.routers import admin_settings as adm
from yunshu_gateway.routers import yunshu as yunshu_router


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "user_config_path", lambda: tmp_path / "config.toml")
    for name in settings.REGISTRY:
        monkeypatch.delenv(name, raising=False)
    settings.clear_overrides()
    api_keys.configure(tmp_path / "keys.json")
    yield
    api_keys._store = None
    settings.clear_overrides()


def _client(peer: str = "127.0.0.1", host: str = "127.0.0.1:8000") -> httpx.AsyncClient:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    for r in (adm.router, yunshu_router.router, admin_keys.router):
        app.include_router(r, prefix="/v1")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(peer, 5555)),
        base_url=f"http://{host}",
    )


PATCH = {"settings": {"YUNSHU_QUEUE_LIMIT": 7}}


@pytest.mark.asyncio
async def test_local_console_writes_settings_with_no_auth_configured():
    async with _client() as c:
        r = await c.patch("/v1/yunshu/config", json=PATCH)
        assert r.status_code == 200, r.text
        assert r.json()["results"]["YUNSHU_QUEUE_LIMIT"]["status"] == "applied"


@pytest.mark.asyncio
async def test_remote_caller_still_refused_with_no_auth():
    async with _client(peer="203.0.113.9", host="engine.lan:8000") as c:
        assert (await c.patch("/v1/yunshu/config", json=PATCH)).status_code == 401
    async with _client() as c:  # the console proxy forwards a LAN browser's address
        r = await c.patch(
            "/v1/yunshu/config", json=PATCH, headers={"x-forwarded-for": "192.168.1.5"}
        )
        assert r.status_code == 401
        r = await c.patch(  # a web page on another site, or a DNS-rebound name
            "/v1/yunshu/config", json=PATCH, headers={"origin": "http://evil.example"}
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_first_key_does_not_lock_the_local_console_out():
    async with _client() as c:
        made = await c.post(
            "/v1/yunshu/keys", json={"name": "app", "scopes": ["infer"]}
        )
        assert made.status_code == 201, made.text
        assert api_keys.get_store().has_keys()
        # stored keys switched auth on; the console (no key, loopback) still manages everything
        assert (await c.get("/v1/yunshu/keys")).status_code == 200
        assert (await c.patch("/v1/yunshu/config", json=PATCH)).status_code == 200
        # a wrong key is still refused, and so is a remote caller
        bad = await c.get("/v1/yunshu/keys", headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401
    async with _client(peer="203.0.113.9", host="engine.lan:8000") as c:
        assert (await c.get("/v1/yunshu/keys")).status_code == 401


@pytest.mark.asyncio
async def test_static_token_gates_the_local_console_too(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "sekret")
    async with _client() as c:
        assert (await c.patch("/v1/yunshu/config", json=PATCH)).status_code == 401
        ok = await c.patch(
            "/v1/yunshu/config", json=PATCH, headers={"Authorization": "Bearer sekret"}
        )
        assert ok.status_code == 200


@pytest.mark.asyncio
async def test_owner_sees_and_writes_every_setting_but_the_config_pointer():
    async with _client() as c:
        rows = (await c.get("/v1/yunshu/config", params={"include": "all"})).json()[
            "settings"
        ]
        locked = {r["name"] for r in rows if r["read_only"]}
        assert locked == {"YUNSHU_CONFIG"}
        for name in ("YUNSHU_MODELS_DIR", "YUNSHU_MCP_SERVERS", "YUNSHU_MODEL"):
            assert name in {r["name"] for r in rows}
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_MODELS_DIR": "/tmp/models-x"}},
        )
        assert r.status_code == 200, r.text
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_ADMIN_WRITE_EXEC_SETTINGS": True}},
        )
        assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_managed_admin_key_sees_host_settings_read_only():
    store = api_keys.get_store()
    _, secret = store.create("adm", ["admin"])
    async with _client() as c:
        h = {"Authorization": f"Bearer {secret}"}
        rows = (await c.get("/v1/yunshu/config", headers=h)).json()["settings"]
        ro = {r["name"] for r in rows if r["read_only"]}
        assert {"YUNSHU_MODELS_DIR", "YUNSHU_ADMIN_WRITE_EXEC_SETTINGS"} <= ro
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_ADMIN_WRITE_EXEC_SETTINGS": True}},
            headers=h,
        )
        assert r.status_code == 422
