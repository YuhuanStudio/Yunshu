"""Console admin routes: settings writes (PATCH /v1/yunshu/config), the service
and its restart, CORS origins, and the `applies` class of every setting."""

from __future__ import annotations

import asyncio
import os
import subprocess

import httpx
import pytest
from fastapi import FastAPI

from yunshu_engine import settings
from yunshu_gateway.middleware.live_cors import LiveCORSMiddleware
from yunshu_gateway.routers import admin_settings as adm
from yunshu_gateway.routers import yunshu as yunshu_router


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Settings written to a temp file; no launchctl; no env bleed."""
    cfg = tmp_path / "user-config.toml"
    monkeypatch.setattr(settings, "user_config_path", lambda: cfg)
    for name in settings.REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", "0")
    settings.clear_overrides()
    calls: list[tuple] = []

    def fake_launchctl(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 113, "", "no such service")

    monkeypatch.setattr(adm, "_launchctl", fake_launchctl)
    yield calls
    settings.clear_overrides()


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.add_middleware(LiveCORSMiddleware)
    app.include_router(adm.router, prefix="/v1")
    app.include_router(yunshu_router.router, prefix="/v1")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# ── classes ────────────────────────────────────────────────────────────


def test_every_setting_has_an_applies_class():
    missing = [
        n
        for n, s in settings.REGISTRY.items()
        if s.applies not in settings.APPLIES_CLASSES
    ]
    assert not missing, (
        f"classify these in settings.py (_LIVE/_RELOAD/_RESTART): {missing}"
    )
    groups = settings._LIVE | settings._RELOAD | settings._RESTART
    assert groups == set(settings.REGISTRY)
    assert len(settings._LIVE) + len(settings._RELOAD) + len(settings._RESTART) == len(
        groups
    )


def test_known_classes_by_read_site():
    r = settings.REGISTRY
    assert r["YUNSHU_QUEUE_LIMIT"].applies == "live"  # admit() per request
    assert r["YUNSHU_CORS_ORIGINS"].applies == "live"
    assert r["YUNSHU_MTP"].applies == "reload"  # spec_select at runner build
    assert r["YUNSHU_KV_QUANT_BITS"].applies == "reload"
    assert r["YUNSHU_RATE_LIMIT_RPM"].applies == "restart"  # middleware __init__
    assert r["YUNSHU_MAX_REQUEST_SIZE"].applies == "restart"  # create_app


@pytest.mark.asyncio
async def test_config_get_exposes_applies_and_limits():
    async with _client() as c:
        j = (await c.get("/v1/yunshu/config?include=all")).json()
    rows = {s["name"]: s for s in j["settings"]}
    assert all(s["applies"] in ("live", "reload", "restart") for s in rows.values())
    assert rows["YUNSHU_QUEUE_LIMIT"]["minimum"] == 0
    assert rows["YUNSHU_BRAVE_API_KEY"]["secret"] is True


# ── PATCH config ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_live_setting_applies_now_and_persists(tmp_path):
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config", json={"settings": {"YUNSHU_QUEUE_LIMIT": 7}}
        )
        assert r.status_code == 200, r.text
        row = r.json()["results"]["YUNSHU_QUEUE_LIMIT"]
        assert (
            row["status"] == "applied" and row["value"] == 7 and row["source"] == "file"
        )
        assert settings.get("YUNSHU_QUEUE_LIMIT") == 7  # live: read back immediately
        assert 'YUNSHU_QUEUE_LIMIT = "7"' in (tmp_path / "user-config.toml").read_text()
        got = (await c.get("/v1/yunshu/config")).json()
        q = next(s for s in got["settings"] if s["name"] == "YUNSHU_QUEUE_LIMIT")
        assert q["value"] == 7 and q["source"] == "file"


@pytest.mark.asyncio
async def test_patch_reports_reload_and_restart():
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config",
            json={
                "settings": {"YUNSHU_MTP_BLOCK_SIZE": 4, "YUNSHU_RATE_LIMIT_RPM": 60}
            },
        )
    j = r.json()
    assert j["results"]["YUNSHU_MTP_BLOCK_SIZE"]["status"] == "needs_reload"
    assert j["results"]["YUNSHU_RATE_LIMIT_RPM"]["status"] == "needs_restart"
    assert j["restart_required"] is True and j["restart_for"] == [
        "YUNSHU_RATE_LIMIT_RPM"
    ]
    assert j["reload_required"] is True
    assert j["restart"]["available"] is False and j["restart"]["manual"]


@pytest.mark.asyncio
async def test_patch_validation_is_all_or_nothing(tmp_path):
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config",
            json={
                "settings": {
                    "YUNSHU_QUEUE_LIMIT": 9,
                    "YUNSHU_MAX_LORAS": 0,  # below minimum 1
                    "YUNSHU_SPEC_TREE": "nope",  # not a choice
                    "YUNSHU_NOPE": 1,
                    "YUNSHU_MTP": "maybe",
                    "YUNSHU_CONFIG": "/x",
                }
            },
        )
    assert r.status_code == 422
    errs = r.json()["detail"]["errors"]
    assert set(errs) == {
        "YUNSHU_MAX_LORAS",
        "YUNSHU_SPEC_TREE",
        "YUNSHU_NOPE",
        "YUNSHU_MTP",
        "YUNSHU_CONFIG",
    }
    assert not (tmp_path / "user-config.toml").exists()


@pytest.mark.asyncio
async def test_patch_experimental_needs_confirmation():
    exp = next(n for n, s in settings.REGISTRY.items() if s.stability == "experimental")
    s = settings.REGISTRY[exp]
    value = True if s.type == "bool" else (s.choices[0] if s.choices else s.default)
    async with _client() as c:
        r = await c.patch("/v1/yunshu/config", json={"settings": {exp: value}})
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "experimental_unconfirmed"
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {exp: value}, "confirm_experimental": True},
        )
        assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_patch_null_resets_to_default():
    async with _client() as c:
        await c.patch("/v1/yunshu/config", json={"settings": {"YUNSHU_QUEUE_LIMIT": 7}})
        r = await c.patch(
            "/v1/yunshu/config", json={"settings": {"YUNSHU_QUEUE_LIMIT": None}}
        )
    assert r.json()["results"]["YUNSHU_QUEUE_LIMIT"]["reset"] is True
    assert settings.get("YUNSHU_QUEUE_LIMIT") == 64


@pytest.mark.asyncio
async def test_patch_environment_still_wins(monkeypatch):
    monkeypatch.setenv("YUNSHU_QUEUE_LIMIT", "3")
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config", json={"settings": {"YUNSHU_QUEUE_LIMIT": 7}}
        )
    row = r.json()["results"]["YUNSHU_QUEUE_LIMIT"]
    assert (
        row["status"] == "overridden" and row["source"] == "env" and row["value"] == 3
    )
    assert settings.get("YUNSHU_QUEUE_LIMIT") == 3


@pytest.mark.asyncio
async def test_patch_secrets_are_write_only(tmp_path):
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_BRAVE_API_KEY": "hunter2-secret"}},
        )
        assert r.status_code == 200
        assert (
            "hunter2" not in r.text
            and "value" not in r.json()["results"]["YUNSHU_BRAVE_API_KEY"]
        )
        assert "hunter2" not in (await c.get("/v1/yunshu/config")).text
        bad = await c.patch(
            "/v1/yunshu/config", json={"settings": {"YUNSHU_BRAVE_API_KEY": "  "}}
        )
        assert bad.status_code == 422


@pytest.mark.asyncio
async def test_patch_dry_run_writes_nothing(tmp_path):
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_QUEUE_LIMIT": 5}, "dry_run": True},
        )
    assert r.status_code == 200 and r.json()["saved_to"] is None
    assert not (tmp_path / "user-config.toml").exists()


@pytest.mark.asyncio
async def test_patch_writes_the_active_config_file(tmp_path):
    explicit = tmp_path / "custom.toml"
    explicit.write_text('YUNSHU_MAX_LORAS = "2"\n')
    settings.set_override("YUNSHU_CONFIG", str(explicit))
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config", json={"settings": {"YUNSHU_QUEUE_LIMIT": 5}}
        )
    assert r.json()["saved_to"] == str(explicit)
    text = explicit.read_text()
    assert "YUNSHU_QUEUE_LIMIT" in text and "YUNSHU_MAX_LORAS" in text  # others kept


@pytest.mark.asyncio
async def test_patch_rejects_empty_and_bad_body():
    async with _client() as c:
        assert (
            await c.patch("/v1/yunshu/config", json={"settings": {}})
        ).status_code == 422
        assert (await c.patch("/v1/yunshu/config", json={})).status_code == 422


# ── permissions ────────────────────────────────────────────────────────

ADMIN_CALLS = [
    ("PATCH", "/v1/yunshu/config", {"settings": {"YUNSHU_QUEUE_LIMIT": 5}}),
    ("GET", "/v1/yunshu/service", None),
    ("POST", "/v1/yunshu/service/restart", {"confirm": True}),
    ("GET", "/v1/yunshu/cors", None),
    ("PATCH", "/v1/yunshu/cors", {"origins": ["http://localhost:3000"]}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
async def test_admin_routes_denied_without_auth(monkeypatch, method, path, body):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED")
    async with _client() as c:
        r = await c.request(method, path, json=body)
    assert r.status_code == 401  # no token configured: privileged ops are denied


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
async def test_admin_routes_need_the_token(monkeypatch, method, path, body):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED")
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "s3cret")
    async with _client() as c:
        assert (await c.request(method, path, json=body)).status_code == 401
        bad = await c.request(
            method, path, json=body, headers={"Authorization": "Bearer x"}
        )
        assert bad.status_code == 401
        ok = await c.request(
            method, path, json=body, headers={"Authorization": "Bearer s3cret"}
        )
        assert ok.status_code not in (401, 403)


# ── service ────────────────────────────────────────────────────────────

PRINT = "state = running\n\tpid = {pid}\n\tlast exit code = (never exited)\n"


@pytest.mark.asyncio
async def test_service_not_installed(_isolated):
    async with _client() as c:
        j = (await c.get("/v1/yunshu/service")).json()
    assert j["loaded"] is False and j["under_launchd"] is False
    assert j["label"] == "com.yuhuanstudio.yunshu" and j["plist"].endswith(".plist")
    assert (
        j["version"]
        and j["uptime_s"] >= 0
        and j["cli"]["restart"] == "yunshu service restart"
    )
    assert j["restart_note"]


@pytest.mark.asyncio
async def test_service_under_launchd(monkeypatch):
    monkeypatch.setattr(
        adm,
        "_launchctl",
        lambda *a: subprocess.CompletedProcess(a, 0, PRINT.format(pid=os.getpid()), ""),
    )
    async with _client() as c:
        j = (await c.get("/v1/yunshu/service")).json()
    assert j["loaded"] and j["pid"] == os.getpid() and j["under_launchd"] is True
    assert j["state"] == "running" and j["restart_available"] is True


@pytest.mark.asyncio
async def test_service_loaded_but_another_process_owns_it(monkeypatch):
    monkeypatch.setattr(
        adm,
        "_launchctl",
        lambda *a: subprocess.CompletedProcess(
            a, 0, PRINT.format(pid=os.getpid() + 1), ""
        ),
    )
    async with _client() as c:
        j = (await c.get("/v1/yunshu/service")).json()
        r = await c.post("/v1/yunshu/service/restart", json={"confirm": True})
    assert j["loaded"] and j["under_launchd"] is False
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_restart_refused_outside_launchd(_isolated):
    async with _client() as c:
        r = await c.post("/v1/yunshu/service/restart", json={"confirm": True})
    d = r.json()["detail"]
    assert r.status_code == 409 and d["code"] == "not_under_launchd" and d["manual"]
    assert not any(a[0] == "kickstart" for a in _isolated)  # never touched launchd


@pytest.mark.asyncio
async def test_restart_under_launchd_drains_then_kickstarts(monkeypatch):
    calls: list[tuple] = []

    def fake(*a):
        calls.append(a)
        if a[0] == "print":
            return subprocess.CompletedProcess(a, 0, PRINT.format(pid=os.getpid()), "")
        return subprocess.CompletedProcess(a, 0, "", "")

    monkeypatch.setattr(adm, "_launchctl", fake)
    active = iter([2, 1, 0])
    monkeypatch.setattr(adm, "_active_requests", lambda: next(active, 0))
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", "5")
    async with _client() as c:
        no = await c.post("/v1/yunshu/service/restart", json={})
        assert no.status_code == 400 and not any(a[0] == "kickstart" for a in calls)
        r = await c.post("/v1/yunshu/service/restart", json={"confirm": True})
        assert r.status_code == 202
        j = r.json()
        assert j["restarting"] and j["drain_timeout_s"] == 5.0 and j["cli"]
        await asyncio.gather(*list(adm._restart_tasks))
    kick = [a for a in calls if a[0] == "kickstart"]
    assert kick == [("kickstart", "-k", adm._target())]


@pytest.mark.asyncio
async def test_restart_drain_timeout_is_respected(monkeypatch):
    calls: list[tuple] = []

    def fake(*a):
        calls.append(a)
        return subprocess.CompletedProcess(a, 0, PRINT.format(pid=os.getpid()), "")

    monkeypatch.setattr(adm, "_launchctl", fake)
    monkeypatch.setattr(adm, "_active_requests", lambda: 3)  # never drains
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", "0")
    async with _client() as c:
        await c.post("/v1/yunshu/service/restart", json={"confirm": True})
        await asyncio.gather(*list(adm._restart_tasks))
    assert any(a[0] == "kickstart" for a in calls)  # gave up waiting, restarted anyway


# ── CORS ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cors_get_default():
    async with _client() as c:
        j = (
            await c.get("/v1/yunshu/cors", headers={"Origin": "http://localhost:3000"})
        ).json()
    assert "http://localhost:3000" in j["origins"] and j["credentials"] is True
    assert j["wildcard"] is False and j["request_origin_allowed"] is True


@pytest.mark.asyncio
async def test_cors_patch_applies_live_to_the_middleware():
    async with _client() as c:
        pre = await c.options(
            "/v1/yunshu/cors",
            headers={
                "Origin": "https://console.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert pre.status_code == 400  # not allowed yet
        r = await c.patch(
            "/v1/yunshu/cors",
            json={"origins": ["https://console.example/", "HTTP://localhost:3000"]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["origins"] == [
            "https://console.example",
            "http://localhost:3000",
        ]
        pre = await c.options(
            "/v1/yunshu/cors",
            headers={
                "Origin": "https://console.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == "https://console.example"
        assert pre.headers["access-control-allow-credentials"] == "true"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "ftp://x.example",
        "localhost:3000",
        "http://x.example/path",
        "http://x.example?q=1",
        "http://user@x.example",
        "http://",
        "not a url",
        "http://x.example:99999",
    ],
)
async def test_cors_rejects_non_origins(bad):
    async with _client() as c:
        r = await c.patch("/v1/yunshu/cors", json={"origins": [bad]})
    assert r.status_code == 422 and bad in r.json()["detail"]["invalid"]


@pytest.mark.asyncio
async def test_cors_wildcard_rules():
    async with _client() as c:
        r = await c.patch("/v1/yunshu/cors", json={"origins": ["*"]})
        assert (
            r.status_code == 400
            and r.json()["detail"]["code"] == "cors_wildcard_unconfirmed"
        )
        mixed = await c.patch(
            "/v1/yunshu/cors",
            json={"origins": ["*", "http://a.example"], "allow_any_origin": True},
        )
        assert mixed.status_code == 422
        ok = await c.patch(
            "/v1/yunshu/cors", json={"origins": ["*"], "allow_any_origin": True}
        )
        j = ok.json()
        assert ok.status_code == 200 and j["wildcard"] is True
        assert j["credentials"] is False and any("WARNING" in w for w in j["warnings"])
        pre = await c.options(
            "/v1/yunshu/cors",
            headers={
                "Origin": "https://any.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert pre.status_code == 200
        assert "access-control-allow-credentials" not in pre.headers


@pytest.mark.asyncio
async def test_cors_empty_and_reset_and_env_override(monkeypatch):
    async with _client() as c:
        assert (
            await c.patch("/v1/yunshu/cors", json={"origins": []})
        ).status_code == 422
        await c.patch("/v1/yunshu/cors", json={"origins": ["http://a.example"]})
        r = await c.patch("/v1/yunshu/cors", json={"origins": None})
        assert r.json()["origins"] == ["http://localhost:3000", "http://localhost:8000"]
        monkeypatch.setenv("YUNSHU_CORS_ORIGINS", "http://env.example")
        r = await c.patch("/v1/yunshu/cors", json={"origins": ["http://b.example"]})
        j = r.json()
        assert j["origins"] == ["http://env.example"] and j["source"] == "env"
        assert any("overrides" in w for w in j["warnings"])


@pytest.mark.asyncio
async def test_config_patch_validates_cors_origins():
    async with _client() as c:
        r = await c.patch(
            "/v1/yunshu/config",
            json={"settings": {"YUNSHU_CORS_ORIGINS": "http://x.example/p"}},
        )
        assert r.status_code == 422
        ok = await c.patch(
            "/v1/yunshu/config",
            json={
                "settings": {
                    "YUNSHU_CORS_ORIGINS": "http://x.example, http://y.example"
                }
            },
        )
        assert ok.status_code == 200
    assert settings.get("YUNSHU_CORS_ORIGINS") == "http://x.example,http://y.example"
