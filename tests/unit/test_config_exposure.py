"""What GET /v1/yunshu/config may show: credential settings are masked, URL user info
is masked, and with auth configured the route needs the admin scope."""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_engine import settings
from yunshu_gateway import api_keys
from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers import yunshu as yunshu_router

_CRED_NAME = re.compile(r"_(KEY|TOKEN|SECRET|PASSWORD)$")
_CRED_DOC = re.compile(r"\b(credentials?|password|secret|bearer (token|key))\b", re.I)
# The description mentions credentials only to say none are stored/shown.
_DOC_NOT_A_CREDENTIAL = {
    "YUNSHU_WEB_SEARCH_HEALTH_FILE",
    "YUNSHU_LOG_KEEP",
    "YUNSHU_DOCS_ASK_MAX_QUESTION",
}


def test_every_credential_setting_is_secret():
    leaks = [
        n
        for n, s in settings.REGISTRY.items()
        if (_CRED_NAME.search(n) or _CRED_DOC.search(s.description))
        and n not in _DOC_NOT_A_CREDENTIAL
        and not s.secret
    ]
    assert not leaks, f"mark secret=True in settings.py: {leaks}"


def test_url_user_info_is_masked(monkeypatch):
    monkeypatch.setenv("YUNSHU_SEARXNG_URL", "http://bob:hunter2@searx.local:8080/x")
    row = next(
        r for r in settings.effective(("stable",)) if r["name"] == "YUNSHU_SEARXNG_URL"
    )
    assert "hunter2" not in str(row["value"]) and "bob" not in str(row["value"])
    assert row["value"] == "http://***@searx.local:8080/x"
    assert (
        settings.mask_url_userinfo("http://localhost:8000") == "http://localhost:8000"
    )


@pytest.fixture
def keystore(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(yunshu_router.router, prefix="/v1")
    return TestClient(app)


def H(secret):
    return {"Authorization": f"Bearer {secret}"}


def test_config_read_needs_admin_when_auth_is_on(keystore):
    _, infer = keystore.create("a", scopes=["infer"])
    _, adm = keystore.create("b", scopes=["infer", "admin"])
    c = _client()
    assert c.get("/v1/yunshu/config", headers=H(infer)).status_code == 403
    assert c.get("/v1/yunshu/config", headers=H(adm)).status_code == 200
    assert c.get("/v1/yunshu/config", headers=H("tok")).status_code == 200


def test_config_read_stays_open_without_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    api_keys.configure(tmp_path / "keys.json")
    try:
        assert _client().get("/v1/yunshu/config").status_code == 200
    finally:
        api_keys._store = None
