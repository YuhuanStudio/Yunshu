"""Per-key ownership: an infer key is neither admin nor the owner of another key's objects."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from yunshu_control.audit_log import resolve_actor
from yunshu_gateway import api_keys
from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers.responses import _is_admin, _owns_stored


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def _app():
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/who")
    async def who(request: Request):
        return {
            "actor": resolve_actor(request),
            "admin": _is_admin(request),
            "owns_legacy": _owns_stored(request, {"_owner": "owner"}),
            "owns_a": _owns_stored(request, {"_owner": "key:A"}),
        }

    return app


def H(secret):
    return {"Authorization": f"Bearer {secret}"}


def test_keys_get_their_own_actor_and_only_admin_scope_is_admin(store, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    _, a = store.create("a", scopes=["infer"])
    rec_b, b = store.create("b", scopes=["infer"])
    _, adm = store.create("adm", scopes=["infer", "admin"])
    c = TestClient(_app())

    rb = c.get("/who", headers=H(b)).json()
    assert rb["actor"] == f"key:{rec_b.id}"
    assert rb["admin"] is False
    assert rb["owns_legacy"] is False  # pre-existing owner="owner" objects
    assert rb["owns_a"] is False

    assert c.get("/who", headers=H(a)).json()["actor"] != rb["actor"]
    assert c.get("/who", headers=H(adm)).json()["admin"] is True

    tok = c.get("/who", headers=H("tok")).json()  # static token stays owner/admin
    assert (tok["actor"], tok["admin"], tok["owns_legacy"]) == ("owner", True, True)


def test_auth_disabled_single_user_is_owner(store, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    r = TestClient(_app()).get("/who").json()
    assert (r["actor"], r["admin"], r["owns_legacy"]) == ("owner", True, True)
