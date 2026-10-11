"""Multiple API keys: back-compat with the single token, scopes, quotas, rotation, hashing,
usage accounting, file permissions, and the admin API."""

import json
import os
import stat
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import api_keys
from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers import admin_keys
from yunshu_gateway.routers.models import _check_permission


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def make_app():
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(admin_keys.router, prefix="/v1")

    @app.post("/v1/chat/completions")
    async def chat(request: __import__("fastapi").Request):
        _check_permission(request, "can_infer")
        return {"ok": True}

    @app.post("/v1/messages")
    async def msgs(request: __import__("fastapi").Request):
        _check_permission(request, "can_infer")
        return {"ok": True}

    @app.post("/v1/models/load")
    async def load(request: __import__("fastapi").Request):
        _check_permission(request, "can_load_models")
        return {"ok": True}

    return app


def H(secret):
    return {"Authorization": f"Bearer {secret}"}


def test_no_keys_no_token_unchanged(store):
    c = TestClient(make_app())
    assert c.post("/v1/chat/completions").status_code == 200  # inference open
    assert c.post("/v1/models/load").status_code == 401  # privileged denied
    assert c.get("/v1/yunshu/keys").status_code == 401


def test_old_token_still_admin(store, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    c = TestClient(make_app())
    assert c.post("/v1/chat/completions").status_code == 401
    assert c.post("/v1/chat/completions", headers=H("tok")).status_code == 200
    assert c.post("/v1/models/load", headers=H("tok")).status_code == 200
    r = c.post("/v1/yunshu/keys", json={"name": "a"}, headers=H("tok"))
    assert r.status_code == 201
    # token keeps working once keys exist
    assert c.post("/v1/models/load", headers=H("tok")).status_code == 200


def test_keys_alone_enable_auth_and_scopes(store):
    infer, sec_i = store.create("i", ["infer"])
    admin, sec_a = store.create("a", ["admin"])
    c = TestClient(make_app())
    assert c.post("/v1/chat/completions").status_code == 401
    assert c.post("/v1/chat/completions", headers=H("nope")).status_code == 401
    assert c.post("/v1/chat/completions", headers=H(sec_i)).status_code == 200
    assert c.post("/v1/messages", headers={"x-api-key": sec_i}).status_code == 200
    assert c.post("/v1/models/load", headers=H(sec_i)).status_code == 403
    assert c.get("/v1/yunshu/keys", headers=H(sec_i)).status_code == 403
    assert c.post("/v1/models/load", headers=H(sec_a)).status_code == 200
    assert c.get("/v1/yunshu/keys", headers=H(sec_a)).status_code == 200


def test_disabled_and_expired(store):
    _, sec = store.create("k")
    kid = store.list_keys()[0]["id"]
    c = TestClient(make_app())
    store.update(kid, {"enabled": False})
    assert c.post("/v1/chat/completions", headers=H(sec)).status_code == 401
    store.update(kid, {"enabled": True, "expires": time.time() - 1})
    r = c.post("/v1/chat/completions", headers=H(sec))
    assert r.status_code == 401 and "expired" in r.json()["error"]["message"]
    store.update(kid, {"expires": time.time() + 100})
    assert c.post("/v1/chat/completions", headers=H(sec)).status_code == 200


def test_request_quota_429_with_retry_after_and_shapes(store):
    _, sec = store.create("q", quotas={"requests_per_day": 2})
    c = TestClient(make_app())
    for _ in range(2):
        assert c.post("/v1/chat/completions", headers=H(sec)).status_code == 200
    r = c.post("/v1/chat/completions", headers=H(sec))
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert r.json()["error"]["code"] == "requests_per_day_exceeded"
    r = c.post("/v1/messages", headers=H(sec))
    assert r.status_code == 429
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert "Retry-After" in r.headers


def test_token_quota_after_accounting(store):
    rec, sec = store.create("t", quotas={"tokens_per_day": 100})
    c = TestClient(make_app())
    assert c.post("/v1/chat/completions", headers=H(sec)).status_code == 200
    store.account(rec.id, 80, 30, 10)
    r = c.post("/v1/chat/completions", headers=H(sec))
    assert (
        r.status_code == 429 and r.json()["error"]["code"] == "tokens_per_day_exceeded"
    )


def test_concurrency_quota_and_release(store):
    rec, _ = store.create("c", quotas={"max_concurrent": 1})
    p = api_keys.Principal("key", frozenset({"infer"}), rec.id)
    store.admit(p)
    with pytest.raises(api_keys.QuotaExceededError) as ei:
        store.admit(p)
    assert ei.value.which == "max_concurrent" and ei.value.retry_after >= 1
    store.release(rec.id)
    store.admit(p)


def test_rolling_window_expires(store):
    rec, _ = store.create("w", quotas={"requests_per_day": 1})
    p = api_keys.Principal("key", frozenset({"infer"}), rec.id)
    t0 = 1_000_000.0
    store.admit(p, now=t0)
    store.release(rec.id)
    with pytest.raises(api_keys.QuotaExceededError) as ei:
        store.admit(p, now=t0 + 3600)
    assert 20 * 3600 < ei.value.retry_after <= 23 * 3600 + 301
    store.admit(p, now=t0 + 86400 + 600)  # window rolled past it


def test_rotation_invalidates_old_secret(store):
    rec, old = store.create("r")
    c = TestClient(make_app())
    admin, sec_a = store.create("a", ["admin"])
    r = c.post(f"/v1/yunshu/keys/{rec.id}/rotate", headers=H(sec_a))
    new = r.json()["secret"]
    assert new != old and "hash" not in r.json()
    assert c.post("/v1/chat/completions", headers=H(old)).status_code == 401
    assert c.post("/v1/chat/completions", headers=H(new)).status_code == 200


def test_no_plaintext_on_disk_and_permissions(store, tmp_path):
    rec, sec = store.create("h")
    store.account(rec.id, 1, 1)
    store.flush(force=True)
    for p in (store.path, store.usage_path):
        assert sec not in p.read_text()
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    data = json.loads(store.path.read_text())
    assert data["keys"][0]["hash"] == api_keys.hash_secret(sec)
    assert data["keys"][0]["prefix"] == sec[:8]
    # list / get never return the hash or secret
    assert "hash" not in store.list_keys()[0] and "secret" not in store.list_keys()[0]


def test_usage_counts_via_record_done_without_double_count(store):
    from yunshu_gateway import x_yunshu

    rec, _ = store.create("u")
    info = x_yunshu.RequestInfo("r1", "POST", "/v1/chat/completions")
    info.api_key_id = rec.id
    x_yunshu.record_done(
        info, {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 4}
    )
    store.account(rec.id, error=True)
    (row,) = store.usage(rec.id)
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]) == (
        10,
        5,
        4,
    )
    assert row["errors"] == 1


def test_usage_endpoint_and_requests_counted(store):
    admin, sec_a = store.create("a", ["admin"])
    other, sec_o = store.create("o")
    c = TestClient(make_app())
    for _ in range(3):
        c.post("/v1/chat/completions", headers=H(sec_o))
    r = c.get(f"/v1/yunshu/usage?key={other.id}&group=day", headers=H(sec_a)).json()
    assert r["data"][0]["requests"] == 3
    r = c.get("/v1/yunshu/usage?since=7d&group=key", headers=H(sec_a)).json()
    assert {d["key"] for d in r["data"]} == {admin.id, other.id}
    assert c.get("/v1/yunshu/usage?since=bad", headers=H(sec_a)).status_code == 400


def test_admin_crud_validation_and_persist(store, tmp_path):
    admin, sec_a = store.create("a", ["admin"])
    c = TestClient(make_app())
    r = c.post(
        "/v1/yunshu/keys", json={"name": "x", "scopes": ["root"]}, headers=H(sec_a)
    )
    assert r.status_code == 400
    r = c.post(
        "/v1/yunshu/keys",
        json={"name": "x", "quotas": {"tokens_per_day": 5}},
        headers=H(sec_a),
    )
    kid = r.json()["id"]
    assert r.json()["secret"].startswith("ysk-")
    r = c.patch(
        f"/v1/yunshu/keys/{kid}",
        json={"enabled": False, "quotas": {"requests_per_day": 9}},
        headers=H(sec_a),
    )
    assert r.json()["enabled"] is False
    assert (
        r.json()["quotas"]["tokens_per_day"] == 5
        and r.json()["quotas"]["requests_per_day"] == 9
    )
    assert (
        c.patch(
            f"/v1/yunshu/keys/{kid}", json={"bogus": 1}, headers=H(sec_a)
        ).status_code
        == 400
    )
    # a fresh store on the same file sees them
    assert {k["id"] for k in api_keys.KeyStore(store.path).list_keys()} == {
        admin.id,
        kid,
    }
    assert (
        c.delete(f"/v1/yunshu/keys/{kid}", headers=H(sec_a)).json()["deleted"] is True
    )
    assert c.delete(f"/v1/yunshu/keys/{kid}", headers=H(sec_a)).status_code == 404


def test_usage_survives_restart(store):
    rec, _ = store.create("p", quotas={"requests_per_day": 5})
    store.admit(api_keys.Principal("key", frozenset({"infer"}), rec.id))
    store.flush(force=True)
    again = api_keys.KeyStore(store.path)
    assert again.list_keys()[0]["window"]["requests"] == 1
    assert again.list_keys()[0]["last_used"] is not None


def test_overhead_microbenchmark(store, capsys):
    rec, sec = store.create(
        "b", quotas={"requests_per_day": 10**9, "max_concurrent": 100}
    )
    p = store.authenticate(sec)
    n = 20000
    t = time.perf_counter()
    for _ in range(n):
        store.authenticate(sec)
        store.admit(p)
        store.release(rec.id)
    us = (time.perf_counter() - t) / n * 1e6
    print(f"api key auth+admit+release: {us:.1f} us/request")
    assert us < 200
