"""Failed-authentication backoff: a client that keeps failing gets 429 + Retry-After."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from yunshu_gateway.middleware import auth as auth_mod
from yunshu_gateway.middleware.auth import AuthMiddleware


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    auth_mod._failures.clear()
    yield
    auth_mod._failures.clear()


def _client(peer):
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/v1/ping")
    async def ping():
        return {"ok": True}

    async def with_peer(scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = (peer, 5555)
        await app(scope, receive, send)

    return TestClient(with_peer)


def test_repeated_failures_get_429_with_retry_after():
    c = _client("203.0.113.9")
    bad = {"Authorization": "Bearer nope"}
    for _ in range(auth_mod._FAIL_LIMIT):
        assert c.get("/v1/ping", headers=bad).status_code == 401
    r = c.get("/v1/ping", headers=bad)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    # a correct token still passes (and clears the count); only wrong credentials are throttled
    assert c.get("/v1/ping", headers={"Authorization": "Bearer tok"}).status_code == 200
    assert c.get("/v1/ping", headers=bad).status_code == 401
    # another client is unaffected
    assert (
        _client("203.0.113.10")
        .get("/v1/ping", headers={"Authorization": "Bearer tok"})
        .status_code
        == 200
    )


def test_missing_header_is_not_a_guess():
    c = _client("203.0.113.9")
    for _ in range(auth_mod._FAIL_LIMIT + 5):
        assert c.get("/v1/ping").status_code == 401
    assert not auth_mod._failures
    assert c.get("/v1/ping", headers={"Authorization": "Bearer tok"}).status_code == 200


def test_trusted_proxy_uses_forwarded_client(monkeypatch):
    monkeypatch.setenv("YUNSHU_TRUSTED_PROXIES", "203.0.113.1")
    c = _client("203.0.113.1")
    bad = {"Authorization": "Bearer nope", "X-Forwarded-For": "198.51.100.7"}
    for _ in range(auth_mod._FAIL_LIMIT):
        assert c.get("/v1/ping", headers=bad).status_code == 401
    assert c.get("/v1/ping", headers=bad).status_code == 429
    # another client behind the same proxy has its own bucket
    other = {"Authorization": "Bearer nope", "X-Forwarded-For": "198.51.100.8"}
    assert c.get("/v1/ping", headers=other).status_code == 401
    # an untrusted peer cannot pick its bucket with a forged header
    monkeypatch.setenv("YUNSHU_TRUSTED_PROXIES", "")
    forged = {"Authorization": "Bearer nope", "X-Forwarded-For": "198.51.100.9"}
    for _ in range(auth_mod._FAIL_LIMIT):
        c.get("/v1/ping", headers=forged)
    assert "198.51.100.9" not in auth_mod._failures
    assert "203.0.113.1" in auth_mod._failures


def test_failures_age_out(monkeypatch):
    c = _client("203.0.113.9")
    bad = {"Authorization": "Bearer nope"}
    for _ in range(auth_mod._FAIL_LIMIT):
        c.get("/v1/ping", headers=bad)
    assert c.get("/v1/ping", headers=bad).status_code == 429
    for q in auth_mod._failures.values():
        q.clear()
        q.append(auth_mod.time.monotonic() - auth_mod._FAIL_WINDOW_S - 1)
    assert c.get("/v1/ping", headers={"Authorization": "Bearer tok"}).status_code == 200


def test_loopback_is_never_throttled():
    c = _client("127.0.0.1")
    for _ in range(auth_mod._FAIL_LIMIT + 5):
        assert c.get("/v1/ping").status_code == 401
    assert c.get("/v1/ping", headers={"Authorization": "Bearer tok"}).status_code == 200
