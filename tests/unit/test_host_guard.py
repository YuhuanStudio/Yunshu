"""Host / Origin validation: DNS rebinding and cross-site simple POSTs against a loopback server."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from yunshu_gateway.middleware.host_guard import HostGuardMiddleware


def _client(server=("127.0.0.1", 8000)):
    app = FastAPI()
    app.add_middleware(HostGuardMiddleware)

    @app.get("/x")
    async def get_x():
        return {"ok": True}

    @app.post("/x")
    async def post_x():
        return {"ok": True}

    # TestClient reports server=("testserver", 80); a real uvicorn socket reports its IP.

    async def with_server(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope["server"] = server
        await app(scope, receive, send)

    return TestClient(with_server, base_url="http://127.0.0.1:8000")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("YUNSHU_HOST", "YUNSHU_ALLOWED_HOSTS", "YUNSHU_CORS_ORIGINS"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize(
    "host", ["localhost:8000", "127.0.0.1", "[::1]:8000", "LOCALHOST", "127.0.0.2:9"]
)
def test_loopback_host_allowed(host):
    assert _client().get("/x", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["attacker.example:8000", "192.168.1.5", "localhost.evil.com"]
)
def test_rebinding_host_refused(host):
    r = _client().get("/x", headers={"Host": host})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden_origin"


def test_not_checked_when_bound_to_all_interfaces(monkeypatch):
    monkeypatch.setenv("YUNSHU_HOST", "0.0.0.0")
    assert _client().get("/x", headers={"Host": "mac.local"}).status_code == 200


def test_not_checked_for_in_process_clients():
    # TestClient / Unix sockets: the ASGI server address is not a loopback IP
    assert (
        _client(server=None).get("/x", headers={"Host": "testserver"}).status_code
        == 200
    )


def test_allowed_hosts_setting(monkeypatch):
    monkeypatch.setenv("YUNSHU_ALLOWED_HOSTS", "mac.local, other.lan")
    c = _client()
    assert c.get("/x", headers={"Host": "mac.local:8000"}).status_code == 200
    assert c.get("/x", headers={"Host": "localhost"}).status_code == 200
    assert c.get("/x", headers={"Host": "evil.example"}).status_code == 403
    monkeypatch.setenv(
        "YUNSHU_HOST", "0.0.0.0"
    )  # an explicit list checks any bind address
    assert _client().get("/x", headers={"Host": "evil.example"}).status_code == 403


def test_cross_site_post_refused_but_reads_and_same_origin_pass(monkeypatch):
    c = _client()
    evil = {"Origin": "https://attacker.example", "Content-Type": "text/plain"}
    assert c.post("/x", headers=evil).status_code == 403
    assert (
        c.get("/x", headers=evil).status_code == 200
    )  # reads: CORS decides what is readable
    assert c.post("/x", headers={"Origin": "http://localhost:5173"}).status_code == 200
    assert c.post("/x", headers={"Origin": "null"}).status_code == 403
    assert (
        c.post("/x", headers={**evil, "Authorization": "Bearer t"}).status_code == 200
    )  # a page cannot add credentials without a preflight
    monkeypatch.setenv("YUNSHU_CORS_ORIGINS", "https://app.example")
    assert c.post("/x", headers={"Origin": "https://app.example"}).status_code == 200
    assert c.post("/x", headers=evil).status_code == 403


def test_same_origin_on_a_named_host(monkeypatch):
    monkeypatch.setenv("YUNSHU_ALLOWED_HOSTS", "mac.local")
    r = _client().post(
        "/x", headers={"Host": "mac.local:8000", "Origin": "http://mac.local:8000"}
    )
    assert r.status_code == 200


def test_observe_loopback_needs_a_loopback_host_header():
    from starlette.requests import Request

    from yunshu_gateway.routers.monitoring import _is_loopback_request

    def req(host):
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/debug/engine",
                "headers": [(b"host", host.encode())] if host else [],
                "client": ("127.0.0.1", 5000),
            }
        )

    assert _is_loopback_request(req("localhost:8000"))
    assert _is_loopback_request(req(""))
    assert not _is_loopback_request(req("attacker.example:8000"))  # rebinding page
