"""The gateway middleware stack is pure ASGI: every BaseHTTPMiddleware layer re-pipes each streamed
chunk through a memory stream and a task hop (~15 us per chunk and layer), and with the default
stack that was seven layers. These tests pin the layers out and the behaviour that moved."""

import pytest
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from yunshu_gateway import api_keys
from yunshu_gateway.middleware.auth import AuthMiddleware


def _app(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from yunshu_gateway.main import create_app

    app = create_app()

    async def echo(request: Request):
        body = await request.body()
        return JSONResponse({"n": len(body), "head": body[:5].decode()})

    app.router.routes.insert(0, Route("/v1/_echo", echo, methods=["POST"]))
    return app


def test_default_stack_has_no_base_http_middleware(monkeypatch):
    app = _app(monkeypatch)
    layers = [m.cls for m in app.user_middleware]
    assert not [c for c in layers if issubclass(c, BaseHTTPMiddleware)], layers


def test_request_id_and_body_replay_without_content_length(monkeypatch):
    c = TestClient(_app(monkeypatch))

    def chunks():
        yield b"hello "
        yield b"world"

    r = c.post("/v1/_echo", content=chunks(), headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 200
    assert r.json() == {
        "n": 11,
        "head": "hello",
    }  # the byte-counted body reaches the route
    assert r.headers["x-request-id"] == "abc-123"
    assert "x-request-id" in c.get("/health").headers


def test_oversized_body_is_refused_with_or_without_content_length(monkeypatch):
    c = TestClient(_app(monkeypatch, YUNSHU_MAX_REQUEST_SIZE="64"))
    r = c.post("/v1/_echo", content=b"x" * 200)  # declared length
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"

    def chunks():
        for _ in range(4):
            yield b"x" * 50

    r = c.post(
        "/v1/_echo", content=chunks()
    )  # undeclared length: counted from the stream
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
    r = c.post("/v1/_echo", content=b"x", headers={"Content-Length": "-1"})
    assert r.status_code in (400, 413)


def test_streamed_response_passes_through_every_layer(monkeypatch):
    app = _app(monkeypatch)

    async def gen():
        for i in range(5):
            yield f"data: {i}\n\n".encode()

    async def stream(request: Request):
        return StreamingResponse(gen(), media_type="text/event-stream")

    app.router.routes.insert(0, Route("/v1/chat/completions", stream, methods=["POST"]))
    r = TestClient(app).post("/v1/chat/completions", content=b"{}")
    assert r.text == "".join(f"data: {i}\n\n" for i in range(5))
    assert r.headers["content-type"].startswith("text/event-stream")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def test_key_slot_is_released_after_the_response_and_after_an_error(store):
    rec, secret = store.create("a", quotas={"max_concurrent": 1})

    async def ok(request: Request):
        return JSONResponse({"ok": True})

    async def boom(request: Request):
        raise RuntimeError("boom")

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.router.routes.append(Route("/ok", ok))
    app.router.routes.append(Route("/boom", boom))
    c = TestClient(app, raise_server_exceptions=False)
    h = {"Authorization": f"Bearer {secret}"}
    for path in ("/ok", "/boom", "/ok", "/boom", "/ok"):
        assert c.get(path, headers=h).status_code in (
            200,
            500,
        )  # never 429: the slot came back
    assert store._state[rec.id].inflight == 0


def test_tavily_body_api_key_is_read_and_the_body_still_reaches_the_route(
    store, monkeypatch
):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")

    async def search(request: Request):
        return JSONResponse(await request.json())

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.router.routes.append(Route("/tavily/search", search, methods=["POST"]))
    c = TestClient(app)
    assert (
        c.post("/tavily/search", json={"api_key": "nope", "q": "x"}).status_code == 401
    )
    r = c.post("/tavily/search", json={"api_key": "tok", "q": "x"})
    assert r.status_code == 200 and r.json() == {"api_key": "tok", "q": "x"}
