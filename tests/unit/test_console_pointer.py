"""/console/ on the engine only points at the console process (deprecated for one release)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import console_ui
from yunshu_gateway.console_ui import mount_console_ui
from yunshu_gateway.middleware.auth import AuthMiddleware


def _client() -> TestClient:
    app = FastAPI()
    mount_console_ui(app)
    return TestClient(app, follow_redirects=False)


def test_redirects_to_the_console_process_when_it_answers(monkeypatch):
    monkeypatch.setattr(console_ui, "_console_up", lambda host, port: True)
    r = _client().get("/console/?x=1", headers={"host": "127.0.0.1:8000"})
    assert r.status_code == 307
    assert r.headers["location"] == "http://127.0.0.1:8100/console/?x=1"
    deep = _client().get("/console/assets/a.js", headers={"host": "mac.local:8000"})
    assert deep.headers["location"] == "http://mac.local:8100/console/assets/a.js"


def test_explains_how_to_start_it_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(console_ui, "_console_up", lambda host, port: False)
    r = _client().get("/console", headers={"host": "127.0.0.1:8000"})
    assert r.status_code == 200
    assert "yunshu console" in r.text and "yunshu service install" in r.text
    assert "http://127.0.0.1:8100/console/" in r.text


def test_the_engine_no_longer_serves_console_files(monkeypatch):
    monkeypatch.setattr(console_ui, "_console_up", lambda host, port: False)
    body = _client().get("/console/assets/index.js").text
    assert "<h1>The console has its own process</h1>" in body


def test_the_pointer_stays_public_but_the_api_keeps_auth(monkeypatch):
    from yunshu_engine import settings

    monkeypatch.setattr(console_ui, "_console_up", lambda host, port: False)
    monkeypatch.setattr(settings, "get_bool", lambda key: False)
    monkeypatch.setattr(
        settings, "get", lambda key: "t" if key == "YUNSHU_AUTH_TOKEN" else None
    )
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    mount_console_ui(app)

    @app.get("/v1/private-test")
    async def private():
        return {"ok": True}

    c = TestClient(app, follow_redirects=False)
    assert c.get("/console/").status_code == 200
    assert c.get("/v1/private-test").status_code == 401
