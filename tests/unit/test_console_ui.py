"""The optional console mount is static-only and does not start inference."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_engine import settings
from yunshu_gateway.console_ui import mount_console_ui
from yunshu_gateway.middleware.auth import AuthMiddleware


def _console_app(static_dir):
    app = FastAPI()
    mount_console_ui(app, static_dir=static_dir)
    return app


def test_console_serves_shell_and_assets_without_history_fallback(tmp_path):
    console = tmp_path / "console_static"
    assets = console / "assets"
    assets.mkdir(parents=True)
    (console / "index.html").write_text(
        '<main id="console">Console shell</main>', encoding="utf-8"
    )
    (assets / "app-a1b2.js").write_text("window.consoleReady = true;", encoding="utf-8")

    client = TestClient(_console_app(console))

    assert client.get("/console/").status_code == 200
    assert "Console shell" in client.get("/console").text
    asset = client.get("/console/assets/app-a1b2.js")
    assert asset.status_code == 200
    assert asset.text == "window.consoleReady = true;"
    assert client.get("/console/assets/missing.js").status_code == 404
    # The Vite app uses hash routing; an unknown pathname must not silently
    # return index.html and disguise a missing asset/route.
    assert client.get("/console/chat/thread-1").status_code == 404


def test_console_missing_build_returns_actionable_not_found(tmp_path):
    client = TestClient(_console_app(tmp_path / "not-built"))

    response = client.get("/console/")

    assert response.status_code == 404
    assert "cd frontend && pnpm build" in response.text
    assert not response.headers["content-type"].startswith("application/json")


def test_console_static_files_cannot_escape_build_root(tmp_path):
    console = tmp_path / "console_static"
    assets = console / "assets"
    assets.mkdir(parents=True)
    (console / "index.html").write_text("ok", encoding="utf-8")
    secret = tmp_path / "private.txt"
    secret.write_text("not public", encoding="utf-8")
    (assets / "private.txt").symlink_to(secret)

    client = TestClient(_console_app(console))

    assert client.get("/console/assets/private.txt").status_code == 404
    assert client.get("/console/assets/%2e%2e/%2e%2e/private.txt").status_code == 404


def test_console_shell_is_public_but_neighboring_and_api_paths_keep_auth(
    monkeypatch, tmp_path
):
    console = tmp_path / "console_static"
    console.mkdir()
    (console / "index.html").write_text("console", encoding="utf-8")

    monkeypatch.setattr(settings, "get_bool", lambda key: False)
    monkeypatch.setattr(
        settings,
        "get",
        lambda key: "console-test-secret" if key == "YUNSHU_AUTH_TOKEN" else None,
    )
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    mount_console_ui(app, static_dir=console)

    @app.get("/consoleX")
    async def console_neighbor():
        return {"ok": True}

    @app.get("/v1/private-test")
    async def private_api():
        return {"ok": True}

    client = TestClient(app)

    assert client.get("/console/").status_code == 200
    assert client.get("/consoleX").status_code == 401
    assert client.get("/v1/private-test").status_code == 401
    authorized = client.get(
        "/v1/private-test", headers={"Authorization": "Bearer console-test-secret"}
    )
    assert authorized.status_code == 200
