"""The console process serves the built console as a same-origin static app (no inference, no mlx)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_console.app import ConsoleStaticFiles


def _console_app(static_dir):
    app = FastAPI()
    app.mount("/console", ConsoleStaticFiles(static_dir), name="console")
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


def test_the_shell_is_revalidated_and_hashed_assets_are_not(tmp_path):
    console = tmp_path / "console_static"
    (console / "assets").mkdir(parents=True)
    (console / "index.html").write_text("ok", encoding="utf-8")
    (console / "assets" / "app-a1b2.js").write_text("1", encoding="utf-8")
    client = TestClient(_console_app(console))
    assert client.get("/console/").headers["cache-control"] == "no-cache"
    assert "no-cache" not in client.get("/console/assets/app-a1b2.js").headers.get(
        "cache-control", ""
    )
