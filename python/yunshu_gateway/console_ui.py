"""``/console/`` on the engine: a pointer to the console process (deprecated for one release).

The web console is its own light process now (``yunshu console``, started next to the engine by
``yunshu serve`` unless ``--no-console``, or run as its own service job). It serves the console and
docs, proxies the engine API and keeps the history, so it stays usable while the engine is down.

Existing bookmarks, docs and scripts that open ``http://<engine>/console/`` keep working for this
release: when the console process answers, the engine redirects there; when it does not, it shows a
short page saying how to start it. The engine no longer serves the console files itself. This
shim is removed in the release after next (see CHANGELOG).
"""

from __future__ import annotations

import html
import socket
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse, RedirectResponse, Response


def _console_port() -> int:
    from yunshu_engine import settings

    return int(settings.get("YUNSHU_CONSOLE_PORT") or 8100)


def _console_up(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def pointer(request: Request, rest: str = "") -> Response:
    """Redirect to the console process when it is up, else explain how to start it."""
    hostname = request.url.hostname or "127.0.0.1"
    port = _console_port()
    probe = "127.0.0.1" if hostname in ("localhost", "0.0.0.0", "::1") else hostname
    target = f"{request.url.scheme}://{hostname if ':' not in hostname else '[' + hostname + ']'}:{port}/console/{rest}"
    if request.url.query:
        target += "?" + request.url.query
    if _console_up(probe, port):
        return RedirectResponse(target, status_code=307)
    page = f"""<!doctype html>
<meta charset="utf-8"><title>Yunshu console</title>
<body style="font:16px system-ui;max-width:42rem;margin:4rem auto;padding:0 1rem;line-height:1.6">
<h1>The console has its own process</h1>
<p>The web console no longer runs inside the engine. It is started next to it by
<code>yunshu serve</code> (unless <code>--no-console</code> was given), at
<a href="{html.escape(target)}">{html.escape(target)}</a>, and nothing answers there right now.</p>
<p>Start it with <code>yunshu console --engine http://{html.escape(hostname)}:{request.url.port or 8000}</code>,
or install it as a service job with <code>yunshu service install</code>.</p>
<p style="color:#666">This pointer page is deprecated and goes away in a later release.</p>
"""
    return HTMLResponse(page, status_code=200)


def mount_console_ui(app: FastAPI, *, static_dir: Path | None = None) -> None:
    """Register the pointer routes. ``static_dir`` is accepted for compatibility and unused."""

    @app.get("/console", include_in_schema=False)
    @app.get("/console/", include_in_schema=False)
    async def console_root(request: Request) -> Response:
        return pointer(request)

    @app.get("/console/{rest:path}", include_in_schema=False)
    async def console_rest(request: Request, rest: str) -> Response:
        return pointer(request, rest)
