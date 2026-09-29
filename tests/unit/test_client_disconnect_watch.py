"""A client that hangs up mid-request is noticed even behind BaseHTTPMiddleware."""

import asyncio

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from yunshu_gateway.middleware.disconnect import (
    DisconnectWatchMiddleware,
    client_disconnected,
)


class _Passthrough(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        return await call_next(request)


def _app(seen: list):
    async def endpoint(request: Request):
        for _ in range(100):
            if await client_disconnected(request):
                seen.append("gone")
                break
            await asyncio.sleep(0.02)
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/", endpoint, methods=["POST"])])
    app.add_middleware(_Passthrough)
    app.add_middleware(_Passthrough)
    app.add_middleware(DisconnectWatchMiddleware)
    return app


def _scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
        "scheme": "http",
    }


async def _run(disconnect_after: float | None):
    seen: list = []
    inbox: asyncio.Queue = asyncio.Queue()
    inbox.put_nowait({"type": "http.request", "body": b"{}", "more_body": False})
    out: list = []

    async def receive():
        return await inbox.get()

    async def send(message):
        out.append(message)

    if disconnect_after is not None:
        asyncio.get_running_loop().call_later(
            disconnect_after, inbox.put_nowait, {"type": "http.disconnect"}
        )
    await asyncio.wait_for(_app(seen)(_scope(), receive, send), 10)
    return seen


def test_disconnect_is_seen_behind_base_http_middleware():
    assert asyncio.run(_run(0.1)) == ["gone"]


def test_connected_client_is_not_reported_gone():
    async def go():
        seen: list = []
        inbox: asyncio.Queue = asyncio.Queue()
        inbox.put_nowait({"type": "http.request", "body": b"{}", "more_body": False})
        sent: list = []

        async def receive():
            return await inbox.get()

        async def send(message):
            sent.append(message)

        # Shorten the endpoint's wait by disconnecting only after it finishes.
        await asyncio.wait_for(_app(seen)(_scope(), receive, send), 10)
        return seen

    assert asyncio.run(go()) == []
