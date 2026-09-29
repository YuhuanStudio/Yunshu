"""Client-disconnect detection that survives ``BaseHTTPMiddleware``.

Starlette's ``Request.is_disconnected()`` never sees the disconnect when the
request passes through ``BaseHTTPMiddleware`` (it polls the wrapped ``receive``
under an already-cancelled scope), so a client that hangs up while its request
is being prefilled or decoded would keep the GPU busy to ``max_tokens``. This
outermost pure-ASGI layer keeps reading the server's ``receive`` in a
background task and records the disconnect in ``scope["yunshu.client_gone"]``
(an ``asyncio.Event``); the app still gets every message through its own
``receive``.
"""

from __future__ import annotations

import asyncio
import contextlib

SCOPE_KEY = "yunshu.client_gone"


class DisconnectWatchMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        gone = asyncio.Event()
        scope[SCOPE_KEY] = gone
        queue: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            while True:
                message = await receive()
                queue.put_nowait(message)
                if message["type"] == "http.disconnect":
                    gone.set()
                    return

        async def recv():
            if queue.empty() and gone.is_set():
                return {"type": "http.disconnect"}
            return await queue.get()

        task = asyncio.create_task(pump())
        try:
            await self.app(scope, recv, send)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def client_disconnected(http_request) -> bool:
    """True once the client of ``http_request`` has gone away."""
    gone = http_request.scope.get(SCOPE_KEY)
    if gone is not None and gone.is_set():
        return True
    return await http_request.is_disconnected()
