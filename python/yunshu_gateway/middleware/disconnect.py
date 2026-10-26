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

        send_failed = False

        async def guarded_send(message):
            # A send that fails means the socket is gone (uvicorn may raise OSError). Treat it
            # as a client disconnect, as a closed socket would be: the app's own disconnect
            # listener then cancels the response (closing the stream, ending the engine call)
            # instead of the stream being abandoned mid-iteration. Later sends are dropped.
            nonlocal send_failed
            if send_failed:
                return
            try:
                await send(message)
            except OSError:
                send_failed = True
                if not gone.is_set():
                    gone.set()
                    queue.put_nowait({"type": "http.disconnect"})

        task = asyncio.create_task(pump())
        try:
            await self.app(scope, recv, guarded_send)
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
