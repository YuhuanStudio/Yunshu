"""Minimal Python client for Yunshu's text WebSocket (``/v1/stream``).

    async with YunshuStream("ws://127.0.0.1:8000/v1/stream") as conn:
        async for ev in conn.chat({"model": "m", "messages": [...]}):
            print(ev["data"]["choices"][0]["delta"].get("content", ""), end="")

One connection carries many concurrent requests; iterate several
``conn.request(...)`` streams with ``asyncio.gather``. See
docs/guides/TRANSPORTS.md for the protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

__all__ = ["StreamError", "YunshuStream"]


class StreamError(RuntimeError):
    def __init__(self, status: int, error: dict):
        super().__init__(f"{status}: {error.get('message', error)}")
        self.status = status
        self.error = error


class YunshuStream:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:8000/v1/stream",
        *,
        token: str | None = None,
        uds: str | None = None,
    ) -> None:
        self.url = url
        self.token = token
        self.uds = uds
        self.session: dict = {}
        self._ws: Any = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._reader: asyncio.Task | None = None
        self._ids = itertools.count(1)

    async def __aenter__(self) -> YunshuStream:
        kw: dict[str, Any] = {"max_size": None}
        if self.token:
            kw["additional_headers"] = {"Authorization": f"Bearer {self.token}"}
        if self.uds:
            from websockets.asyncio.client import unix_connect

            self._ws = await unix_connect(self.uds, self.url, **kw)
        else:
            from websockets.asyncio.client import connect

            self._ws = await connect(self.url, **kw)
        self.session = json.loads(await self._ws.recv())
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        if self._ws:
            await self._ws.close()

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await self._ws.send(json.dumps({"type": "pong", "t": msg.get("t")}))
                    continue
                q = self._queues.get(str(msg.get("id")))
                if q is not None:
                    q.put_nowait(msg)
        finally:
            for q in self._queues.values():
                q.put_nowait({"type": "closed"})

    async def _send(self, msg: dict) -> None:
        await self._ws.send(json.dumps(msg))

    async def request(
        self,
        api: str,
        body: dict,
        *,
        id: str | None = None,
        stream_id: str | None = None,
        with_done: bool = False,
    ) -> AsyncIterator[dict]:
        """Yield ``{"type": "event", "event", "data"}`` messages as they stream.

        Raises :class:`StreamError` on an error message. With ``with_done`` the
        final ``done`` message (reason + stats) is yielded too.
        """
        rid = id or f"c{next(self._ids)}"
        q: asyncio.Queue = asyncio.Queue()
        self._queues[rid] = q
        msg = {"type": "request", "id": rid, "api": api, "body": body}
        if stream_id:
            msg["stream_id"] = stream_id
        try:
            await self._send(msg)
            while True:
                m = await q.get()
                kind = m["type"]
                if kind == "event":
                    yield m
                elif kind == "error":
                    raise StreamError(m.get("status", 500), m.get("error", {}))
                elif kind == "done":
                    if with_done:
                        yield m
                    return
                elif kind == "closed":
                    raise ConnectionError("WebSocket closed")
                else:
                    yield m  # updated, progress, ... passed through
        finally:
            self._queues.pop(rid, None)

    def chat(self, body: dict, **kw) -> AsyncIterator[dict]:
        return self.request("chat.completions", body, **kw)

    def responses(self, body: dict, **kw) -> AsyncIterator[dict]:
        return self.request("responses", body, **kw)

    def messages(self, body: dict, **kw) -> AsyncIterator[dict]:
        return self.request("messages", body, **kw)

    async def cancel(self, id: str) -> None:
        await self._send({"type": "cancel", "id": id})

    async def stop(self, id: str) -> None:
        await self._send({"type": "stop", "id": id})

    async def set_max_tokens(self, id: str, max_tokens: int) -> None:
        await self._send({"type": "update", "id": id, "max_tokens": max_tokens})
