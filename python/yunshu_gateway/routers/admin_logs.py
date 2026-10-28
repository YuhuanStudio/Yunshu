"""Server logs for the console.

- ``GET /v1/yunshu/logs?level=&since=&since_id=&q=&limit=``  recent records from an
  in-memory ring (2,000 records, redacted when emitted)
- ``GET /v1/yunshu/logs/stream``  the same as server-sent events (live tail)

Logs can hold paths and error text, so they need the ``admin`` permission.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .. import log_ring
from .models import _check_permission

router = APIRouter(tags=["yunshu"])

_STREAM_POLL_S = 1.0
_WAKE_COALESCE_S = 0.05
_STREAM_KEEPALIVE_S = 15.0


def _ring() -> log_ring.RingHandler:
    return log_ring.handler() or log_ring.install()


@router.get("/yunshu/logs")
async def logs(
    request: Request,
    level: str | None = Query(None),
    since: float | None = Query(None, description="unix seconds"),
    since_id: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=200),
    limit: int = Query(500, ge=1, le=2000),
) -> dict[str, Any]:
    _check_permission(request, "admin")
    try:
        return _ring().query(
            level=level, since=since, since_id=since_id, q=q, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/yunshu/logs/stream")
async def logs_stream(
    request: Request,
    level: str | None = Query(None),
    q: str | None = Query(None, max_length=200),
    since_id: int | None = Query(None, ge=0),
) -> StreamingResponse:
    _check_permission(request, "admin")
    ring = _ring()
    try:
        log_ring.level_no(level)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Start at the tail unless the client resumes from a cursor (Last-Event-ID style).
    cursor = since_id if since_id is not None else ring.query(limit=1)["next_id"]

    async def events():
        nonlocal cursor
        last_send = time.monotonic()
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()

        def poke() -> None:  # called from any logging thread
            loop.call_soon_threadsafe(wake.set)

        ring.add_listener(poke)
        try:
            async for chunk in _tail(request, ring, level, q, wake, last_send):
                yield chunk
        finally:
            ring.remove_listener(poke)

    async def _tail(request, ring, level, q, wake, last_send):
        nonlocal cursor
        while not await request.is_disconnected():
            res = ring.query(level=level, since_id=cursor, q=q, limit=500)
            cursor = res["next_id"]
            for rec in res["records"]:
                yield f"id: {rec['id']}\ndata: {json.dumps(rec)}\n\n"
                last_send = time.monotonic()
            if time.monotonic() - last_send > _STREAM_KEEPALIVE_S:
                yield ": keepalive\n\n"
                last_send = time.monotonic()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), timeout=_STREAM_POLL_S)
            wake.clear()
            await asyncio.sleep(
                _WAKE_COALESCE_S
            )  # a burst of records goes out as one batch

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
