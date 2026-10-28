"""``GET /v1/yunshu/events?topics=status,host,...``: one server-sent-event stream per console tab.

Replaces the console's timers: each topic is the payload of the polled endpoint of the same
name, sent when it changes (see ``event_hub``). The polled endpoints stay, as the fallback.
Auth is the polled endpoints' own: ``can_infer`` for every topic, through the same middleware.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ..event_hub import Hub, Topic
from ..x_yunshu import BOOT_ID, registry
from .models import _check_permission

router = APIRouter(tags=["yunshu"])

_BUSY_LINGER_S = 2.0
_busy_until = 0.0


def _status_interval(payload: Any) -> float:
    """4 Hz while a request runs (and for a moment after), 1 Hz otherwise."""
    global _busy_until
    now = time.monotonic()
    try:
        if payload and payload["requests"]["active"] > 0:
            _busy_until = now + _BUSY_LINGER_S
    except (KeyError, TypeError):
        pass
    return 0.25 if now < _busy_until else 1.0


async def _status(request: Any) -> Any:
    from .yunshu import status

    return await status(request)


async def _host(request: Any) -> Any:
    from .yunshu import host

    return await host(request)


async def _memory(request: Any) -> Any:
    from .yunshu import memory_ledger

    return await memory_ledger(request)


async def _requests(request: Any) -> Any:
    from .yunshu import recent_requests

    return await recent_requests(
        request, limit=512, model=None, since=None, after_seq=None
    )


async def _downloads(request: Any) -> Any:
    from .admin_models import list_downloads

    return await list_downloads(request)


def _downloads_interval(payload: Any) -> float:
    try:
        return 1.5 if payload and payload.get("active") else 4.0
    except AttributeError:
        return 4.0


def build_hub() -> Hub:
    hub = Hub()
    hub.register(
        Topic("status", _status, _status_interval, volatile=frozenset({"uptime_s"}))
    )
    hub.register(Topic("host", _host, lambda _p: 2.0, volatile=frozenset()))
    hub.register(Topic("memory", _memory, lambda _p: 2.0))
    hub.register(Topic("requests", _requests, lambda _p: 0.5))
    hub.register(Topic("downloads", _downloads, _downloads_interval))
    return hub


_hub: Hub | None = None


def get_hub() -> Hub:
    global _hub
    if _hub is None:
        _hub = build_hub()
        hub = _hub
        # A request arriving, leaving or being recorded is reported at once, not at the next tick.
        registry.add_listener(lambda: hub.poke("status", "requests"))
        # A model starting or finishing a load (or unloading) moves the status and the memory ledger.
        from yunshu_engine import model_manager

        model_manager.state_listeners.append(lambda: hub.poke("status", "memory"))
    return _hub


@router.get("/yunshu/events")
async def events(
    request: Request,
    topics: str = Query("status", max_length=200),
) -> StreamingResponse:
    _check_permission(request, "can_infer")
    hub = get_hub()
    wanted = [t.strip() for t in topics.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in hub.topics]
    if not wanted or unknown:
        raise HTTPException(400, f"unknown topics {unknown or topics!r}")
    return StreamingResponse(
        hub.subscribe(
            wanted, request, boot_id=BOOT_ID, is_disconnected=request.is_disconnected
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
