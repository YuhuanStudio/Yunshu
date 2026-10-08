"""Local pre-admission handshake with gpuq; disabled unless explicitly configured.

An acknowledgement is a fresh, request-specific ticket, never a PID to signal.
The scheduler owns all stopping. Activity covers the complete ASGI stream.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from starlette.responses import JSONResponse

from yunshu_engine import settings

from .x_yunshu import TRACKED_PATHS


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value))
    tmp.chmod(0o600)
    tmp.replace(path)


class ResidentActivityMiddleware:
    def __init__(self, app):
        self.app = app
        self.active = 0
        self.ticket = ""

    async def __call__(self, scope, receive, send):
        configured = settings.get("YUNSHU_RESEARCH_ACTIVITY_FILE")
        if (
            not configured
            or scope["type"] != "http"
            or scope["path"] not in TRACKED_PATHS
        ):
            return await self.app(scope, receive, send)
        path = Path(configured).expanduser()
        timeout = settings.get_float("YUNSHU_RESEARCH_PAUSE_TIMEOUT_S") or 2.0
        self.active += 1
        # Each arrival requires a fresh acknowledgement. Concurrent waiters follow
        # the newest ticket; an old acknowledgement can never admit a new request.
        self.ticket = uuid.uuid4().hex

        def publish():
            atomic_json(
                path,
                {
                    "schema": 1,
                    "active": self.active,
                    "ticket": self.ticket,
                    "updated": time.time(),
                },
            )

        try:
            publish()
            deadline = time.monotonic() + timeout
            while True:
                try:
                    ack = json.loads(Path(str(path) + ".ack").read_text())
                    if ack.get("ticket") == self.ticket and ack.get("safe") is True:
                        break
                except (OSError, ValueError, TypeError):
                    pass
                if time.monotonic() >= deadline:
                    response = JSONResponse(
                        {
                            "error": {
                                "message": "Research pause acknowledgement timed out",
                                "type": "server_error",
                                "code": "research_pause_timeout",
                            }
                        },
                        status_code=503,
                        headers={"Retry-After": "1"},
                    )
                    return await response(scope, receive, send)
                await asyncio.sleep(0.01)
            await self.app(scope, receive, send)
        finally:
            self.active -= 1
            publish()
