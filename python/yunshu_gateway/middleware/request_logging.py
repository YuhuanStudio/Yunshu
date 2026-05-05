"""Yunshu Gateway — Request logging middleware.

Structured request/response logging with:
- Request ID tracking (X-Request-ID header)
- Latency measurement
- Active request counting for graceful shutdown drain
- Configurable log levels per status code
"""
from __future__ import annotations

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("yunshu.gateway")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with structured fields."""

    SKIP_PATHS = {"/health", "/health/ready", "/health/live", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        import yunshu_gateway.main as _main

        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        _main._active_requests += 1

        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"ERROR {elapsed*1000:.1f}ms — {e}"
            )
            raise
        finally:
            _main._active_requests -= 1
            if _main._active_requests == 0 and _main._drain_event is not None:
                _main._drain_event.set()

        elapsed = time.monotonic() - t0
        response.headers["X-Request-ID"] = request_id

        if request.url.path not in self.SKIP_PATHS:
            level = logging.DEBUG
            if response.status_code >= 500:
                level = logging.ERROR
            elif response.status_code >= 400:
                level = logging.WARNING

            logger.log(
                level,
                f"[{request_id}] {request.method} {request.url.path} "
                f"{response.status_code} {elapsed*1000:.1f}ms",
            )

        return response
