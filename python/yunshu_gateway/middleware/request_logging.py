"""Yunshu Gateway — Request logging middleware.

Structured request/response logging with:
- Request ID tracking (X-Request-ID header)
- Latency measurement with slow request warnings
- Configurable log levels per status code

Note: Active request counting for graceful shutdown is handled by
track_active_requests middleware in main.py, not here.
"""

import logging
import threading
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders

from yunshu_engine import settings

logger = logging.getLogger("yunshu.gateway")

# Slow request threshold (seconds). Requests exceeding this get a WARNING log.
SLOW_REQUEST_THRESHOLD = settings.get("YUNSHU_SLOW_REQUEST_THRESHOLD")

# Memory pressure warning threshold (fraction of system memory).
# Logged when active memory exceeds this fraction of hw.memsize.
_MEMORY_PRESSURE_THRESHOLD = 0.85


def _check_memory_pressure() -> None:
    """Log a warning if GPU memory usage is above the pressure threshold."""
    try:
        import mlx.core as mx

        active = mx.get_active_memory()
        if active <= 0:
            return
        # Only check against threshold if we can read system memory
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return
        total = int(result.stdout.strip())
        if total > 0:
            ratio = active / total
            if ratio >= _MEMORY_PRESSURE_THRESHOLD:
                logger.warning(
                    "MEMORY_PRESSURE: %.1f%% of UMA used (%.1fGB / %.1fGB)",
                    ratio * 100,
                    active / 1024**3,
                    total / 1024**3,
                )
    except Exception:
        pass


class RequestLoggingMiddleware:
    """Log all requests with structured fields (pure ASGI: streamed chunks pass untouched)."""

    SKIP_PATHS = {
        "/health",
        "/health/ready",
        "/health/live",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    # Periodic memory pressure check (avoid checking every single request).
    # Use threading.Lock for safe concurrent access across async coroutines
    # (asyncio is single-threaded in practice, but thread-safety is defensive).
    _last_mem_check: float = 0.0
    _mem_check_interval: float = 10.0  # seconds
    _mem_check_lock = threading.Lock()

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        request_id = (
            state.get("request_id")
            or Headers(scope=scope).get("X-Request-ID")
            or f"req_{uuid.uuid4().hex[:24]}"
        )
        state["request_id"] = request_id
        method, path = scope["method"], scope["path"]

        # Periodic memory pressure check (thread-safe compare-and-swap)
        now = time.monotonic()
        should_check = False
        with RequestLoggingMiddleware._mem_check_lock:
            if (
                now - RequestLoggingMiddleware._last_mem_check
                >= RequestLoggingMiddleware._mem_check_interval
            ):
                RequestLoggingMiddleware._last_mem_check = now
                should_check = True
        if should_check:
            _check_memory_pressure()

        t0 = time.monotonic()
        started = False

        async def send_with_id(message):
            nonlocal started
            if message["type"] == "http.response.start" and not started:
                started = True
                elapsed = time.monotonic() - t0
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
                status = message["status"]
                if path not in self.SKIP_PATHS:
                    level = logging.DEBUG
                    if status >= 500:
                        level = logging.ERROR
                    elif status >= 400:
                        level = logging.WARNING

                    logger.log(
                        level,
                        f"[{request_id}] {method} {path} {status} {elapsed * 1000:.1f}ms",
                    )

                    # Slow request warning
                    if elapsed >= SLOW_REQUEST_THRESHOLD and status < 500:
                        logger.warning(
                            "SLOW_REQUEST: [%s] %s %s %.1fms (threshold: %.0fms)",
                            request_id,
                            method,
                            path,
                            elapsed * 1000,
                            SLOW_REQUEST_THRESHOLD * 1000,
                        )
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception as e:
            if not started:
                elapsed = time.monotonic() - t0
                logger.error(
                    f"[{request_id}] {method} {path} ERROR {elapsed * 1000:.1f}ms — {e}"
                )
            raise
