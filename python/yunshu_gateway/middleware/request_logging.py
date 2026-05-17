"""Yunshu Gateway — Request logging middleware.

Structured request/response logging with:
- Request ID tracking (X-Request-ID header)
- Latency measurement with slow request warnings
- Active request counting for graceful shutdown drain
- SSE-aware counting: keeps stream counted until body fully consumed
- Configurable log levels per status code
"""

import os
import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse

logger = logging.getLogger("yunshu.gateway")

# Slow request threshold (seconds). Requests exceeding this get a WARNING log.
SLOW_REQUEST_THRESHOLD = float(os.environ.get("YUNSHU_SLOW_REQUEST_THRESHOLD", "30.0"))

# Memory pressure warning threshold (fraction of system memory).
# Logged when active memory exceeds this fraction of hw.memsize.
_MEMORY_PRESSURE_THRESHOLD = float(os.environ.get("YUNSHU_MEM_WARNING", "0.85"))


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
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
        )
        if result.returncode != 0:
            return
        total = int(result.stdout.strip())
        if total > 0:
            ratio = active / total
            if ratio >= _MEMORY_PRESSURE_THRESHOLD:
                logger.warning(
                    "MEMORY_PRESSURE: %.1f%% of UMA used (%.1fGB / %.1fGB)",
                    ratio * 100, active / 1024**3, total / 1024**3,
                )
    except Exception:
        pass


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with structured fields."""

    SKIP_PATHS = {"/health", "/health/ready", "/health/live", "/metrics", "/docs", "/openapi.json", "/redoc"}

    # Periodic memory pressure check (avoid checking every single request)
    _last_mem_check: float = 0.0
    _mem_check_interval: float = 10.0  # seconds

    async def dispatch(self, request: Request, call_next):
        import yunshu_gateway.main as _main

        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:24]}"
        request.state.request_id = request_id

        _main._active_requests += 1

        # Periodic memory pressure check
        now = time.monotonic()
        if now - RequestLoggingMiddleware._last_mem_check >= RequestLoggingMiddleware._mem_check_interval:
            RequestLoggingMiddleware._last_mem_check = now
            _check_memory_pressure()

        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"ERROR {elapsed*1000:.1f}ms — {e}"
            )
            # Decrement counter that was incremented above — the response
            # path below won't run since we're re-raising.
            _main._active_requests -= 1
            if _main._active_requests <= 0 and _main._drain_event is not None:
                _main._drain_event.set()
            raise

        elapsed = time.monotonic() - t0
        response.headers["X-Request-ID"] = request_id

        # For SSE responses, keep request counted until stream completes
        is_sse = isinstance(response, StreamingResponse)
        if is_sse:
            original_body = response.body_iterator

            async def _tracked_body():
                try:
                    async for chunk in original_body:
                        yield chunk
                finally:
                    _main._active_requests -= 1
                    if _main._active_requests == 0 and _main._drain_event is not None:
                        _main._drain_event.set()

            response.body_iterator = _tracked_body()
        else:
            _main._active_requests -= 1
            if _main._active_requests == 0 and _main._drain_event is not None:
                _main._drain_event.set()

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

            # Slow request warning
            if elapsed >= SLOW_REQUEST_THRESHOLD and response.status_code < 500:
                logger.warning(
                    "SLOW_REQUEST: [%s] %s %s %.1fms (threshold: %.0fms)",
                    request_id, request.method, request.url.path,
                    elapsed * 1000, SLOW_REQUEST_THRESHOLD * 1000,
                )

        return response
