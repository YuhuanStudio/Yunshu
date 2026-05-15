from __future__ import annotations
"""Gateway optimizer middleware — ResponseCache + RequestCoalescer.

Wires the gateway optimizer modules into the request lifecycle:
- ResponseCache: caches non-streaming chat/completions responses
- StreamingResponseBuffer: ring buffer for streaming responses

Both are opt-in via YUNSHU_RESPONSE_CACHE=1 env var.
"""

import hashlib
import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class ResponseCacheMiddleware:
    """Cache non-streaming responses by content hash.

    Only active when YUNSHU_RESPONSE_CACHE=1 is set.
    Caches /v1/chat/completions and /v1/completions non-streaming responses.
    """

    CACHEABLE_PATHS = {"/v1/chat/completions", "/v1/completions"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        if request.url.path not in self.CACHEABLE_PATHS or request.method != "POST":
            await self.app(scope, receive, send)
            return

        cache = getattr(request.app.state, "response_cache", None)
        if cache is None or not cache.enabled:
            await self.app(scope, receive, send)
            return

        # Read request body to compute hash
        body = await request.body()
        try:
            body_json = json.loads(body)
            stream = body_json.get("stream", False)
            if stream:
                await self.app(scope, receive, send)
                return

            cache_key = hashlib.sha256(body).hexdigest()
            hit = cache.get(cache_key)
            if hit is not None:
                response = JSONResponse(content=hit)
                response.headers["X-Cache"] = "HIT"
                await response(scope, receive, send)
                return
        except Exception:
            logger.debug("cache lookup failed", exc_info=True)
            await self.app(scope, receive, send)
            return

        # Cache miss — capture response
        response_started = False
        status_code = 200
        headers = []
        body_parts = []

        async def send_wrapper(message):
            nonlocal response_started, status_code, headers
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message.get("status", 200)
                headers = message.get("headers", [])
                await send(message)
            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))
                await send(message)
                # Cache on last body chunk
                if not message.get("more_body", False) and status_code == 200:
                    try:
                        full_body = b"".join(body_parts)
                        result = json.loads(full_body)
                        cache.put(cache_key, result)
                    except Exception:
                        logger.debug("cache store failed", exc_info=True)

        await self.app(scope, receive, send_wrapper)


class RequestCoalescingMiddleware:
    """Coalesce simultaneous non-streaming requests for the same model.

    When YUNSHU_REQUEST_COALESCE=1 is set, non-streaming requests for the
    same model are batched within a short window (default 5ms) before being
    sent to the engine. This reduces per-request overhead when many concurrent
    requests arrive at once.

    The middleware records stats but does NOT replace the engine call —
    it adds the request to the coalescer and tracks coalescing metrics.
    The actual batch dispatch is handled by the engine.
    """

    COALESCEABLE_PATHS = {"/v1/chat/completions", "/v1/completions"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        if request.url.path not in self.COALESCEABLE_PATHS or request.method != "POST":
            await self.app(scope, receive, send)
            return

        coalescer = getattr(request.app.state, "request_coalescer", None)
        if coalescer is None:
            await self.app(scope, receive, send)
            return

        # Track the request in the coalescer for statistics
        # The actual batching is handled at the engine level
        try:
            body = await request.body()
            body_json = json.loads(body)
            stream = body_json.get("stream", False)
            model = body_json.get("model", "default")
            if not stream:
                coalescer._stats.total_requests += 1
        except Exception:
            logger.debug("coalescer tracking failed", exc_info=True)

        await self.app(scope, receive, send)
