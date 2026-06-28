from __future__ import annotations

"""Gateway optimizer middleware — ResponseCache + RequestCoalescer.

Wires the gateway optimizer modules into the request lifecycle:
- ResponseCache: caches non-streaming chat/completions responses
- StreamingResponseBuffer: ring buffer for streaming responses

Both are opt-in via YUNSHU_RESPONSE_CACHE=1 env var.
"""

import asyncio
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

    # hold strong refs to cache-put tasks so GC doesn't
    # eat them before completion (Python event loop only holds weak refs).
    _pending_cache_tasks: set = set()

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
                # Must provide the cached body back to downstream via a
                # synthetic receive — the original receive was consumed.
                async def _cached_receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                await self.app(scope, _cached_receive, send)
                return

            # namespace the cache by the caller's
            # credential. The key was body-only, so under multi-tenant RBAC a HIT
            # returned User A's cached response to User B for a byte-identical body —
            # a cross-tenant data leak AND a model-authorization bypass (the HIT
            # short-circuits before the route's _check_model_access runs). Folding the
            # bearer token in means a HIT can only be served to the same key that
            # cached it (and that key already passed model-access for that body).
            _auth = request.headers.get("Authorization", "")
            cache_key = hashlib.sha256(_auth.encode("utf-8") + b"\x00" + body).hexdigest()

            # Skip caching for non-deterministic sampling (temperature > 0, no seed)
            temperature = body_json.get("temperature", 1.0)
            seed = body_json.get("seed")
            if temperature > 0 and seed is None:
                async def _nondet_receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                await self.app(scope, _nondet_receive, send)
                return

            # cache.get() is async (uses asyncio.Lock internally)
            hit = await cache.get(cache_key)
            if hit is not None:
                response = JSONResponse(content=hit)
                response.headers["X-Cache"] = "HIT"
                await response(scope, receive, send)
                return
        except Exception:
            logger.debug("cache lookup failed", exc_info=True)
            async def _fallback_receive():
                return {"type": "http.request", "body": body, "more_body": False}
            await self.app(scope, _fallback_receive, send)
            return

        # Cache miss — capture response.
        # Provide the cached body via synthetic receive so downstream
        # handlers don't get an empty body from the consumed receive.
        async def _replay_receive():
            return {"type": "http.request", "body": body, "more_body": False}

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
                        # cache.put() is async (uses asyncio.Lock internally).
                        # Add done callback for error logging so failures aren't
                        # silently swallowed by the fire-and-forget task.
                        _cache_task = asyncio.ensure_future(cache.put(cache_key, result))
                        # hold strong ref in module-level set so
                        # GC doesn't collect the task before cache.put completes.
                        self._pending_cache_tasks.add(_cache_task)
                        def _log_cache_error(t):
                            self._pending_cache_tasks.discard(t)
                            if not t.cancelled():
                                exc = t.exception()
                                if exc:
                                    logger.debug("Cache put failed: %s", exc, exc_info=True)
                        _cache_task.add_done_callback(_log_cache_error)
                    except Exception:
                        logger.debug("cache store failed", exc_info=True)

        await self.app(scope, _replay_receive, send_wrapper)


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
        body = None
        try:
            body = await request.body()
            body_json = json.loads(body)
            stream = body_json.get("stream", False)
            body_json.get("model", "default")
            if not stream:
                coalescer._stats.total_requests += 1
        except Exception:
            logger.debug("coalescer tracking failed", exc_info=True)

        # Replay the consumed body so downstream handlers can read it.
        # If body read failed (body is None), fall through with original
        # receive — downstream will attempt to read the body itself.
        if body is not None:
            async def _replay_receive():
                return {"type": "http.request", "body": body, "more_body": False}
            await self.app(scope, _replay_receive, send)
        else:
            await self.app(scope, receive, send)
