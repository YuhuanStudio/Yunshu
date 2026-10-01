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

from .body_replay import replay_receive

logger = logging.getLogger(__name__)

_STATEFUL_MARKERS = (
    b'"file_id"',
    b'"previous_response_id"',
    b'"conversation"',
    b'"input_file"',
)


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
                await self.app(scope, replay_receive(body, receive), send)
                return

            # namespace the cache by the caller's
            # credential. The key was body-only, so under multi-tenant RBAC a HIT
            # returned User A's cached response to User B for a byte-identical body —
            # a cross-tenant data leak AND a model-authorization bypass (the HIT
            # short-circuits before the route's _check_model_access runs). Folding the
            # bearer token in means a HIT can only be served to the same key that
            # cached it (and that key already passed model-access for that body).
            _auth = request.headers.get("Authorization", "")
            # A body that references stored state (uploaded files, earlier responses,
            # conversations) can change meaning without the body changing (file expiry
            # or deletion, store edits), and a HIT would skip the route's validation.
            if any(m in body for m in _STATEFUL_MARKERS):
                await self.app(scope, replay_receive(body, receive), send)
                return
            # The engine generation (which loaded model instance serves the request) is
            # part of the key so an unload / reload / swap never serves old output.
            from yunshu_gateway.engine import engine_generation

            cache_key = hashlib.sha256(
                _auth.encode("utf-8")
                + b"\x00"
                + str(engine_generation()).encode("utf-8")
                + b"\x00"
                + body
            ).hexdigest()

            # Skip caching for non-deterministic sampling (temperature > 0, no seed)
            temperature = body_json.get("temperature", 1.0)
            seed = body_json.get("seed")
            if temperature > 0 and seed is None:
                await self.app(scope, replay_receive(body, receive), send)
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
            await self.app(scope, replay_receive(body, receive), send)
            return

        # Cache miss — capture response.
        # Provide the cached body via synthetic receive so downstream
        # handlers don't get an empty body from the consumed receive.
        _replay_receive = replay_receive(body, receive)

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
                        _cache_task = asyncio.ensure_future(
                            cache.put(cache_key, result)
                        )
                        # hold strong ref in module-level set so
                        # GC doesn't collect the task before cache.put completes.
                        self._pending_cache_tasks.add(_cache_task)

                        def _log_cache_error(t):
                            self._pending_cache_tasks.discard(t)
                            if not t.cancelled():
                                exc = t.exception()
                                if exc:
                                    logger.debug(
                                        "Cache put failed: %s", exc, exc_info=True
                                    )

                        _cache_task.add_done_callback(_log_cache_error)
                    except Exception:
                        logger.debug("cache store failed", exc_info=True)

        await self.app(scope, _replay_receive, send_wrapper)
