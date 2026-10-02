"""Hold model leases for the lifetime of each request / connection.

Pure ASGI, outermost: opens a ``LeaseScope`` before the app runs and releases it
when the app returns, whether the response finished, errored, was cancelled or the
client disconnected. Every ``ModelManager.get_engine`` inside the request attaches
its lease to the scope, so unload / TTL / LRU eviction skip the model meanwhile.
"""

from __future__ import annotations

from yunshu_engine.model_manager import lease_scope


class ModelLeaseMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        with lease_scope():
            await self.app(scope, receive, send)
