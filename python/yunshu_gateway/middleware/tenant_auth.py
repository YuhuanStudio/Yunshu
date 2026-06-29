"""Yunshu Gateway — single optional bearer-token auth middleware.

Authenticates requests via a single static bearer token (``YUNSHU_AUTH_TOKEN``).
This engine serves a single consumer (the digital being Yunmo); the multi-tenant
RBACManager / TenantManager machinery has been removed.

Security posture (unchanged, safe by default):
- Auth is enabled when ``YUNSHU_AUTH_TOKEN`` is set.
- Set ``YUNSHU_AUTH_DISABLED=true`` to disable auth (dev only).
- When no token is configured AND auth is not explicitly disabled, admin
  endpoints remain protected by their own ``_check_auth`` (deny by default);
  inference endpoints stay accessible.
- Health/docs endpoints remain public.

On every authenticated request the single-owner identity (constant ``"owner"``,
overridable via ``YUNSHU_ACTOR_IDENTITY``) is stamped on
``request.state.role="owner"`` and on ``current_actor`` so that the
``request_tracker`` ownership / ``engine_core`` dedup keys keep working in the
single-consumer model.
"""

import contextlib
import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# Paths served by the Anthropic router — must use Anthropic error format.
# Use exact matching, not endswith, to avoid overmatching paths like
# /api/v1/admin/messages that merely end in '/messages'.
_ANTHROPIC_PATHS = frozenset(
    {
        "/v1/messages",
        "/messages",
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
    }
)


class _ErrorFormatter:
    """Format errors consistently per API family (OpenAI vs Anthropic)."""

    @staticmethod
    def auth_error(
        request: Request, message: str, status_code: int = 401
    ) -> JSONResponse:
        path = request.url.path
        # WWW-Authenticate header only valid on 401, NOT on 429
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else {}
        # Anthropic 429: include Retry-After header
        if status_code == 429:
            headers["Retry-After"] = "1"
        # MCP is JSON-RPC — a middleware-level auth/rate-limit denial on /v1/mcp
        # must return a JSON-RPC 2.0 error object, not the OpenAI envelope a JSON-RPC client
        # can't parse. id is null (no parsed body).
        if path == "/v1/mcp":
            return JSONResponse(
                status_code=status_code,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": message},  # INVALID_REQUEST
                    "id": None,
                },
                headers=headers,
            )
        if path in _ANTHROPIC_PATHS:
            # Map per Anthropic's error-type contract — 429 → rate_limit_error.
            error_type = {
                401: "authentication_error",
                403: "permission_error",
                429: "rate_limit_error",
            }.get(status_code, "invalid_request_error")
            return JSONResponse(
                status_code=status_code,
                content={
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": message,
                    },
                },
                headers=headers,
            )
        code = "invalid_api_key" if status_code == 401 else "rate_limit_exceeded"
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": "authentication_error"
                    if status_code == 401
                    else "rate_limit_error",
                    "code": code,
                }
            },
            headers=headers,
        )


def _owner_identity() -> str:
    """Single-consumer owner identity, overridable via YUNSHU_ACTOR_IDENTITY."""
    custom = os.environ.get("YUNSHU_ACTOR_IDENTITY", "").strip()
    return custom or "owner"


class TenantAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via a single optional static bearer token.

    Name kept for backward-compat with ``app.add_middleware(TenantAuthMiddleware)``
    in main.py and existing tests; the per-tenant/RBAC behavior is gone.
    """

    PUBLIC_PATHS = {
        "/health",
        "/health/live",
        "/health/ready",
        "/version",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/",
        "/favicon.ico",
    }
    # Prefixes that are always public (e.g., static assets)
    PUBLIC_PREFIXES = ("/static/", "/assets/")

    def _is_auth_enabled(self) -> bool:
        if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
            return False
        _static = os.environ.get("YUNSHU_AUTH_TOKEN")
        return bool(_static)

    async def dispatch(self, request: Request, call_next):
        # Public paths never need auth
        path = request.url.path
        if path in self.PUBLIC_PATHS:
            return await call_next(request)
        if any(path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return await call_next(request)

        # CORS preflight (OPTIONS) must pass through without auth —
        # browsers send OPTIONS without Authorization headers, and the
        # CORSMiddleware (inner) needs to respond before any auth check.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Stamp the single-owner identity for every request so downstream
        # ownership (request_tracker) and dedup (engine_core) keep working
        # even when auth is disabled — single consumer, no isolation needed.
        owner = _owner_identity()
        with contextlib.suppress(Exception):
            request.state.role = "owner"
        with contextlib.suppress(Exception):
            from yunshu_engine.request_tracker import current_actor

            current_actor.set(owner)

        # Check if auth is enabled
        if not self._is_auth_enabled():
            return await call_next(request)

        auth_token = os.environ.get("YUNSHU_AUTH_TOKEN", "")
        auth = request.headers.get("Authorization", "")

        if not auth.startswith("Bearer "):
            return _ErrorFormatter.auth_error(
                request, "Missing or invalid Authorization header"
            )

        token = auth[7:]

        # Static token auth (constant-time comparison). The single consumer
        # authenticates with the one configured bearer token.
        if auth_token and hmac.compare_digest(token, auth_token):
            return await call_next(request)

        return _ErrorFormatter.auth_error(request, "Invalid or missing API key")
