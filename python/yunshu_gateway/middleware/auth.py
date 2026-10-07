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
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from yunshu_engine import settings
from yunshu_gateway import api_keys
from yunshu_gateway.error_envelope import format_error_response
from yunshu_gateway.token_compare import tokens_equal

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


async def _release_after(body, store, key_id):
    """Yield the response body, then free the key's concurrency slot (stream end or abort)."""
    try:
        async for chunk in body:
            yield chunk
    finally:
        store.release(key_id)


def _owner_identity() -> str:
    """Single-consumer owner identity, overridable via YUNSHU_ACTOR_IDENTITY."""
    return settings.get("YUNSHU_ACTOR_IDENTITY") or "owner"


class AuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via a single optional static bearer token."""

    PUBLIC_PATHS = {
        "/health",
        "/health/live",
        "/health/ready",
        "/version",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/",
        "/console",
        "/favicon.ico",
    }
    # Prefixes that are always public (e.g., static assets)
    PUBLIC_PREFIXES = ("/static/", "/assets/", "/console/")

    def _is_auth_enabled(self) -> bool:
        if settings.get_bool("YUNSHU_AUTH_DISABLED"):
            return False
        _static = settings.get("YUNSHU_AUTH_TOKEN")
        if _static:
            return True
        from yunshu_gateway import api_keys

        return (
            api_keys.get_store().has_keys()
        )  # stored keys switch auth on, like a token

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

        auth_token = settings.get("YUNSHU_AUTH_TOKEN") or ""
        auth = request.headers.get("Authorization", "")
        # Anthropic SDKs send the key as x-api-key instead of a bearer token.
        api_key = request.headers.get("x-api-key", "")

        if auth.startswith("Bearer "):
            token = auth[7:]
        elif api_key:
            token = api_key
        else:
            return _ErrorFormatter.auth_error(
                request, "Missing or invalid Authorization header"
            )

        # Static token auth (constant-time comparison). The single consumer
        # authenticates with the one configured bearer token.
        if auth_token and tokens_equal(token, auth_token):
            with contextlib.suppress(Exception):
                request.state.principal = api_keys.ADMIN_TOKEN
            return await call_next(request)

        # Stored API keys: enabled / expiry, then quotas (429 + Retry-After).
        store = api_keys.get_store()
        if store.has_keys():
            try:
                principal = store.authenticate(token)
                store.admit(principal)
            except api_keys.AuthFailureError as exc:
                return _ErrorFormatter.auth_error(request, str(exc), exc.status)
            except api_keys.QuotaExceededError as exc:
                return format_error_response(
                    request.url.path,
                    str(exc),
                    429,
                    code=f"{exc.which}_exceeded",
                    retry_after=exc.retry_after,
                )
            request.state.principal = principal
            request.state.api_key_id = principal.key_id
            try:
                response = await call_next(request)
            except BaseException:
                store.release(principal.key_id)
                raise
            response.body_iterator = _release_after(
                response.body_iterator, store, principal.key_id
            )
            return response

        return _ErrorFormatter.auth_error(request, "Invalid or missing API key")
