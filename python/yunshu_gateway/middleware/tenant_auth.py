"""Yunshu Gateway — Tenant authentication middleware.

Authenticates requests via:
1. RBACManager (ys_ prefixed keys) — role + SLO class
2. TenantManager (legacy) — simple quota enforcement
3. YUNSHU_AUTH_TOKEN (static env token) — fallback

Security:
- Auth is enabled when YUNSHU_AUTH_TOKEN is set or RBACManager is initialized
- Set YUNSHU_AUTH_DISABLED=true to disable auth (dev only)
- Health/docs endpoints remain public
"""

import hmac
import os
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)


# Paths served by the Anthropic router — must use Anthropic error format.
# Use exact matching, not endswith, to avoid overmatching paths like
# /api/v1/admin/messages that merely end in '/messages'.
_ANTHROPIC_PATHS = frozenset({
    "/v1/messages", "/messages",
    "/v1/messages/count_tokens", "/messages/count_tokens",
})


class _ErrorFormatter:
    """Format errors consistently per API family (OpenAI vs Anthropic)."""

    @staticmethod
    def auth_error(request: Request, message: str, status_code: int = 401) -> JSONResponse:
        path = request.url.path
        # WWW-Authenticate header only valid on 401, NOT on 429
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else {}
        # Anthropic 429: include Retry-After header
        if status_code == 429:
            headers["Retry-After"] = "1"
        if path in _ANTHROPIC_PATHS:
            error_type = "authentication_error" if status_code == 401 else "invalid_request_error"
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
                    "type": "authentication_error" if status_code == 401 else "rate_limit_error",
                    "code": code,
                }
            },
            headers=headers,
        )


class TenantAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via RBAC API keys.

    Auth is enforced when YUNSHU_AUTH_TOKEN is set or RBACManager is initialized.
    ys_ prefixed keys go through RBAC (role + SLO + permissions).
    Other keys fall through to legacy tenant/static token auth.

    Set YUNSHU_AUTH_DISABLED=true to disable auth (dev only).
    """

    PUBLIC_PATHS = {
        "/health", "/health/live", "/health/ready", "/version",
        "/docs", "/openapi.json", "/redoc",
        "/", "/favicon.ico",
    }
    # Prefixes that are always public (e.g., static assets)
    PUBLIC_PREFIXES = ("/static/", "/assets/")

    def _is_auth_enabled(self) -> bool:
        if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
            return False
        # Auth enabled when non-empty static token OR RBAC keys are configured.
        # An empty YUNSHU_AUTH_TOKEN is treated as unset — setting it to '' should
        # not silently enable auth with a token that can never match.
        _static = os.environ.get("YUNSHU_AUTH_TOKEN")
        if _static is not None and _static:
            return True
        try:
            rbac = getattr(self.app.state, "rbac_manager", None)
            if rbac is not None and hasattr(rbac, "is_enabled") and rbac.is_enabled():
                return True
        except Exception:
            pass
        return False

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

        # Check if auth is enabled
        if not self._is_auth_enabled():
            return await call_next(request)

        auth_token = os.environ.get("YUNSHU_AUTH_TOKEN", "")
        auth = request.headers.get("Authorization", "")

        if not auth.startswith("Bearer "):
            return _ErrorFormatter.auth_error(request, "Missing or invalid Authorization header")

        token = auth[7:]

        # ── RBAC auth (ys_ prefixed keys) ──
        if token.startswith("ys_"):
            rbac = getattr(request.app.state, "rbac_manager", None)
            if rbac is not None:
                api_key = rbac.authenticate(token)
                if api_key is None:
                    return _ErrorFormatter.auth_error(request, "Invalid or expired API key")
                request.state.rbac_key = api_key
                request.state.role = api_key.role
                request.state.slo_class = api_key.slo_class
                return await call_next(request)
            # ys_ prefixed tokens MUST go through RBAC — reject if unavailable
            return _ErrorFormatter.auth_error(request, "Invalid or expired API key")

        # ── Static token auth (constant-time comparison) ──
        if auth_token and hmac.compare_digest(token, auth_token):
            return await call_next(request)

        # ── Legacy TenantManager auth ──
        try:
            manager = getattr(request.app.state, "tenant_manager", None)
            if manager is not None:
                tenant = manager.authenticate(token)
                if tenant is None:
                    return _ErrorFormatter.auth_error(request, "Invalid API key")
                if not tenant.check_and_record():
                    return _ErrorFormatter.auth_error(
                        request, "Rate limit exceeded", status_code=429,
                    )
                request.state.tenant = tenant
                response = await call_next(request)
                # Decrement active_requests when the response finishes
                if isinstance(response, StreamingResponse):
                    response.background = BackgroundTask(tenant.finish_request)
                else:
                    tenant.finish_request()
                return response
        except Exception:
            # Catch ALL exceptions (not just ImportError) so that a broken
            # TenantManager falls through to the final 401 instead of
            # surfacing a 500 to the client.
            pass

        return _ErrorFormatter.auth_error(request, "Invalid or missing API key")
