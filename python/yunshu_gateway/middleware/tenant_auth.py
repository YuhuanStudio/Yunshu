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
from __future__ import annotations

import hmac
import os
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


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
    PUBLIC_PREFIXES = ("/static/", "/assets/", "/api/v1/gw/monitoring/")

    def _is_auth_enabled(self) -> bool:
        if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
            return False
        # Auth enabled when token is set
        return os.environ.get("YUNSHU_AUTH_TOKEN") is not None

    async def dispatch(self, request: Request, call_next):
        # Public paths never need auth
        path = request.url.path
        if path in self.PUBLIC_PATHS:
            return await call_next(request)
        if any(path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return await call_next(request)

        # Check if auth is enabled
        if not self._is_auth_enabled():
            return await call_next(request)

        auth_token = os.environ.get("YUNSHU_AUTH_TOKEN", "")
        auth = request.headers.get("Authorization", "")

        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth[7:]

        # ── RBAC auth (ys_ prefixed keys) ──
        if token.startswith("ys_"):
            rbac = getattr(request.app.state, "rbac_manager", None)
            if rbac is not None:
                api_key = rbac.authenticate(token)
                if api_key is None:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or expired API key"},
                    )
                request.state.rbac_key = api_key
                request.state.role = api_key.role
                request.state.slo_class = api_key.slo_class
                return await call_next(request)

        # ── Static token auth (constant-time comparison) ──
        if auth_token and hmac.compare_digest(token, auth_token):
            return await call_next(request)

        # ── Legacy TenantManager auth ──
        try:
            manager = getattr(request.app.state, "tenant_manager", None)
            if manager is not None:
                tenant = manager.authenticate(token)
                if tenant is None:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid API key"},
                    )
                if not tenant.check_rate_limit():
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                    )
                tenant.record_request()
                request.state.tenant = tenant
                return await call_next(request)
        except ImportError:
            pass

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )
