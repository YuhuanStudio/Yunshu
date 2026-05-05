"""Yunshu Control Plane — Token-based authentication middleware.

Supports Bearer token authentication (oMLX pattern).
Token is set via YUNSHU_AUTH_TOKEN env var.
Skips auth for health/live/ready/version endpoints.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Paths that never require authentication
PUBLIC_PATHS = {
    "/health",
    "/health/live",
    "/health/ready",
    "/version",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on all requests except health/version."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        # Allow public endpoints without auth
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check Authorization header
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
            if provided == self.token:
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Provide valid Bearer token."},
            headers={"WWW-Authenticate": "Bearer"},
        )
