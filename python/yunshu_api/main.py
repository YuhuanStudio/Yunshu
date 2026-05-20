"""Yunshu Control Plane — FastAPI app factory.

Separate FastAPI app for the admin/management API (L2).
This mounts under /admin or runs as a separate service.
"""


import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.auth import AuthMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield


def create_admin_app() -> FastAPI:
    app = FastAPI(
        title="Yunshu Control Plane",
        version="0.1.0-dev",
        description="Admin and monitoring API for Yunshu inference platform",
        lifespan=lifespan,
    )

    # CORS: configurable via YUNSHU_CORS_ORIGINS (comma-separated).
    # Defaults to localhost origins for admin API security.
    _cors_str = os.environ.get("YUNSHU_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    if _cors_str == "*":
        _cors_origins = ["*"]
        logger.warning("CORS: allow_origins=['*'] on control plane — restrict YUNSHU_CORS_ORIGINS for production")
    else:
        _cors_origins = [o.strip() for o in _cors_str.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_origins != ["*"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
    )

    # Auth middleware (token-based)
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if auth_token:
        app.add_middleware(AuthMiddleware, token=auth_token)

    # Register routers
    from .routers import admin, monitoring

    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(monitoring.router, prefix="/api/v1")

    @app.get("/health")
    async def admin_health():
        return {"status": "ok", "service": "yunshu-control-plane"}

    return app


app = create_admin_app()
