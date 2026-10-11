"""Registers the console's management routers (downloads, cache, logs) on the app."""

from __future__ import annotations

from fastapi import FastAPI

from .. import log_ring
from . import admin_cache, admin_logs, admin_models


def register(app: FastAPI) -> None:
    log_ring.install()
    for mod in (admin_models, admin_cache, admin_logs):
        app.include_router(mod.router, prefix="/v1")
