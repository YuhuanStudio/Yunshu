"""Prefix-cache (APC) management for the console.

- ``GET  /v1/yunshu/cache/tiers``  tiers (RAM / WARM / SSD): bytes, entries, hits; lookup
                                   counters; entry metadata capped at 200 (hash label,
                                   token count, bytes, tier, LRU rank, hits; never text)
- ``POST /v1/yunshu/cache/tiers/clear``  ``{"tier": "ram"|"warm"|"ssd"|null, "model": id|null}``
                                   drop a tier (all tiers when omitted), report bytes freed

(``/v1/yunshu/cache`` and ``/cache/clear`` are the entry-lifecycle view and the resident-APC
clear in ``yunshu.py``.) Accounting and eviction only: lookups, keys and token identity are untouched. Reads follow
the inference endpoints' access; clearing needs the ``admin`` permission.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from yunshu_engine.mlx_executor import get_mlx_executor

from ..engine import get_engine, get_model_manager
from .models import _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["yunshu"])

MAX_ENTRIES = 200
TIERS = ("ram", "warm", "ssd")


class ClearRequest(BaseModel):
    tier: Literal["ram", "warm", "ssd"] | None = None
    model: str | None = None


def _engines() -> list[tuple[str, Any]]:
    """(model id, engine) for every loaded engine that has a prefix cache."""
    out: list[tuple[str, Any]] = []
    manager = get_model_manager()
    if manager is not None:
        for e in manager.list_entries():
            if e.is_loaded and callable(getattr(e.engine, "apc_overview", None)):
                out.append((e.model_id, e.engine))
    else:
        eng = get_engine()
        if eng is not None and callable(getattr(eng, "apc_overview", None)):
            out.append((str(getattr(eng, "model_name", "model")), eng))
    return out


@router.get("/yunshu/cache/tiers")
async def cache_overview(
    request: Request,
    entries: int = Query(MAX_ENTRIES, ge=0, le=MAX_ENTRIES),
) -> dict[str, Any]:
    _check_permission(request, "can_infer")
    caches = []
    for model_id, eng in _engines():
        try:
            view = eng.apc_overview(entries)
        except Exception:  # noqa: BLE001
            logger.debug("cache overview failed for %s", model_id, exc_info=True)
            caches.append({"model": model_id, "error": "unavailable"})
            continue
        if view is not None:
            caches.append({"model": model_id, **view})
    return {"caches": caches, "enabled": bool(caches)}


@router.post("/yunshu/cache/tiers/clear")
async def clear_cache(
    request: Request, body: ClearRequest | None = None
) -> dict[str, Any]:
    _check_permission(request, "admin")
    body = body or ClearRequest()
    tiers = (body.tier,) if body.tier else TIERS
    targets = _engines()
    if body.model is not None:
        targets = [t for t in targets if t[0] == body.model]
        if not targets:
            raise HTTPException(404, f"no prefix cache for model '{body.model}'")
    loop = asyncio.get_running_loop()
    results = []
    for model_id, eng in targets:
        for tier in tiers:
            # On the MLX thread, like every other owner of those buffers.
            res = await loop.run_in_executor(get_mlx_executor(), eng.apc_clear, tier)
            if res is not None:
                results.append({"model": model_id, **res})
    return {
        "cleared": results,
        "freed_bytes": sum(r.get("freed_bytes", 0) for r in results),
    }
