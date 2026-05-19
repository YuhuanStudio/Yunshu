from __future__ import annotations
"""OpenAI Models API compatible router."""

import asyncio
import logging
import os
import threading
import time

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["models"])


def _check_permission(request: Request, permission: str) -> None:
    """Check RBAC permission on gateway endpoints.

    Security (deny-by-default):
    1. If YUNSHU_AUTH_DISABLED=true, allow (dev opt-in, logged at startup)
    2. If rbac_key is set (from TenantAuthMiddleware), check has_permission()
    3. If rbac_key is None but YUNSHU_AUTH_TOKEN is set, allow (static token = admin)
    4. If no auth configured and not disabled — DENY access (secure default)
    """
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return
    rbac_key = getattr(request.state, "rbac_key", None)
    if rbac_key is not None:
        if not rbac_key.has_permission(permission):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return
    # No RBAC key — check if static token auth is active
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if auth_token is not None and auth_token:
        return  # Static token = admin access
    # No auth configured and not explicitly disabled — deny by default
    logger.warning(
        "Admin endpoint access denied: no auth configured. "
        "Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true to control access."
    )
    raise HTTPException(
        status_code=401,
        detail="No authentication configured. Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true.",
    )

# Guard against concurrent load/unload of the same model
_model_ops_lock = asyncio.Lock()
_model_ops_inflight: set[str] = set()


class LoadModelRequest(BaseModel):
    model: str
    pin: bool = False


@router.get("/models")
async def list_models() -> dict:
    """List available models (OpenAI-compatible)."""
    models = []

    # Multi-model mode
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            model_info = {
                "id": entry.model_id,
                "object": "model",
                "created": int(entry.load_time) if hasattr(entry, 'load_time') and entry.load_time else int(time.time()),
                "owned_by": "yunshu",
                "loaded": entry.is_loaded,
                "type": entry.model_type.name,
                "size_gb": round(entry.estimated_bytes / 1e9, 1),
            }
            if entry.is_loaded and entry.engine is not None:
                try:
                    stats = entry.engine.get_stats() if hasattr(entry.engine, 'get_stats') else {}
                    model_info["stats"] = stats
                except Exception:
                    logger.debug(f"failed to get stats for {entry.model_id}", exc_info=True)
            models.append(model_info)
        result = {"object": "list", "data": models}
        # Include model registry stats for debugging/monitoring
        try:
            from yunshu_engine.model_registry import get_registry
            result["registry"] = get_registry().get_stats()
        except Exception:
            logger.debug("model_registry stats unavailable", exc_info=True)
        return result

    # Single-engine mode
    engine = get_engine()
    if engine and engine.is_loaded:
        _load_time = getattr(engine, '_load_time', None) or getattr(engine, 'load_time', None)
        models.append({
            "id": engine.model_name,
            "object": "model",
            "created": int(_load_time) if _load_time else int(time.time()),
            "owned_by": "yunshu",
        })
    return {"object": "list", "data": models}


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> dict:
    """Get details for a specific model."""
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return {
            "id": entry.model_id,
            "object": "model",
            "owned_by": "yunshu",
            "loaded": entry.is_loaded,
        }

    engine = get_engine()
    if not engine or not engine.resolve_model_id(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {
        "id": engine.model_name,
        "object": "model",
        "owned_by": "yunshu",
    }


@router.post("/models/load")
async def load_model(req: LoadModelRequest, request: Request) -> dict:
    """Load a model (supports both single-engine and multi-model modes)."""
    _check_permission(request, "can_load_models")
    if not req.model or not req.model.strip():
        raise HTTPException(status_code=400, detail="model field cannot be empty")

    # Guard against concurrent load/unload of the same model
    async with _model_ops_lock:
        if req.model in _model_ops_inflight:
            raise HTTPException(status_code=409, detail=f"Model '{req.model}' is already being loaded or unloaded")
        _model_ops_inflight.add(req.model)

    try:
        manager = get_model_manager()

        if manager is not None:
            try:
                engine = await manager.get_engine(req.model)
                if hasattr(engine, 'is_running') and not engine.is_running:
                    await engine.start()
                return {"status": "loaded", "model": req.model}
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Model '{req.model}' not registered")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Model load error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Model loading failed")

        # Single-engine mode
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        from yunshu_engine.batched_engine import BatchedEngine
        if isinstance(engine, BatchedEngine):
            raise HTTPException(status_code=400, detail="Single-engine load not supported in batched mode — use model manager")
        from yunshu_engine.mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), engine.load, req.model)
        await engine.start()

        return {"status": "loaded", "model": req.model}
    finally:
        async with _model_ops_lock:
            _model_ops_inflight.discard(req.model)


@router.post("/models/unload/{model_id}")
async def unload_model(model_id: str, request: Request) -> dict:
    """Unload a model and release memory."""
    _check_permission(request, "can_unload_models")
    # Guard against concurrent load/unload of the same model
    async with _model_ops_lock:
        if model_id in _model_ops_inflight:
            raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already being loaded or unloaded")
        _model_ops_inflight.add(model_id)

    try:
        manager = get_model_manager()
        if manager is None:
            raise HTTPException(status_code=400, detail="Multi-model mode not active")

        entry = manager.get_entry(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        await manager.unload_model(model_id)
        return {"status": "unloaded", "model": model_id}
    finally:
        async with _model_ops_lock:
            _model_ops_inflight.discard(model_id)
