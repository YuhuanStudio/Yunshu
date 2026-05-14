"""Yunshu Control Plane — Admin router.

Model lifecycle management: register, load, unload, list, delete.
Configuration management: engine config, auth tokens.
RBAC key management: create, list, revoke, delete API keys with roles.
System monitoring: hardware status, server metrics, prefill progress, model discovery.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..schemas.models import (
    AuthTokenCreate,
    AuthTokenResponse,
    EngineConfigUpdate,
    ModelListResponse,
    ModelLoadRequest,
    ModelRegisterRequest,
    ModelResponse,
    ModelUnloadRequest,
    RBACTokenCreate,
    RBACTokenResponse,
)

router = APIRouter(prefix="/admin", tags=["admin"])

logger = logging.getLogger(__name__)


# ── RBAC Helpers ──


def _get_rbac_manager(request: Request):
    from yunshu_control.role_manager import RBACManager

    manager = getattr(request.app.state, "rbac_manager", None)
    if manager is None:
        manager = RBACManager()
        request.app.state.rbac_manager = manager
    return manager


def require_permission(permission: str):
    """FastAPI dependency that checks RBAC permission on admin endpoints."""
    def _check(request: Request):
        import os
        auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
        if not auth_token:
            return  # No auth configured, allow all

        rbac_key = getattr(request.state, "rbac_key", None)
        if rbac_key is not None:
            if not rbac_key.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: requires {permission}",
                )
    return _check


# ── Model Management ──


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request, _=Depends(require_permission("can_view_admin"))):
    """List all registered models and their status."""
    from yunshu_gateway.engine import get_model_manager, get_engine

    models = []
    manager = get_model_manager()

    if manager is not None:
        for mid, entry in manager._entries.items():
            models.append(ModelResponse(
                model_id=mid,
                model_type=entry.model_type.value if hasattr(entry.model_type, 'value') else str(entry.model_type),
                status="loaded" if entry.is_loaded else "registered",
                size_bytes=entry.estimated_bytes,
                pinned=entry.pinned,
            ))
    else:
        engine = get_engine()
        if engine and engine.is_loaded:
            models.append(ModelResponse(
                model_id=engine.model_name or "default",
                model_type="LLM",
                status="loaded" if engine.is_loaded else "registered",
            ))

    return ModelListResponse(models=models, total=len(models))


@router.post("/models/register", response_model=ModelResponse)
async def register_model(req: ModelRegisterRequest, request: Request, _=Depends(require_permission("can_register_models"))):
    """Register a new model for serving."""
    import os
    from pathlib import Path
    from yunshu_gateway.engine import get_model_manager

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized. Start with multi-model mode.")

    model_path = Path(req.model_path)
    if not model_path.exists():
        # Try as HuggingFace ID
        models_dir = os.environ.get("YUNSHU_MODELS_DIR", "models")
        local_path = Path(models_dir) / req.model_path
        if not local_path.exists():
            raise HTTPException(status_code=404, detail=f"Model path not found: {req.model_path}")
        model_path = local_path

    # Estimate size
    estimated = sum(f.stat().st_size for f in model_path.rglob("*.safetensors")) if model_path.is_dir() else 0

    manager.register_model(
        model_id=req.model_id,
        model_path=str(model_path),
        estimated_bytes=estimated,
        pinned=req.pinned,
    )

    return ModelResponse(
        model_id=req.model_id,
        model_type="auto",
        status="registered",
        size_bytes=estimated,
        pinned=req.pinned,
    )


@router.post("/models/load")
async def load_model(req: ModelLoadRequest, request: Request, _=Depends(require_permission("can_load_models"))):
    """Load a registered model into memory."""
    import asyncio
    from yunshu_gateway.engine import get_model_manager, get_engine

    manager = get_model_manager()

    if manager is not None:
        try:
            engine = await manager.get_engine(req.model_id)
            if hasattr(engine, 'is_running') and not engine.is_running:
                await engine.start()
            return {"status": "loaded", "model_id": req.model_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Single-engine mode
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="No engine available")

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, engine.load, req.model_id)
        await engine.start()
        return {"status": "loaded", "model_id": req.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.post("/models/unload")
async def unload_model(req: ModelUnloadRequest, request: Request, _=Depends(require_permission("can_unload_models"))):
    """Unload a model from memory."""
    from yunshu_gateway.engine import get_model_manager

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    entry = manager._entries.get(req.model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")

    if not entry.is_loaded:
        return {"status": "already_unloaded", "model_id": req.model_id}

    if not req.force and hasattr(entry, '_ref_count') and entry._ref_count > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Model has active requests. Use force=true to override.",
        )

    try:
        await manager.unload_model(req.model_id)
        return {"status": "unloaded", "model_id": req.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload: {e}")


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, request: Request, _=Depends(require_permission("can_register_models"))):
    """Remove a model registration (must be unloaded first)."""
    from yunshu_gateway.engine import get_model_manager

    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    entry = manager._entries.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if entry.is_loaded:
        raise HTTPException(status_code=409, detail="Unload the model first")

    del manager._entries[model_id]
    return {"status": "deleted", "model_id": model_id}


# ── Engine Config ──


@router.get("/config/engine")
async def get_engine_config():
    """Get current engine configuration."""
    from yunshu_gateway.engine import get_engine

    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    cfg = engine.config
    return {
        "completion_batch_size": cfg.completion_batch_size,
        "prefill_batch_size": cfg.prefill_batch_size,
        "prefill_step_size": cfg.prefill_step_size,
        "max_kv_size": cfg.max_kv_size,
        "deferred_clear_delay": cfg.deferred_clear_delay,
        "cache_cleanup_interval": cfg.cache_cleanup_interval,
    }


@router.patch("/config/engine")
async def update_engine_config(req: EngineConfigUpdate):
    """Update engine configuration (takes effect on next request cycle)."""
    from yunshu_gateway.engine import get_engine

    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    cfg = engine.config
    updated = []
    for field, value in req.model_dump(exclude_none=True).items():
        if hasattr(cfg, field):
            setattr(cfg, field, value)
            updated.append(field)

    return {"status": "updated", "fields": updated}


# ── Per-Model Settings ──


@router.get("/models/{model_id}/settings")
async def get_model_settings(model_id: str, _=Depends(require_permission("can_view_admin"))):
    """Get per-model runtime settings."""
    from yunshu_engine.model_manager import get_model_manager
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not available")

    entry = manager._entries.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not registered: {model_id}")

    settings_data = {}
    if entry.settings is not None:
        settings_data = entry.settings.to_dict()
    else:
        from yunshu_engine.model_settings import ModelSettings
        settings_data = ModelSettings().to_dict()

    return {
        "model_id": model_id,
        "model_type": getattr(entry, 'model_type', 'unknown'),
        "loaded": entry.is_loaded,
        "pinned": getattr(entry, 'is_pinned', False),
        "settings": settings_data,
    }


@router.patch("/models/{model_id}/settings")
async def update_model_settings(
    model_id: str,
    request: Request,
    _=Depends(require_permission("can_load_models")),
):
    """Update per-model runtime settings (hot-reloadable)."""
    from yunshu_engine.model_manager import get_model_manager
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager not available")

    entry = manager._entries.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not registered: {model_id}")

    if entry.settings is None:
        from yunshu_engine.model_settings import ModelSettings
        entry.settings = ModelSettings()

    overrides = await request.json()
    changed = entry.settings.apply_overrides(overrides)

    return {"status": "updated", "model_id": model_id, "changed_fields": changed}


# ── Auth Tokens (RBAC-backed) ──


@router.post("/tokens", response_model=AuthTokenResponse)
async def create_token(req: AuthTokenCreate, request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """Create a new API auth token (simple)."""
    import secrets

    token = f"ys_{secrets.token_hex(24)}"
    now = datetime.now(tz=None)
    expires = now + timedelta(days=req.expires_days) if req.expires_days else None

    manager = _get_rbac_manager(request)
    import time as _time
    from yunshu_control.role_manager import Role

    expires_at = None
    if req.expires_days:
        expires_at = _time.time() + req.expires_days * 86400

    raw_key, api_key = manager.create_key(
        name=req.name,
        role=Role.USER,
        expires_days=req.expires_days,
    )

    return AuthTokenResponse(
        token=raw_key,
        name=req.name,
        created_at=now,
        expires_at=expires,
    )


@router.get("/tokens")
async def list_tokens(request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """List all API tokens (masked)."""
    manager = _get_rbac_manager(request)
    return manager.list_keys()


@router.delete("/tokens/{token_prefix}")
async def revoke_token(token_prefix: str, request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """Revoke an API token by name or prefix."""
    manager = _get_rbac_manager(request)
    count = manager.revoke_key(token_prefix)
    if count == 0:
        raise HTTPException(status_code=404, detail="Token not found")
    return {"status": "revoked", "count": count}


# ── RBAC Key Management ──


@router.post("/keys", response_model=RBACTokenResponse)
async def create_rbac_key(req: RBACTokenCreate, request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """Create a new RBAC API key with role and SLO class."""
    from yunshu_control.role_manager import Role, SLOClass

    try:
        role = Role[req.role.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {req.role}. Use: admin, developer, user")

    try:
        slo = SLOClass[req.slo_class.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid SLO class: {req.slo_class}. Use: best_effort, standard, premium")

    manager = _get_rbac_manager(request)
    raw_key, api_key = manager.create_key(
        name=req.name,
        role=role,
        slo_class=slo,
        expires_days=req.expires_days,
        requests_per_minute=req.requests_per_minute,
        tokens_per_minute=req.tokens_per_minute,
    )

    return RBACTokenResponse(
        key=raw_key,
        name=api_key.name,
        role=api_key.role.name,
        slo_class=api_key.slo_class.name,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
    )


@router.get("/keys")
async def list_rbac_keys(request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """List all RBAC API keys."""
    manager = _get_rbac_manager(request)
    return manager.list_keys()


@router.delete("/keys/{key_name}")
async def delete_rbac_key(key_name: str, request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """Delete an RBAC API key by name."""
    manager = _get_rbac_manager(request)
    count = manager.delete_key(key_name)
    if count == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "deleted", "count": count}


@router.delete("/keys")
async def delete_rbac_key_by_body(request: Request, _=Depends(require_permission("can_manage_tokens"))):
    """Delete an RBAC API key — body-based variant for WebUI compatibility.

    WebUI sends DELETE /admin/keys with body {key: "..."} instead of
    using the path param /admin/keys/{key_name}. This handler bridges
    the gap.
    """
    body = await request.json()
    key = body.get("key") or body.get("key_name")
    if not key:
        raise HTTPException(status_code=400, detail="Missing 'key' in request body")
    manager = _get_rbac_manager(request)
    count = manager.delete_key(key)
    if count == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "deleted", "count": count}


# ── System Monitoring (oMLX pattern) ──


@router.get("/hardware")
async def get_hardware_status(_=Depends(require_permission("can_view_admin"))):
    """Get hardware detection and MLX optimization status."""
    from yunshu_engine.optimizations import get_optimization_status
    return get_optimization_status()


@router.get("/metrics")
async def get_server_metrics(
    model_id: str = "",
    scope: str = "session",
    _=Depends(require_permission("can_view_admin")),
):
    """Get server-level metrics (oMLX ServerMetrics pattern).

    Query params:
      model_id: per-model breakdown (empty = global)
      scope: "session" (since restart) or "alltime" (persisted)
    """
    from yunshu_engine.server_metrics import get_server_metrics
    return get_server_metrics().get_snapshot(model_id=model_id, scope=scope)


@router.post("/metrics/clear")
async def clear_session_metrics(_=Depends(require_permission("can_manage_tokens"))):
    """Clear session metrics."""
    from yunshu_engine.server_metrics import get_server_metrics
    get_server_metrics().clear_session()
    return {"status": "cleared"}


@router.get("/metrics/prefill")
async def get_prefill_progress(
    model_id: str = "",
    _=Depends(require_permission("can_view_admin")),
):
    """Get live prefill progress for active requests (oMLX pattern)."""
    from yunshu_engine.prefill_progress import get_prefill_tracker
    tracker = get_prefill_tracker()
    if model_id:
        return {"model_id": model_id, "requests": tracker.get_model_progress(model_id)}
    return tracker.get_all_progress()


@router.get("/memory")
async def get_memory_status(_=Depends(require_permission("can_view_admin"))):
    """Get GPU memory status from MLX Metal."""
    from yunshu_engine.memory_monitor import MemoryMonitor
    monitor = MemoryMonitor()
    return monitor.get_stats()


@router.get("/models/discover")
async def discover_models(
    models_dir: str = "models",
    _=Depends(require_permission("can_view_admin")),
):
    """Auto-discover models from disk with modality detection."""
    from yunshu_engine.model_discovery import discover_models

    path = Path(models_dir)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {models_dir}")

    try:
        models = discover_models(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "models_dir": str(path.resolve()),
        "total": len(models),
        "models": {
            mid: {
                "model_type": m.model_type,
                "engine_type": m.engine_type,
                "estimated_size_gb": round(m.estimated_size / 1024 ** 3, 2),
                "config_model_type": m.config_model_type,
            }
            for mid, m in models.items()
        },
    }


@router.get("/registry")
async def get_model_registry(_=Depends(require_permission("can_view_admin"))):
    """Get model ownership registry stats."""
    from yunshu_engine.model_registry import get_registry
    return get_registry().get_stats()


@router.get("/kv-cache")
async def get_kv_cache_stats(_=Depends(require_permission("can_view_admin"))):
    """Get KV prefix cache statistics.

    Returns block pool usage, free blocks, prefix cache hit rate,
    and active block tables count.
    """
    from yunshu_gateway.engine import get_engine, get_model_manager

    # Try BatchedEngine path (EngineCore)
    engine = get_engine()
    if engine is not None and hasattr(engine, "get_kv_cache_stats"):
        return engine.get_kv_cache_stats()

    # Try ModelManager path
    manager = get_model_manager()
    if manager is not None:
        for mid, entry in manager._entries.items():
            if entry.is_loaded and hasattr(entry, "_engine"):
                eng = entry._engine
                if hasattr(eng, "get_kv_cache_stats"):
                    return eng.get_kv_cache_stats()

    return {"enabled": False}


@router.get("/memory-guard")
async def get_memory_guard_stats(_=Depends(require_permission("can_view_admin"))):
    """Get memory guard stats and recommendations.

    Returns guard statistics (total checks, rejections, rates) along
    with current memory info and recommended max tokens for a sample prompt.
    """
    from yunshu_gateway.engine import get_engine

    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    guard = None
    # Check BatchedEngine -> EngineCore -> memory_guard
    if hasattr(engine, '_engine_core') and engine._engine_core is not None:
        guard = getattr(engine._engine_core, '_memory_guard', None)

    if guard is None:
        # Try the older Engine class path
        if hasattr(engine, '_memory_guard'):
            guard = engine._memory_guard

    if guard is None:
        return {
            "enabled": False,
            "message": "MemoryGuard not configured. Call setup_memory_guard() after model load.",
        }

    stats = guard.get_stats()

    # Add recommendations for common prompt lengths
    recommendations = {}
    for prompt_len in [128, 512, 1024, 4096]:
        recommended = guard.get_recommended_max_tokens(prompt_len)
        recommendations[f"prompt_{prompt_len}_tokens"] = recommended

    return {
        "enabled": True,
        "stats": stats,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# WebUI-required endpoints (frontend calls these)
# ---------------------------------------------------------------------------


@router.get("/logs")
async def get_admin_logs(
    level: str = "info",
    lines: int = 100,
    _=Depends(require_permission("can_view_admin")),
):
    """Return recent log entries (WebUI Admin logs tab)."""
    import logging

    logger = logging.getLogger("yunshu")
    if not hasattr(logger, 'recent_logs'):
        return {"logs": [], "total": 0, "level": level}

    logs = getattr(logger, 'recent_logs', [])
    filtered = [l for l in logs if level == "all" or l.get("level", "").lower() == level]
    return {"logs": filtered[-lines:], "total": len(filtered), "level": level}


@router.get("/cache/status")
async def get_cache_status(_=Depends(require_permission("can_view_admin"))):
    """Return KV cache status (WebUI Admin cache tab)."""
    from ..engine import get_engine, get_model_manager

    caches = []
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and hasattr(entry.engine, 'get_kv_cache_stats'):
                try:
                    stats = entry.engine.get_kv_cache_stats()
                    caches.append({"model_id": entry.model_id, "stats": stats})
                except Exception:
                    logger.debug("failed to get KV cache stats for %s", entry.model_id, exc_info=True)
                    caches.append({"model_id": entry.model_id, "stats": {}})
    else:
        engine = get_engine()
        if engine and hasattr(engine, 'get_kv_cache_stats'):
            try:
                caches.append({"model_id": getattr(engine, 'model_name', 'default'), "stats": engine.get_kv_cache_stats()})
            except Exception:
                logger.debug("failed to get KV cache stats for single engine", exc_info=True)

    return {"caches": caches, "total": len(caches)}


@router.post("/cache/clear")
async def clear_cache(_=Depends(require_permission("can_load_models"))):
    """Clear KV caches (WebUI Admin cache tab)."""
    from ..engine import get_engine, get_model_manager

    cleared = 0
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and hasattr(entry.engine, '_kv_prefix_cache'):
                try:
                    entry.engine._kv_prefix_cache.clear()
                    cleared += 1
                except Exception:
                    logger.debug("failed to clear KV cache for %s", entry.model_id, exc_info=True)
    return {"cleared": cleared, "status": "ok"}


@router.get("/queue/stats")
async def get_queue_stats(_=Depends(require_permission("can_view_system"))):
    """Request queue statistics (from yunshu_control.request_queue)."""
    from yunshu_control.request_queue import get_request_queue_manager
    manager = get_request_queue_manager()
    if manager is None:
        return {"enabled": False, "stats": None}
    return {"enabled": True, "stats": manager.get_stats()}


# ── LoRA Adapter Management ──


class LoRALoadRequest(BaseModel):
    model_id: str
    adapter_id: str


class LoRAMergeRequest(BaseModel):
    model_id: str
    adapter_id: str


class LoRARegisterRequest(BaseModel):
    model_id: str
    adapter_id: str
    adapter_path: str


@router.get("/models/{model_id}/adapters")
async def list_lora_adapters(model_id: str, _=Depends(require_permission("can_view_admin"))):
    """List LoRA adapters for a model."""
    engine = _get_engine_for_model(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        return {"enabled": False, "adapters": []}
    return {"enabled": True, **lora_mgr.get_stats()}


@router.post("/models/{model_id}/adapters/load")
async def load_lora_adapter(model_id: str, req: LoRALoadRequest, _=Depends(require_permission("can_manage_models"))):
    """Load a LoRA adapter for a model."""
    engine = _get_engine_for_model(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        raise HTTPException(status_code=400, detail="LoRA not supported for this model")
    success = lora_mgr.load_adapter(req.adapter_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load adapter '{req.adapter_id}'")
    return {"status": "loaded", "adapter_id": req.adapter_id}


@router.post("/models/{model_id}/adapters/unload")
async def unload_lora_adapter(model_id: str, req: LoRALoadRequest, _=Depends(require_permission("can_manage_models"))):
    """Unload a LoRA adapter from a model."""
    engine = _get_engine_for_model(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        raise HTTPException(status_code=400, detail="LoRA not supported for this model")
    success = lora_mgr.unload_adapter(req.adapter_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to unload adapter '{req.adapter_id}'")
    return {"status": "unloaded", "adapter_id": req.adapter_id}


@router.post("/models/{model_id}/adapters/merge")
async def merge_lora_adapter(model_id: str, req: LoRAMergeRequest, _=Depends(require_permission("can_manage_models"))):
    """Merge a LoRA adapter permanently into the base model."""
    engine = _get_engine_for_model(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        raise HTTPException(status_code=400, detail="LoRA not supported for this model")
    success = lora_mgr.merge_adapter(req.adapter_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to merge adapter '{req.adapter_id}'")
    return {"status": "merged", "adapter_id": req.adapter_id}


@router.post("/models/{model_id}/adapters/register")
async def register_lora_adapter(model_id: str, req: LoRARegisterRequest, _=Depends(require_permission("can_manage_models"))):
    """Register a LoRA adapter path for a model."""
    engine = _get_engine_for_model(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        raise HTTPException(status_code=400, detail="LoRA not supported for this model")
    lora_mgr.register_adapter(req.adapter_id, req.adapter_path)
    return {"status": "registered", "adapter_id": req.adapter_id}


@router.get("/radix-tree")
async def get_radix_tree_stats(_=Depends(require_permission("can_view_admin"))):
    """Get RadixTree prefix cache statistics.

    Returns tree size, node count, eviction metrics, and block usage.
    """
    from yunshu_gateway.engine import get_engine, get_model_manager

    engine = get_engine()
    if engine is not None and hasattr(engine, "get_radix_tree_stats"):
        return engine.get_radix_tree_stats()

    manager = get_model_manager()
    if manager is not None:
        for mid, entry in manager._entries.items():
            if entry.is_loaded and hasattr(entry, "_engine"):
                eng = entry._engine
                if hasattr(eng, "get_radix_tree_stats"):
                    return eng.get_radix_tree_stats()

    return {"enabled": False}


@router.get("/hardware-profile")
async def get_hardware_profile(_=Depends(require_permission("can_view_admin"))):
    """Get hardware profile with adaptive defaults.

    Returns chip info, memory, GPU cores, and computed optimal settings.
    """
    try:
        from yunshu_engine.utils.hardware import get_hardware_profile
        return get_hardware_profile()
    except Exception as e:
        return {"error": str(e)}


def _get_engine_for_model(model_id: str):
    from yunshu_gateway.engine import get_engine, get_model_manager
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        if entry is not None and entry.is_loaded and entry.engine is not None:
            return entry.engine
    engine = get_engine()
    if engine and engine.is_loaded:
        return engine
    return None


# ── System Info (SDK endpoint) ──


@router.get("/info")
async def get_system_info(_=Depends(require_permission("can_view_admin"))):
    """Get system information: version, loaded models, uptime, memory."""
    import time as _time
    from yunshu_gateway.engine import get_engine, get_model_manager
    info = {
        "version": "0.1.0",
        "uptime_s": round(_time.monotonic(), 1),
        "loaded_models": [],
        "memory": {},
    }
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            info["loaded_models"].append({
                "model_id": entry.model_id,
                "model_path": entry.model_path,
                "is_loaded": entry.is_loaded,
                "engine_type": entry.engine_type,
            })
        info["memory"] = manager.memory_usage()
    engine = get_engine()
    if engine and engine.is_loaded:
        info["current_model"] = engine.model_name
    return info


@router.get("/config/scheduler")
async def get_scheduler_config(_=Depends(require_permission("can_view_admin"))):
    """Get scheduler configuration."""
    from yunshu_gateway.engine import get_engine
    engine = get_engine()
    if engine is None:
        return {"enabled": False}
    stats = getattr(engine, 'get_stats', lambda: {})()
    return {
        "enabled": True,
        "max_batch_size": stats.get("max_batch_size", 1),
        "max_num_seqs": stats.get("max_num_seqs", 1),
        "use_engine_loop": stats.get("use_engine_loop", False),
    }
