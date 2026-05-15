from __future__ import annotations
"""Yunshu Control Plane — Dashboard and configuration router.

Provides the WebUI dashboard with server configuration,
model status, usage stats, and system info.
"""


import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

logger = logging.getLogger(__name__)


class DashboardConfig(BaseModel):
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    theme: str = "dark"
    refresh_interval_ms: int = 5000
    show_gpu_stats: bool = True
    show_mesh_status: bool = False
    show_kv_cache: bool = False


# In-memory config (Phase 1)
_dashboard_config = DashboardConfig()


@router.get("/config")
async def get_dashboard_config():
    """Get dashboard configuration."""
    return _dashboard_config.model_dump()


@router.put("/config")
async def update_dashboard_config(config: DashboardConfig):
    """Update dashboard configuration."""
    global _dashboard_config
    _dashboard_config = config
    return {"status": "updated", **config.model_dump()}


@router.get("/summary")
async def dashboard_summary(request: Request):
    """Get a complete dashboard summary in one call.

    Combines engine status, model info, GPU memory, and mesh status
    into a single response optimized for dashboard rendering.
    """
    from yunshu_gateway.engine import get_engine, get_model_manager

    result = {
        "engine": None,
        "models": [],
        "gpu": None,
        "mesh": None,
    }

    # Engine status
    engine = get_engine()
    if engine:
        result["engine"] = engine.get_stats()

    # Models
    manager = get_model_manager()
    if manager:
        for mid, entry in manager._entries.items():
            model_info = {
                "id": mid,
                "type": str(entry.model_type.name if hasattr(entry.model_type, 'name') else entry.model_type),
                "loaded": entry.is_loaded,
                "pinned": entry.is_pinned,
                "size_bytes": entry.estimated_bytes,
            }
            if entry.is_loaded and entry.engine and hasattr(entry.engine, "get_stats"):
                model_info["stats"] = entry.engine.get_stats()
            result["models"].append(model_info)
    elif engine and engine.is_loaded:
        result["models"].append({
            "id": engine.model_name or "default",
            "type": "LLM",
            "loaded": True,
        })

    # GPU memory
    try:
        import mlx.core as mx
        result["gpu"] = {
            "active_bytes": mx.get_active_memory(),
            "peak_bytes": mx.get_peak_memory(),
            "cache_bytes": mx.get_cache_memory(),
        }
    except Exception:
        logger.debug("failed to query GPU memory via MLX", exc_info=True)

    # Mesh status
    mesh_manager = getattr(request.app.state, "mesh_manager", None)
    if mesh_manager:
        result["mesh"] = mesh_manager.get_stats()

    # RBAC keys count
    rbac_manager = getattr(request.app.state, "rbac_manager", None)
    result["auth"] = {
        "rbac_enabled": rbac_manager is not None,
        "key_count": len(rbac_manager._keys) if rbac_manager else 0,
    }

    return result


@router.get("/usage")
async def dashboard_usage():
    """Get usage statistics for the dashboard."""
    from yunshu_gateway.middleware.metrics import get_metrics

    metrics = get_metrics()
    return {
        "total_requests": sum(metrics.request_count.values()),
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "inference_count": metrics.inference_count,
        "error_count": metrics.error_count,
    }
