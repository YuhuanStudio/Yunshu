"""Yunshu Control Plane — Monitoring router.

System stats, GPU memory, request metrics, health checks.
Supports both single-engine and multi-model mode.
"""


import logging
import platform

from fastapi import APIRouter, HTTPException

from ..schemas.models import (
    EngineStatsResponse,
    GPUMemoryStats,
    RequestStatsResponse,
    SystemStatsResponse,
)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

logger = logging.getLogger(__name__)


def _get_gpu_stats() -> GPUMemoryStats:
    """Get GPU memory stats from MLX."""
    import mlx.core as mx

    active = mx.get_active_memory()
    peak = mx.get_peak_memory()
    cache = mx.get_cache_memory()
    total_uma = 0
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        total_uma = int(result.stdout.strip())
    except Exception:
        logger.debug("failed to query hw.memsize via sysctl", exc_info=True)

    return GPUMemoryStats(
        total_bytes=total_uma,
        active_bytes=active,
        peak_bytes=peak,
        cache_bytes=cache,
        available_bytes=max(0, total_uma - active),
        utilization_pct=(active / total_uma * 100) if total_uma > 0 else 0.0,
    )


def _aggregate_model_stats() -> dict:
    """Aggregate stats from all loaded engines in multi-model mode."""
    from yunshu_gateway.engine import get_model_manager

    manager = get_model_manager()
    if manager is None:
        return None

    total_active = 0
    total_waiting = 0
    total_processed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    step_counter = 0
    model_names = []
    loaded_count = 0
    running_count = 0

    for entry in manager.list_entries():
        if entry.is_loaded:
            loaded_count += 1
            model_names.append(entry.model_id)
            if entry.engine is not None:
                running_count += 1
                if hasattr(entry.engine, "get_stats"):
                    stats = entry.engine.get_stats()
                    total_active += stats.get("active_collectors",
                                              stats.get("scheduler_running",
                                                        stats.get("active", 0)))
                    total_waiting += stats.get("scheduler_waiting",
                                               stats.get("waiting", 0))
                    total_processed += stats.get("num_requests_processed", 0)
                    total_prompt_tokens += stats.get("scheduler_total_prompt_tokens",
                                                     stats.get("total_prompt_tokens", 0))
                    total_completion_tokens += stats.get("scheduler_total_completion_tokens",
                                                         stats.get("total_completion_tokens", 0))
                    step_counter = max(step_counter,
                                       stats.get("scheduler_step_counter",
                                                 stats.get("step_counter", 0)))

    return {
        "model": ", ".join(model_names) if model_names else None,
        "loaded": loaded_count > 0,
        "running": running_count > 0,
        "active_requests": total_active,
        "waiting_requests": total_waiting,
        "step_counter": step_counter,
        "requests_processed": total_processed,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "models_loaded": loaded_count,
        "models_registered": len(list(manager.list_entries())),
    }


@router.get("/system", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system-level statistics (CPU, memory, GPU)."""
    import mlx.core as mx

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_total = mem.total
        mem_used = mem.used
        mem_avail = mem.available
    except ImportError:
        cpu_pct = 0.0
        mem_total = 0
        mem_used = 0
        mem_avail = 0

    gpu = _get_gpu_stats()

    try:
        mlx_version = mx.__version__
    except AttributeError:
        mlx_version = "unknown"

    return SystemStatsResponse(
        cpu_percent=cpu_pct,
        memory_total_bytes=mem_total,
        memory_used_bytes=mem_used,
        memory_available_bytes=mem_avail,
        gpu=gpu,
        uptime_seconds=0.0,
        python_version=platform.python_version(),
        mlx_version=mlx_version,
    )


@router.get("/engine", response_model=EngineStatsResponse)
async def get_engine_stats():
    """Get engine-level statistics (works in both single and multi-model mode)."""
    from yunshu_gateway.engine import get_engine, get_model_manager

    # Try multi-model mode first
    manager = get_model_manager()
    if manager is not None:
        agg = _aggregate_model_stats()
        if agg is not None and agg["loaded"]:
            return EngineStatsResponse(
                model=agg["model"],
                loaded=agg["loaded"],
                running=agg["running"],
                active_requests=agg["active_requests"],
                waiting_requests=agg["waiting_requests"],
                step_counter=agg["step_counter"],
                requests_processed=agg["requests_processed"],
                total_prompt_tokens=agg["total_prompt_tokens"],
                total_completion_tokens=agg["total_completion_tokens"],
                uptime_seconds=0.0,
                gpu_memory=_get_gpu_stats(),
            )

    # Single-engine fallback
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    stats = engine.get_stats()

    return EngineStatsResponse(
        model=stats.get("model"),
        loaded=stats.get("loaded", False),
        running=stats.get("running", False),
        active_requests=stats.get("active", 0),
        waiting_requests=stats.get("waiting", 0),
        step_counter=stats.get("step_counter", 0),
        requests_processed=stats.get("num_requests_processed", 0),
        total_prompt_tokens=stats.get("total_prompt_tokens", 0),
        total_completion_tokens=stats.get("total_completion_tokens", 0),
        uptime_seconds=stats.get("uptime_seconds", 0.0),
        gpu_memory=_get_gpu_stats(),
    )


@router.get("/requests", response_model=RequestStatsResponse)
async def get_request_stats():
    """Get request-level statistics (aggregated across all models in multi-model mode)."""
    from yunshu_gateway.engine import get_engine, get_model_manager

    total_req = 0
    active_req = 0
    total_comp = 0
    uptime = 1.0

    # Multi-model mode
    manager = get_model_manager()
    if manager is not None:
        agg = _aggregate_model_stats()
        if agg is not None:
            total_req = agg["requests_processed"]
            active_req = agg["active_requests"]
            total_comp = agg["total_completion_tokens"]
    else:
        # Single-engine fallback
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        stats = engine.get_stats()
        uptime = stats.get("uptime_seconds", 1.0)
        total_req = stats.get("num_requests_processed", 0)
        active_req = stats.get("active", 0)
        total_comp = stats.get("total_completion_tokens", 0)

    return RequestStatsResponse(
        total_requests=total_req,
        active_requests=active_req,
        avg_latency_ms=0.0,
        p50_latency_ms=0.0,
        p95_latency_ms=0.0,
        p99_latency_ms=0.0,
        tokens_per_second=total_comp / uptime if uptime > 0 else 0.0,
        requests_per_second=total_req / uptime if uptime > 0 else 0.0,
    )


@router.get("/models")
async def get_model_stats():
    """Get per-model statistics (multi-model mode)."""
    from yunshu_engine.model_manager import ModelType
    from yunshu_gateway.engine import get_model_manager, get_engine

    manager = get_model_manager()
    results = []

    if manager is not None:
        for entry in manager.list_entries():
            engine_stats = {}
            if entry.is_loaded and entry.engine:
                if hasattr(entry.engine, "get_stats"):
                    engine_stats = entry.engine.get_stats()
            results.append({
                "model_id": entry.model_id,
                "type": entry.model_type.name if isinstance(entry.model_type, ModelType) else str(entry.model_type),
                "loaded": entry.is_loaded,
                "pinned": entry.is_pinned,
                "loading": entry.is_loading,
                "size_bytes": entry.estimated_bytes,
                "last_access": entry.last_access,
                "error": entry.load_error,
                "stats": engine_stats,
            })
    else:
        engine = get_engine()
        if engine and engine.is_loaded:
            results.append({
                "model_id": engine.model_name,
                "type": "LLM",
                "loaded": True,
                "pinned": False,
                "size_bytes": 0,
                "stats": engine.get_stats(),
            })

    return {"models": results, "total": len(results)}
