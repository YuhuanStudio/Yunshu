"""Yunshu Gateway — Monitoring router.

L1 gateway monitoring endpoints: system stats, model status, active requests.
These are distinct from the L2 control-plane monitoring endpoints in
yunshu_api/routers/monitoring.py — these are for real-time gateway
observability by operators and Prometheus scraping.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

from ..middleware.metrics_aggregator import get_metrics_aggregator
from ..middleware.prometheus_exporter import get_prometheus_metrics

router = APIRouter(prefix="/gw/monitoring", tags=["monitoring"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sysctl(name: str) -> Optional[int]:
    """Read a macOS sysctl integer value, or None on failure."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2
        )
        return int(r.stdout.strip())
    except Exception:
        return None


def _get_cpu_info() -> dict[str, Any]:
    """Return CPU usage percent and core count."""
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        cores = psutil.cpu_count(logical=True)
        phys_cores = psutil.cpu_count(logical=False)
    except ImportError:
        cpu_pct = 0.0
        phys_cores = _sysctl("hw.physicalcpu") or 0
        cores = _sysctl("hw.logicalcpu") or phys_cores
    return {"percent": cpu_pct, "physical_cores": phys_cores, "logical_cores": cores}


def _get_memory_info() -> dict[str, Any]:
    """Return RAM usage stats."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_bytes": mem.total,
            "used_bytes": mem.used,
            "available_bytes": mem.available,
            "percent": mem.percent,
        }
    except ImportError:
        total = _sysctl("hw.memsize") or 0
        return {
            "total_bytes": total,
            "used_bytes": 0,
            "available_bytes": 0,
            "percent": 0.0,
            "note": "psutil not available — install for detailed memory stats",
        }


def _get_gpu_info() -> dict[str, Any]:
    """Return Apple GPU / UMA memory info via MLX."""
    try:
        import mlx.core as mx
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache = mx.get_cache_memory()
        total = _sysctl("hw.memsize") or 0
        util = (active / total * 100) if total > 0 else 0.0
        mlx_version = getattr(mx, "__version__", "unknown")
        return {
            "active_bytes": active,
            "peak_bytes": peak,
            "cache_bytes": cache,
            "total_uma_bytes": total,
            "utilization_pct": round(util, 1),
            "mlx_version": mlx_version,
        }
    except Exception as e:
        return {"error": str(e)}


def _get_model_status() -> list[dict[str, Any]]:
    """Return status list of loaded/registered models."""
    from ..engine import get_engine, get_model_manager

    results: list[dict[str, Any]] = []

    manager = get_model_manager()
    if manager is not None:
        for mid, entry in ((e.model_id, e) for e in manager.list_entries()):
            info: dict[str, Any] = {
                "model_id": mid,
                "loaded": entry.is_loaded,
                "pinned": entry.is_pinned,
                "size_bytes": entry.estimated_bytes,
            }
            if entry.is_loaded and entry.engine and hasattr(entry.engine, "get_stats"):
                try:
                    info["stats"] = entry.engine.get_stats()
                except Exception:
                    logger.debug(f"failed to get stats for {entry.model_id}", exc_info=True)
            results.append(info)
        return results

    engine = get_engine()
    if engine and engine.is_loaded:
        results.append({
            "model_id": engine.model_name,
            "loaded": True,
            "stats": engine.get_stats() if hasattr(engine, "get_stats") else {},
        })
    return results


def _get_active_requests() -> dict[str, Any]:
    """Return active request statistics."""
    from ..engine import get_engine, get_model_manager
    from ..middleware.metrics import get_metrics

    active = 0
    waiting = 0
    processed = 0

    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and entry.engine and hasattr(entry.engine, "get_stats"):
                s = entry.engine.get_stats()
                active += s.get("active", 0)
                waiting += s.get("waiting", 0)
                processed += s.get("num_requests_processed", 0)
    else:
        engine = get_engine()
        if engine and hasattr(engine, "get_stats"):
            s = engine.get_stats()
            active = s.get("active", 0)
            waiting = s.get("waiting", 0)
            processed = s.get("num_requests_processed", 0)

    # Also include metrics aggregator data.
    agg = get_metrics_aggregator()
    summary = agg.get_summary(window_seconds=60)

    return {
        "active": active,
        "waiting": waiting,
        "total_processed": processed,
        "last_minute": summary,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/system")
async def system_stats() -> dict[str, Any]:
    """System-level statistics: CPU, memory, GPU, runtime info."""
    return {
        "cpu": _get_cpu_info(),
        "memory": _get_memory_info(),
        "gpu": _get_gpu_info(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "hostname": platform.node(),
    }


@router.get("/models")
async def models_status() -> dict[str, Any]:
    """Model status list (loaded, stats, etc.)."""
    models = _get_model_status()
    return {
        "models": models,
        "total": len(models),
    }


@router.get("/requests")
async def requests_stats(
    window: int = Query(60, ge=1, le=3600, description="Aggregation window in seconds"),
) -> dict[str, Any]:
    """Active and recent request statistics."""
    data = _get_active_requests()
    # Add aggregator percentiles.
    agg = get_metrics_aggregator()
    data["latency_percentiles"] = agg.get_percentiles("duration_ms", window_seconds=window)
    data["token_percentiles"] = agg.get_percentiles("tokens_out", window_seconds=window)
    data["endpoint_breakdown"] = agg.get_endpoint_breakdown(window_seconds=window)

    # ITL stats from ServerMetrics (batch-path ITL tracking)
    try:
        from ..middleware.metrics import get_metrics
        metrics = get_metrics()
        if hasattr(metrics, '_server_metrics') and metrics._server_metrics is not None:
            data["itl"] = metrics._server_metrics.get_itl_stats()
    except Exception:
        pass

    return data


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_export() -> str:
    """Full Prometheus exposition-format output with live engine stats."""
    pm = get_prometheus_metrics()

    # Refresh spec decode stats into Prometheus gauges
    from ..engine import get_model_manager
    from yunshu_engine.batched_engine import BatchedEngine
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), BatchedEngine):
                ngram_stats = getattr(entry.engine, '_ngram_stats', {})
                pm.set_gauge("spec_ngram_proposals", ngram_stats.get("proposals", 0))
                pm.set_gauge("spec_ngram_accepted", ngram_stats.get("accepted", 0))
                pm.set_gauge("spec_ngram_draft", ngram_stats.get("total_draft", 0))
                pm.set_gauge("spec_enabled",
                    1 if (entry.engine._spec_enabled or entry.engine._ngram_proposer is not None) else 0)

                # ITL stats from ServerMetrics
                try:
                    core = entry.engine._engine_core
                    if core and hasattr(core, '_memory_guard') and core._memory_guard:
                        sm = core._memory_guard._monitor if hasattr(core._memory_guard, '_monitor') else None
                except Exception:
                    pass
                try:
                    from ..middleware.metrics import get_metrics
                    metrics = get_metrics()
                    if hasattr(metrics, '_server_metrics') and metrics._server_metrics:
                        itl = metrics._server_metrics.get_itl_stats()
                        pm.set_gauge("itl_p50_ms", itl.get("itl_p50_ms", 0))
                        pm.set_gauge("itl_p99_ms", itl.get("itl_p99_ms", 0))
                except Exception:
                    pass

    return pm.generate()


@router.get("/kv-cache")
async def kv_cache_stats() -> dict[str, Any]:
    """KV prefix cache statistics."""
    from ..engine import get_engine, get_model_manager
    from yunshu_engine.batched_engine import BatchedEngine

    caches = []
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), BatchedEngine):
                try:
                    stats = entry.engine.get_kv_cache_stats()
                    caches.append({"model_id": entry.model_id, **stats})
                except Exception:
                    caches.append({"model_id": entry.model_id, "error": "unavailable"})
    else:
        engine = get_engine()
        if engine and isinstance(engine, BatchedEngine):
            try:
                caches.append({"model_id": engine.model_name, **engine.get_kv_cache_stats()})
            except Exception:
                logger.debug("kv cache stats unavailable", exc_info=True)
    return {"caches": caches}


@router.get("/spec-decode")
async def spec_decode_stats() -> dict[str, Any]:
    """Speculative decoding statistics."""
    from ..engine import get_engine, get_model_manager
    from yunshu_engine.batched_engine import BatchedEngine

    results = []
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), BatchedEngine):
                decoder = getattr(entry.engine, '_spec_decoder', None)
                ngram = getattr(entry.engine, '_ngram_proposer', None)
                mtp = getattr(entry.engine, '_mtp_decoder', None)
                info = {
                    "model_id": entry.model_id,
                    "spec_enabled": entry.engine._spec_enabled,
                    "ngram_enabled": ngram is not None,
                    "mtp_enabled": mtp is not None,
                }
                if decoder is not None:
                    info["spec_stats"] = getattr(decoder, '_stats', {})
                if ngram is not None:
                    info["ngram_stats"] = getattr(entry.engine, '_ngram_stats', {})
                if mtp is not None:
                    s = mtp.stats
                    info["mtp_stats"] = {
                        "accepts": s.accepts,
                        "rejects": s.rejects,
                        "cooldowns": s.cooldowns,
                        "tokens_generated": s.tokens_generated,
                        "total_cycles": s.total_cycles,
                        "acceptance_rate": (
                            s.accepts / s.total_cycles if s.total_cycles > 0 else 0.0
                        ),
                    }
                # Adaptive spec stats
                adaptive_spec = getattr(entry.engine, '_adaptive_spec', None)
                if adaptive_spec is not None:
                    info["adaptive_spec"] = adaptive_spec.get_stats()
                results.append(info)
    return {"models": results}


@router.get("/radix-tree")
async def radix_tree_stats() -> dict[str, Any]:
    """Radix tree statistics for KV cache prefix matching."""
    from ..engine import get_engine, get_model_manager
    from yunshu_engine.batched_engine import BatchedEngine

    results = []
    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), BatchedEngine):
                try:
                    stats = entry.engine.get_radix_tree_stats()
                    results.append({"model_id": entry.model_id, **stats})
                except Exception:
                    logger.debug("radix tree stats unavailable", exc_info=True)
    return {"models": results}


@router.get("/prefill-progress")
async def prefill_progress() -> dict[str, Any]:
    """Prefill progress tracking."""
    from yunshu_engine.prefill_progress import get_prefill_tracker
    tracker = get_prefill_tracker()
    if tracker is None or tracker.active_count == 0:
        return {"active": False}
    return {"active": True, "requests": tracker.get_all_progress()}


@router.get("/memory-guard")
async def memory_guard_stats() -> dict[str, Any]:
    """Memory guard pressure statistics."""
    from ..engine import get_engine
    engine = get_engine()
    if engine is None:
        return {"active": False}

    guard = getattr(engine, '_memory_guard', None)
    if guard is None:
        # Try BatchedEngine's engine_core
        core = getattr(engine, '_engine_core', None)
        if core:
            guard = getattr(core, '_memory_guard', None)

    if guard is None:
        return {"active": False}

    return {
        "active": True,
        "pressure_level": getattr(guard, 'pressure_level', 'unknown'),
        "eviction_count": getattr(guard, '_eviction_count', 0),
    }


@router.get("/ssd-cache")
async def ssd_cache_stats() -> dict[str, Any]:
    """SSD KV cache statistics."""
    from ..engine import get_engine
    engine = get_engine()
    if engine is None:
        return {"active": False}

    cache = getattr(engine, '_kv_prefix_cache', None)
    if cache is None:
        return {"active": False}

    ssd = getattr(cache, '_ssd_store', None)
    if ssd is None:
        return {"active": False, "entries": 0}

    stats = getattr(ssd, 'get_stats', lambda: {})()
    return {"active": True, **stats}


@router.get("/per-model")
async def per_model_stats() -> dict[str, Any]:
    """Per-model request statistics."""
    from yunshu_engine.server_metrics import get_server_metrics
    metrics = get_server_metrics()
    model_ids = list(getattr(metrics, '_per_model', {}).keys())
    result = {}
    for mid in model_ids:
        result[mid] = metrics.get_snapshot(model_id=mid)
    result["_summary"] = metrics.get_snapshot()
    return result


@router.get("/thinking-segments")
async def thinking_segment_stats() -> dict[str, Any]:
    """Thinking segment KV substore statistics."""
    from ..engine import get_engine
    engine = get_engine()
    if engine is None:
        return {"active": False}

    store = getattr(engine, '_thinking_store', None)
    if store is None:
        return {"active": False}

    return {"active": True, **store.get_stats()}
