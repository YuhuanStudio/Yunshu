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


def _collect_engines(default_engine, model_manager) -> list[tuple[str, Any]]:
    """Collect all loaded engines from model_manager, falling back to default engine."""
    engines = []
    if model_manager is not None:
        for entry in model_manager.list_entries():
            if entry.is_loaded and hasattr(entry, 'engine') and entry.engine is not None:
                engines.append((entry.model_id, entry.engine))
    if not engines and default_engine is not None:
        engines.append(("default", default_engine))
    return engines


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
        logger.debug(f"sysctl {name} read failed", exc_info=True)
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
        logger.debug(f"gpu info collection failed: {e}", exc_info=True)
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
        logger.debug("failed", exc_info=True)

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
                    logger.debug("failed", exc_info=True)
                try:
                    from ..middleware.metrics import get_metrics
                    metrics = get_metrics()
                    if hasattr(metrics, '_server_metrics') and metrics._server_metrics:
                        itl = metrics._server_metrics.get_itl_stats()
                        pm.set_gauge("itl_p50_ms", itl.get("itl_p50_ms", 0))
                        pm.set_gauge("itl_p99_ms", itl.get("itl_p99_ms", 0))
                except Exception:
                    logger.debug("failed", exc_info=True)

                # KV cache block gauges (paged KV from engine_core)
                try:
                    kv_stats = entry.engine.get_kv_cache_stats()
                    paged = kv_stats.get("paged_kv", {})
                    if paged.get("enabled"):
                        pm.set_gauge("kv_cache_blocks_used", paged.get("used_blocks", 0))
                        pm.set_gauge("kv_cache_blocks_total", paged.get("total_blocks", 0))
                    # KV prefix cache gauges
                    prefix = kv_stats.get("prefix_cache", {})
                    if prefix:
                        pm.set_gauge("kv_prefix_cache_entries", prefix.get("entries", 0))
                        pm.set_gauge("kv_prefix_cache_hits", prefix.get("hits", 0))
                        pm.set_gauge("kv_prefix_cache_misses", prefix.get("misses", 0))
                except Exception:
                    logger.debug("KV cache gauge population failed", exc_info=True)

                # RadixTree gauges
                try:
                    radix_stats = entry.engine.get_radix_tree_stats()
                    if radix_stats.get("enabled"):
                        pm.set_gauge("radix_total_nodes", radix_stats.get("total_nodes", 0))
                        pm.set_gauge("radix_total_tokens", radix_stats.get("total_tokens", 0))
                        evictions = radix_stats.get("evictions", {})
                        pm.set_gauge("radix_evictions_lru", evictions.get("lru", 0))
                        pm.set_gauge("radix_evictions_lfu", evictions.get("lfu", 0))
                        pm.set_gauge("radix_evictions_fifo", evictions.get("fifo", 0))
                        pm.set_gauge("radix_evictions_freed_blocks", radix_stats.get("freed_blocks", 0))
                except Exception:
                    logger.debug("RadixTree gauge population failed", exc_info=True)

                # MON-2/4/5: Scheduler monitoring gauges from engine_core
                try:
                    core = entry.engine._engine_core
                    if core is not None:
                        pm.set_gauge("scheduler_waiting_queue_depth", getattr(core, '_last_queue_depth', 0))
                        pm.set_gauge("scheduler_batch_size", getattr(core, '_last_batch_size', 0))
                        pm.set_gauge("compute_utilization_pct", core.get_compute_utilization())
                        pm.set_gauge("step_duration_ms", getattr(core, '_last_step_wall_ms', 0.0))
                except Exception:
                    logger.debug("scheduler monitoring gauge population failed", exc_info=True)

                # H2O attention eviction gauges
                try:
                    core = entry.engine._engine_core
                    if core is not None:
                        tracker = getattr(core.scheduler, '_attention_score_tracker', None)
                        if tracker is not None:
                            at_stats = tracker.get_stats()
                            pm.set_gauge("attention_eviction_tracked_requests", at_stats.get("tracked_requests", 0))
                            pm.set_gauge("attention_eviction_total_blocks", at_stats.get("total_blocks", 0))
                except Exception:
                    logger.debug("attention eviction gauge population failed", exc_info=True)

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
                    logger.debug(f"kv cache stats unavailable for {entry.model_id}", exc_info=True)
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
    """Memory guard pressure statistics across all loaded engines."""
    from ..engine import get_engine, get_model_manager

    engines = _collect_engines(get_engine(), get_model_manager())
    results = []
    for model_id, engine in engines:
        guard = getattr(engine, '_memory_guard', None)
        if guard is None:
            core = getattr(engine, '_engine_core', None)
            if core:
                guard = getattr(core, '_memory_guard', None)
        if guard is not None:
            results.append({
                "model_id": model_id,
                "pressure_level": getattr(guard, 'pressure_level', 'unknown'),
                "eviction_count": getattr(guard, '_eviction_count', 0),
            })
    if not results:
        return {"active": False}
    return {"active": True, "models": results}


@router.get("/ssd-cache")
async def ssd_cache_stats() -> dict[str, Any]:
    """SSD KV cache statistics across all loaded engines."""
    from ..engine import get_engine, get_model_manager

    engines = _collect_engines(get_engine(), get_model_manager())
    results = []
    for model_id, engine in engines:
        cache = getattr(engine, '_kv_prefix_cache', None)
        if cache is not None:
            ssd = getattr(cache, '_ssd_store', None)
            if ssd is not None:
                stats = getattr(ssd, 'get_stats', lambda: {})()
                results.append({"model_id": model_id, **stats})
    if not results:
        return {"active": False}
    return {"active": True, "models": results}


@router.get("/data-parallel")
async def data_parallel_stats() -> dict[str, Any]:
    """DataParallelRouter statistics — load distribution across replicas.

    When the DP middleware is active, includes health checking and per-node
    latency histograms from DPLoadBalancer. Falls back to raw router stats.
    """
    try:
        from ..dp_middleware import get_dp_load_balancer
        lb = get_dp_load_balancer()
        if lb is not None:
            return lb.get_stats()
    except Exception:
        logger.debug("dp_middleware unavailable", exc_info=True)

    from ..engine import get_dp_router

    router = get_dp_router()
    if router is None:
        return {"active": False}
    return {"active": True, **router.get_stats()}


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
    """Thinking segment KV substore statistics across all loaded engines."""
    from ..engine import get_engine, get_model_manager

    engines = _collect_engines(get_engine(), get_model_manager())
    results = []
    for model_id, engine in engines:
        store = getattr(engine, '_thinking_store', None)
        if store is not None:
            results.append({"model_id": model_id, **store.get_stats()})
    if not results:
        return {"active": False}
    return {"active": True, "models": results}


@router.get("/metal-kernels")
async def metal_kernel_stats() -> dict[str, Any]:
    """Metal kernel manager status across all loaded engines.

    Reports kernel compilation status, available kernels, and per-engine
    availability. Enabled via YUNSHU_METAL_KERNELS=1 environment variable.
    """
    from ..engine import get_engine, get_model_manager

    engines = _collect_engines(get_engine(), get_model_manager())
    results = []
    for model_id, engine in engines:
        mgr = getattr(engine, '_metal_kernel_manager', None)
        entry = {"model_id": model_id, "enabled": mgr is not None}
        if mgr is not None:
            try:
                from yunshu_engine.metal_kernels import get_compilation_status
                entry.update(get_compilation_status())
            except Exception:
                logger.debug("metal kernel status failed", exc_info=True)
        # Also pull from engine stats (which includes scheduler-level info)
        stats = getattr(engine, 'get_stats', lambda: {})()
        if "metal_kernels" in stats:
            entry["scheduler"] = stats["metal_kernels"]
        results.append(entry)
    if not results:
        return {"active": False, "env_hint": "Set YUNSHU_METAL_KERNELS=1 to enable"}
    return {"active": True, "models": results}


@router.get("/ane-embeddings")
async def ane_embedding_stats() -> dict[str, Any]:
    """ANE embedding co-processor status.

    Reports ANE availability, CoreML model compilation status, inference
    count, and average latency. Enabled via YUNSHU_ANE_EMBEDDINGS=1.
    """
    from yunshu_engine.ane_embedding import get_ane_embedding_stats
    return get_ane_embedding_stats()


@router.get("/external-prefill")
async def external_prefill_stats() -> dict[str, Any]:
    """External prefill server/client statistics.

    Reports disaggregated prefill status when YUNSHU_EXTERNAL_PREFILL=1 is set.
    Includes server stats (requests served, avg prefill time, bytes transferred)
    and client stats (requests sent, avg latency, success rate).
    """
    from yunshu_engine.external_prefill import get_external_prefill_stats
    return get_external_prefill_stats()


@router.get("/health-dashboard")
async def health_dashboard() -> dict[str, Any]:
    """Aggregated health dashboard with 0-100 scoring.

    Collects system resources, model status, request health,
    memory guard, KV cache, and spec decode into a single
    health score report.
    """
    from yunshu_engine.tracing import get_health_dashboard
    dashboard = get_health_dashboard()
    return dashboard.get_report()


@router.get("/reasoning-tokens")
async def reasoning_tokens_stats() -> dict[str, Any]:
    """Reasoning token usage across all engines.

    Reports thinking/reasoning token counts from BatchedEngine
    and VLM engine stats.
    """
    from ..engine import get_engine, get_model_manager
    stats = {"engines": []}
    total_reasoning = 0

    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and entry.engine and hasattr(entry.engine, 'get_stats'):
                s = entry.engine.get_stats()
                rt = s.get("reasoning_tokens", 0)
                if rt > 0:
                    total_reasoning += rt
                    stats["engines"].append({
                        "model_id": entry.model_id,
                        "reasoning_tokens": rt,
                    })

    stats["total_reasoning_tokens"] = total_reasoning
    return stats


@router.get("/response-cache")
async def response_cache_stats() -> dict[str, Any]:
    """Response cache hit/miss statistics.

    Shows per-engine response cache metrics (hits, misses) and
    the underlying ResponseCache module stats.
    """
    from ..engine import get_engine, get_model_manager
    stats = {"engines": []}

    manager = get_model_manager()
    if manager is not None:
        for entry in manager.list_entries():
            if entry.is_loaded and entry.engine and hasattr(entry.engine, 'get_stats'):
                s = entry.engine.get_stats()
                rc = s.get("response_cache")
                if rc and (rc.get("hits", 0) > 0 or rc.get("misses", 0) > 0):
                    stats["engines"].append({
                        "model_id": entry.model_id,
                        "response_cache": rc,
                    })

    # Underlying cache module stats
    try:
        from yunshu_engine.gateway_optimizer import get_response_cache
        cache = get_response_cache()
        stats["cache_module"] = cache.get_stats()
    except Exception:
        logger.debug("operation failed", exc_info=True)
        stats["cache_module"] = {"enabled": False}

    return stats


@router.get("/inflight-prefix-sharing")
async def inflight_prefix_sharing_stats() -> dict[str, Any]:
    """Inflight prefix sharing statistics (SGLang cache_unfinished_req pattern).

    Tracks how many concurrent requests share KV prefix blocks during
    prefill, reducing redundant computation for shared system prompts.
    """
    try:
        from yunshu_engine.inflight_prefix_sharing import get_inflight_tracker
        return get_inflight_tracker().get_stats()
    except Exception:
        logger.debug("operation failed", exc_info=True)
        return {"enabled": False}


@router.get("/request-coalescer")
async def request_coalescer_stats() -> dict[str, Any]:
    """Request coalescing statistics (batch simultaneous requests for same model).

    Shows how many requests were batched together within the coalescing
    window (default 5ms), reducing per-request overhead.
    """
    try:
        from yunshu_engine.gateway_optimizer import get_request_coalescer
        coalescer = get_request_coalescer()
        return coalescer.get_stats()
    except Exception:
        logger.debug("operation failed", exc_info=True)
        return {"enabled": False}


@router.get("/token-scheduler")
async def token_scheduler_stats() -> dict[str, Any]:
    """Token-level scheduler statistics (WFQ token budget + priority inversion).

    Shows token budget allocation stats, fairness metrics, and priority
    inversion detection/resolution counts.
    """
    try:
        from yunshu_engine.engine_core import EngineCore
        # Access the singleton engine core's scheduler stats
        engine = None
        try:
            from ..engine import get_engine
            engine = get_engine()
        except Exception:
            pass
        if engine is not None and hasattr(engine, '_engine_core'):
            core = engine._engine_core
            if core is not None:
                return {
                    "token_scheduler": core._token_scheduler.get_stats(),
                    "priority_inversion": core._priority_guard.get_stats(),
                    "fairness": core._fairness_tracker.get_stats(),
                }
        return {"enabled": False, "reason": "engine_core not active"}
    except Exception:
        logger.debug("operation failed", exc_info=True)
        return {"enabled": False}


@router.get("/kv-migration")
async def kv_migration_stats() -> dict[str, Any]:
    """KV migration statistics (multi-tier GPU/CPU/SSD block management).

    Shows how many KV blocks have been migrated between tiers,
    per-tier capacity usage, and temperature distribution.
    """
    try:
        from yunshu_engine.engine_core import EngineCore
        engine = None
        try:
            from ..engine import get_engine
            engine = get_engine()
        except Exception:
            pass
        if engine is not None and hasattr(engine, '_engine_core'):
            core = engine._engine_core
            if core is not None:
                return core._kv_migration.get_stats()
        return {"enabled": False, "reason": "engine_core not active"}
    except Exception:
        logger.debug("operation failed", exc_info=True)
        return {"enabled": False}


@router.get("/attention-eviction")
async def attention_eviction_stats() -> dict[str, Any]:
    """H2O-style attention-score-based KV eviction statistics.

    Shows how many requests are being tracked, total blocks scored,
    and heuristic vs real attention weight update counts.
    """
    try:
        engines = _collect_engines()
        results = []
        for model_id, engine in engines:
            if hasattr(engine, '_engine_core') and engine._engine_core is not None:
                scheduler = engine._engine_core.scheduler
                tracker = getattr(scheduler, '_attention_score_tracker', None)
                if tracker is not None:
                    results.append({"model_id": model_id, **tracker.get_stats()})
                else:
                    results.append({"model_id": model_id, "enabled": False})
        if not results:
            return {"enabled": False, "reason": "no engines with attention eviction"}
        return {"models": results}
    except Exception:
        logger.debug("attention eviction stats failed", exc_info=True)
        return {"enabled": False}


@router.get("/batch-size")
async def batch_size_stats() -> dict[str, Any]:
    """Batch size distribution statistics from server metrics.

    Shows batch size percentiles (p50, p99) for scheduler steps.
    """
    try:
        from ..middleware.metrics import get_metrics
        metrics = get_metrics()
        if hasattr(metrics, '_server_metrics') and metrics._server_metrics is not None:
            return metrics._server_metrics.get_batch_size_stats()
        return {"enabled": False, "reason": "server_metrics not active"}
    except Exception:
        logger.debug("batch size stats failed", exc_info=True)
        return {"enabled": False}
