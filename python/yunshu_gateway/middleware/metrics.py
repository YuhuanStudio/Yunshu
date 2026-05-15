"""Yunshu Gateway — Prometheus-compatible metrics middleware.

Tracks request counts, latencies, token throughput, and error rates.
Exposes /metrics endpoint for Prometheus scraping.

Metrics follow the OpenAI/vLLM pattern:
  - yunshu_request_count{method,endpoint,status}
  - yunshu_request_latency_seconds{endpoint}
  - yunshu_tokens_total{type}  (prompt/completion)
  - yunshu_inference_count
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class _Metrics:
    """Thread-safe metrics store."""

    request_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    request_latency: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inference_count: int = 0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock)

    def record_request(self, endpoint: str, method: str, status: int, latency: float) -> None:
        key = f"{method}:{endpoint}:{status}"
        with self._lock:
            self.request_count[key] += 1
            self.request_latency[endpoint].append(latency)
            if len(self.request_latency[endpoint]) > 1000:
                self.request_latency[endpoint] = self.request_latency[endpoint][-500:]

    def record_tokens(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    def record_inference(self) -> None:
        with self._lock:
            self.inference_count += 1

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def to_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        uptime = time.time() - self.start_time

        lines.append("# HELP yunshu_uptime_seconds Server uptime in seconds")
        lines.append("# TYPE yunshu_uptime_seconds gauge")
        lines.append(f"yunshu_uptime_seconds {uptime:.1f}")

        lines.append("")
        lines.append("# HELP yunshu_request_count Total requests")
        lines.append("# TYPE yunshu_request_count counter")
        with self._lock:
            for key, count in sorted(self.request_count.items()):
                parts = key.split(":")
                if len(parts) == 3:
                    lines.append(
                        f'yunshu_request_count{{method="{parts[0]}",endpoint="{parts[1]}",status="{parts[2]}"}} {count}'
                    )

        lines.append("")
        lines.append("# HELP yunshu_request_latency_seconds Request latency")
        lines.append("# TYPE yunshu_request_latency_seconds summary")
        with self._lock:
            for endpoint, latencies in sorted(self.request_latency.items()):
                if latencies:
                    sorted_lat = sorted(latencies)
                    avg = sum(latencies) / len(latencies)
                    p50 = sorted_lat[len(sorted_lat) // 2]
                    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
                    lines.append(
                        f'yunshu_request_latency_seconds{{endpoint="{endpoint}",quantile="0.5"}} {p50:.4f}'
                    )
                    lines.append(
                        f'yunshu_request_latency_seconds{{endpoint="{endpoint}",quantile="0.99"}} {p99:.4f}'
                    )
                    lines.append(
                        f'yunshu_request_latency_seconds_avg{{endpoint="{endpoint}"}} {avg:.4f}'
                    )
                    lines.append(
                        f'yunshu_request_latency_seconds_count{{endpoint="{endpoint}"}} {len(latencies)}'
                    )

        lines.append("")
        lines.append("# HELP yunshu_tokens_total Token counts")
        lines.append("# TYPE yunshu_tokens_total counter")
        lines.append(f'yunshu_tokens_total{{type="prompt"}} {self.prompt_tokens}')
        lines.append(f'yunshu_tokens_total{{type="completion"}} {self.completion_tokens}')

        lines.append("")
        lines.append("# HELP yunshu_inference_count Total inference operations")
        lines.append("# TYPE yunshu_inference_count counter")
        lines.append(f"yunshu_inference_count {self.inference_count}")

        lines.append("")
        lines.append("# HELP yunshu_error_count Total errors")
        lines.append("# TYPE yunshu_error_count counter")
        lines.append(f"yunshu_error_count {self.error_count}")

        # GPU memory gauges
        try:
            import mlx.core as mx
            active = mx.get_active_memory()
            peak = mx.get_peak_memory()
            cache = mx.get_cache_memory()
            lines.append("")
            lines.append("# HELP yunshu_gpu_memory_bytes GPU memory usage")
            lines.append("# TYPE yunshu_gpu_memory_bytes gauge")
            lines.append(f'yunshu_gpu_memory_bytes{{type="active"}} {active}')
            lines.append(f'yunshu_gpu_memory_bytes{{type="peak"}} {peak}')
            lines.append(f'yunshu_gpu_memory_bytes{{type="cache"}} {cache}')
        except Exception:
            logger.debug("GPU memory stats unavailable", exc_info=True)

        # Engine stats (if available)
        try:
            from yunshu_gateway.engine import get_engine, get_model_manager
            manager = get_model_manager()
            total_active = 0
            total_waiting = 0
            total_running = 0
            total_registered = 0
            if manager is not None:
                for entry in manager.list_entries():
                    total_registered += 1
                    if entry.is_loaded:
                        total_running += 1
                        if entry.engine and hasattr(entry.engine, "get_stats"):
                            s = entry.engine.get_stats()
                            total_active += s.get("active", 0)
                            total_waiting += s.get("waiting", 0)
            else:
                engine = get_engine()
                if engine and hasattr(engine, "get_stats"):
                    s = engine.get_stats()
                    total_active = s.get("active", 0)
                    total_waiting = s.get("waiting", 0)
                    if s.get("loaded"):
                        total_running = 1

            lines.append("")
            lines.append("# HELP yunshu_engine_requests Engine request gauges")
            lines.append("# TYPE yunshu_engine_requests gauge")
            lines.append(f'yunshu_engine_requests{{state="active"}} {total_active}')
            lines.append(f'yunshu_engine_requests{{state="waiting"}} {total_waiting}')
            lines.append(f'yunshu_engine_models{{state="running"}} {total_running}')
            lines.append(f'yunshu_engine_models{{state="registered"}} {total_registered}')
        except Exception:
            logger.debug("engine stats unavailable", exc_info=True)

        return "\n".join(lines) + "\n"


# Module-level metrics singleton
_metrics = _Metrics()


def get_metrics() -> _Metrics:
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect request metrics for Prometheus."""

    async def dispatch(self, request: Request, call_next):
        # Serve metrics endpoint
        if request.url.path == "/metrics":
            parts = [_metrics.to_prometheus()]
            # Append Prometheus exporter gauges (engine-level metrics)
            try:
                from .prometheus_exporter import get_prometheus_metrics
                pm = get_prometheus_metrics()
                # Collect RadixTree eviction metrics from loaded engines
                try:
                    from ..engine import get_engine, get_model_manager
                    from yunshu_engine.batched_engine import BatchedEngine
                    engines = []
                    engine = get_engine()
                    if engine and engine.is_loaded:
                        engines.append(engine)
                    manager = get_model_manager()
                    if manager:
                        for entry in manager.list_entries():
                            if entry.is_loaded and entry.engine:
                                engines.append(entry.engine)
                    for eng in engines:
                        if isinstance(eng, BatchedEngine):
                            radix_stats = eng.get_radix_tree_stats()
                            ev = radix_stats.get("eviction_stats", {})
                            pm.set_gauge("radix_evictions_lru", ev.get("lru", 0))
                            pm.set_gauge("radix_evictions_lfu", ev.get("lfu", 0))
                            pm.set_gauge("radix_evictions_fifo", ev.get("fifo", 0))
                            pm.set_gauge("radix_evictions_freed_blocks", ev.get("total_freed_blocks", 0))
                            pm.set_gauge("radix_total_nodes", radix_stats.get("total_nodes", 0))
                            pm.set_gauge("radix_total_tokens", radix_stats.get("total_tokens", 0))
                except Exception:
                    pass
                pm_text = pm.generate()
                if pm_text:
                    parts.append(pm_text)
            except Exception:
                pass
            return Response(
                content="\n".join(parts),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        t0 = time.monotonic()
        response = await call_next(request)
        latency = time.monotonic() - t0

        _metrics.record_request(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            latency=latency,
        )

        # Also record into the time-windowed aggregator and the
        # Prometheus exporter for richer observability.
        try:
            from .metrics_aggregator import get_metrics_aggregator
            get_metrics_aggregator().record_request(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=latency * 1000,
            )
        except Exception:
            logger.debug("metrics aggregator recording failed", exc_info=True)

        try:
            from .prometheus_exporter import get_prometheus_metrics
            pm = get_prometheus_metrics()
            pm.inc_counter("request_total", {
                "method": request.method,
                "status": str(response.status_code),
                "endpoint": request.url.path,
            })
            pm.observe_histogram("request_duration_seconds", latency, {
                "endpoint": request.url.path,
            })
        except Exception:
            logger.debug("prometheus recording failed", exc_info=True)

        return response
