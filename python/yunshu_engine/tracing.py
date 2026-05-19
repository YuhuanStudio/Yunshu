from __future__ import annotations
"""Deep observability: structured tracing, logging, enhanced metrics, health dashboard.

Components:
- InferenceTracer: per-request trace spans with OpenTelemetry-compatible export
- StructuredLogger: JSON-formatted structured log entries with context binding
- MetricsAggregatorV2: enhanced metrics with histogram support and Prometheus output
- HealthDashboard: aggregated system health with 0-100 scoring

Thread-safe throughout. No external dependencies beyond stdlib.
"""


import io
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# InferenceTracer — structured tracing for inference requests
# ---------------------------------------------------------------------------


class SpanKind(str, Enum):
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


@dataclass
class Span:
    """A single span within a trace."""

    span_id: str
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "UNSET"  # UNSET, OK, ERROR
    parent_span_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "spanId": self.span_id,
            "name": self.name,
            "kind": self.kind.value,
            "startTimeUnixNano": int(self.start_time * 1e9) if self.start_time else 0,
            "endTimeUnixNano": int(self.end_time * 1e9) if self.end_time else 0,
            "attributes": {k: v for k, v in self.attributes.items()},
            "status": {"code": self.status},
            "parentSpanId": self.parent_span_id,
        }


@dataclass
class Trace:
    """A full trace for one inference request."""

    trace_id: str
    start_time: float
    end_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    spans: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0

    @property
    def span_count(self) -> int:
        return len(self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "traceId": self.trace_id,
            "startTimeUnixNano": int(self.start_time * 1e9),
            "endTimeUnixNano": int(self.end_time * 1e9) if self.end_time else 0,
            "durationMs": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "result": self.result,
            "spans": [s.to_dict() for s in self.spans],
        }


class InferenceTracer:
    """Structured tracing for inference requests.

    Creates a trace span for each request with:
    - Request metadata (model, params, timestamps)
    - Prefill duration
    - Per-token decode timing (ITL)
    - Spec decode events (draft/verify/accept)
    - KV cache events (hit/miss/evict)
    - Memory pressure events
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self._lock = threading.Lock()
        self._traces: dict[str, Trace] = {}
        self._completed: list[Trace] = []
        self._max_traces = max_traces

    def start_trace(
        self, request_id: str, metadata: Optional[dict[str, Any]] = None
    ) -> Trace:
        """Create a new trace for an inference request."""
        trace = Trace(
            trace_id=request_id or uuid.uuid4().hex,
            start_time=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            self._traces[trace.trace_id] = trace
        return trace

    def span(
        self,
        request_id: str,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Optional[Span]:
        """Add a span to an active trace."""
        with self._lock:
            trace = self._traces.get(request_id)
        if trace is None:
            return None

        s = Span(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {},
        )
        trace.spans.append(s)
        return s

    def end_span(self, request_id: str, span_name: str) -> None:
        """End a named span within a trace."""
        with self._lock:
            trace = self._traces.get(request_id)
        if trace is None:
            return
        for s in reversed(trace.spans):
            if s.name == span_name and s.end_time == 0.0:
                s.end_time = time.time()
                return

    def end_trace(
        self, request_id: str, result: Optional[dict[str, Any]] = None
    ) -> Optional[Trace]:
        """Complete a trace and move it to the completed buffer."""
        with self._lock:
            trace = self._traces.pop(request_id, None)
        if trace is None:
            return None

        trace.end_time = time.time()
        trace.result = result

        # End any still-open spans
        for s in trace.spans:
            if s.end_time == 0.0:
                s.end_time = trace.end_time

        with self._lock:
            self._completed.append(trace)
            # Evict oldest if over limit
            while len(self._completed) > self._max_traces:
                self._completed.pop(0)

        return trace

    def get_trace(self, request_id: str) -> Optional[Trace]:
        """Retrieve a trace (active or completed)."""
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is not None:
                return trace
            for t in reversed(self._completed):
                if t.trace_id == request_id:
                    return t
        return None

    def export_traces(self, fmt: str = "json") -> str:
        """Export completed traces in OpenTelemetry-compatible format."""
        with self._lock:
            all_traces = list(self._completed)

        if fmt == "json":
            payload = {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [
                                {"key": "service.name", "value": {"stringValue": "yunshu-engine"}},
                            ]
                        },
                        "scopeSpans": [
                            {
                                "scope": {"name": "yunshu.inference"},
                                "spans": [
                                    s.to_dict()
                                    for t in all_traces
                                    for s in t.spans
                                ],
                            }
                        ],
                    }
                ],
                "traces": [t.to_dict() for t in all_traces],
            }
            return json.dumps(payload, indent=2, default=str)
        raise ValueError(f"Unsupported export format: {fmt}")

    def get_stats(self) -> dict[str, Any]:
        """Return tracer statistics."""
        with self._lock:
            active_count = len(self._traces)
            completed_count = len(self._completed)
            all_completed = list(self._completed)

        avg_spans = 0.0
        avg_duration = 0.0
        span_types: dict[str, int] = defaultdict(int)

        if all_completed:
            total_spans = sum(t.span_count for t in all_completed)
            avg_spans = round(total_spans / len(all_completed), 2)
            avg_duration = round(
                sum(t.duration_ms for t in all_completed) / len(all_completed), 2
            )
            for t in all_completed:
                for s in t.spans:
                    span_types[s.name] += 1

        return {
            "active_traces": active_count,
            "completed_traces": completed_count,
            "avg_spans_per_trace": avg_spans,
            "avg_trace_duration_ms": avg_duration,
            "span_types": dict(span_types),
        }

    def clear(self) -> None:
        """Clear all traces (for testing)."""
        with self._lock:
            self._traces.clear()
            self._completed.clear()


# ---------------------------------------------------------------------------
# StructuredLogger — JSON-formatted structured log entries
# ---------------------------------------------------------------------------


class LogLevel(int, Enum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


class StructuredLogger:
    """Structured JSON logger with context binding.

    Replaces ad-hoc logger.debug() calls with structured log entries
    that have consistent field names and JSON formatting.
    """

    def __init__(
        self,
        name: str = "yunshu",
        level: LogLevel = LogLevel.DEBUG,
        output: Optional[io.StringIO] = None,
    ) -> None:
        self._name = name
        self._level = level
        self._output = output
        self._lock = threading.Lock()
        self._context: dict[str, Any] = {}
        # Stats tracking
        self._log_counts: dict[str, int] = defaultdict(int)
        self._event_counts: dict[str, int] = defaultdict(int)
        self._total_entries = 0

    def _should_log(self, level: LogLevel) -> bool:
        return level >= self._level

    def _write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, default=str, ensure_ascii=False)
        if self._output is not None:
            self._output.write(line + "\n")
        else:
            # Fallback to standard logging
            logger = logging.getLogger(self._name)
            level = entry.get("level", "INFO")
            msg = entry.get("event", "unknown")
            getattr(logger, level.lower(), logger.info)(msg, extra=entry)

    def log(self, event: str, **kwargs: Any) -> None:
        """Emit a structured log entry."""
        level_str = kwargs.pop("level", "INFO")
        level = LogLevel[level_str] if isinstance(level_str, str) else level_str

        with self._lock:
            self._log_counts[level_str.upper() if isinstance(level_str, str) else LogLevel(level).name] += 1
            self._event_counts[event] += 1
            self._total_entries += 1

        if not self._should_log(level):
            return

        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "logger": self._name,
            "level": level_str.upper() if isinstance(level_str, str) else LogLevel(level).name,
            "event": event,
        }

        # Merge bound context
        with self._lock:
            entry.update(self._context)

        # Merge call-specific fields
        entry.update(kwargs)

        self._write(entry)

    def bind_context(self, **kwargs: Any) -> None:
        """Bind context fields that will be included in all subsequent logs."""
        with self._lock:
            self._context.update(kwargs)

    def unbind_context(self, *keys: str) -> None:
        """Remove previously bound context fields."""
        with self._lock:
            for key in keys:
                self._context.pop(key, None)

    def trace(self, event: str, **kwargs: Any) -> None:
        self.log(event, level="TRACE", **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self.log(event, level="DEBUG", **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self.log(event, level="INFO", **kwargs)

    def warn(self, event: str, **kwargs: Any) -> None:
        self.log(event, level="WARN", **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self.log(event, level="ERROR", **kwargs)

    def get_stats(self) -> dict[str, Any]:
        """Return logger statistics."""
        with self._lock:
            return {
                "name": self._name,
                "level": self._level.name,
                "total_entries": self._total_entries,
                "log_counts_by_level": dict(self._log_counts),
                "event_types": dict(self._event_counts),
                "bound_context_keys": list(self._context.keys()),
            }

    def clear_stats(self) -> None:
        """Reset stats (for testing)."""
        with self._lock:
            self._log_counts.clear()
            self._event_counts.clear()
            self._total_entries = 0
            self._context.clear()


# ---------------------------------------------------------------------------
# MetricsAggregatorV2 — enhanced metrics with histogram support
# ---------------------------------------------------------------------------


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


# Pre-allocated bucket boundaries for latency histograms
DEFAULT_LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
    0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)


class MetricsAggregatorV2:
    """Enhanced metrics aggregator with counter, gauge, and histogram support.

    Outputs in Prometheus exposition format. Thread-safe throughout.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Metric storage: name -> {labels_key -> value}
        self._counters: dict[str, dict[frozenset, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[frozenset, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Histogram: name -> {labels_key -> (bucket_def, observations)}
        self._histograms: dict[str, dict[frozenset, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._histogram_buckets: dict[str, tuple[float, ...]] = {}
        # Help text registry
        self._help: dict[str, str] = {}
        # Scrape tracking
        self._scrape_count = 0
        self._last_scrape_duration_ms = 0.0

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        help_text: str = "",
        buckets: Optional[tuple[float, ...]] = None,
    ) -> None:
        """Pre-register a metric with help text and optional histogram buckets."""
        with self._lock:
            self._help[name] = help_text
            if metric_type == MetricType.HISTOGRAM and buckets:
                self._histogram_buckets[name] = buckets

    def counter(
        self, name: str, labels: Optional[dict[str, str]] = None, value: float = 1.0
    ) -> None:
        """Increment a counter metric."""
        key = frozenset((labels or {}).items())
        with self._lock:
            self._counters[name][key] += value

    def gauge(
        self, name: str, labels: Optional[dict[str, str]] = None, value: float = 0.0
    ) -> None:
        """Set a gauge metric."""
        key = frozenset((labels or {}).items())
        with self._lock:
            self._gauges[name][key] = value

    def histogram(
        self, name: str, labels: Optional[dict[str, str]] = None, value: float = 0.0
    ) -> None:
        """Record a histogram observation."""
        key = frozenset((labels or {}).items())
        with self._lock:
            self._histograms[name][key].append(value)
            # Cap observations per label set
            lst = self._histograms[name][key]
            if len(lst) > 100_000:
                self._histograms[name][key] = lst[-50_000:]

    def _format_labels(self, key: frozenset) -> str:
        if not key:
            return ""

        def _esc(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        pairs = ",".join(
            f'{k}="{_esc(v)}"' for k, v in sorted(key)
        )
        return f"{{{pairs}}}"

    def get_prometheus_output(self) -> str:
        """Generate Prometheus exposition format output."""
        t0 = time.monotonic()
        lines: list[str] = []

        with self._lock:
            # Counters
            for name in sorted(self._counters):
                help_text = self._help.get(name, "")
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for key in sorted(self._counters[name], key=lambda k: sorted(k)):
                    val = self._counters[name][key]
                    label_str = self._format_labels(key)
                    lines.append(f"{name}{label_str} {val}")
                lines.append("")

            # Gauges
            for name in sorted(self._gauges):
                help_text = self._help.get(name, "")
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                for key in sorted(self._gauges[name], key=lambda k: sorted(k)):
                    val = self._gauges[name][key]
                    label_str = self._format_labels(key)
                    lines.append(f"{name}{label_str} {val}")
                lines.append("")

            # Histograms
            for name in sorted(self._histograms):
                help_text = self._help.get(name, "")
                buckets = self._histogram_buckets.get(
                    name, DEFAULT_LATENCY_BUCKETS
                )
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} histogram")
                for key in sorted(self._histograms[name], key=lambda k: sorted(k)):
                    values = self._histograms[name][key]
                    if not values:
                        continue
                    label_str = self._format_labels(key)
                    count = len(values)
                    total = sum(values)
                    # Inject le= into existing labels
                    extra = label_str[1:-1] if label_str else ""
                    for upper in buckets:
                        in_bucket = sum(1 for v in values if v <= upper)
                        if extra:
                            lines.append(
                                f'{name}_bucket{{le="{upper}",{extra}}} {in_bucket}'
                            )
                        else:
                            lines.append(
                                f'{name}_bucket{{le="{upper}"}} {in_bucket}'
                            )
                    if extra:
                        lines.append(
                            f'{name}_bucket{{le="+Inf",{extra}}} {count}'
                        )
                    else:
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    lines.append(f"{name}_sum{label_str} {total}")
                    lines.append(f"{name}_count{label_str} {count}")
                lines.append("")

        self._scrape_count += 1
        self._last_scrape_duration_ms = (time.monotonic() - t0) * 1000.0

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregator statistics."""
        with self._lock:
            total_metrics = len(self._counters) + len(self._gauges) + len(self._histograms)
            total_series = (
                sum(len(v) for v in self._counters.values())
                + sum(len(v) for v in self._gauges.values())
                + sum(len(v) for v in self._histograms.values())
            )
        return {
            "total_metrics": total_metrics,
            "total_series": total_series,
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
            "scrape_count": self._scrape_count,
            "last_scrape_duration_ms": round(self._last_scrape_duration_ms, 3),
        }


# ---------------------------------------------------------------------------
# HealthDashboard — aggregated system health with 0-100 scoring
# ---------------------------------------------------------------------------


class HealthDashboard:
    """Aggregates all system health information and computes a 0-100 score.

    Score components (weighted):
    - System resources (CPU, memory, GPU): 30%
    - Model status (loaded, responding): 25%
    - Request health (error rate, latency): 20%
    - Memory guard (pressure level): 15%
    - KV cache (hit rate, utilisation): 10%
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_collect_time: float = 0.0
        self._last_report: dict[str, Any] = {}

    def collect(self) -> dict[str, Any]:
        """Gather health data from all subsystems."""
        report: dict[str, Any] = {
            "timestamp": time.time(),
            "system": self._collect_system(),
            "models": self._collect_models(),
            "requests": self._collect_requests(),
            "memory_guard": self._collect_memory_guard(),
            "kv_cache": self._collect_kv_cache(),
            "spec_decode": self._collect_spec_decode(),
        }
        report["health_score"] = self.compute_health_score(report)

        with self._lock:
            self._last_collect_time = time.monotonic()
            self._last_report = report

        return report

    def compute_health_score(self, report: Optional[dict[str, Any]] = None) -> int:
        """Compute a 0-100 health score based on all metrics."""
        if report is None:
            with self._lock:
                report = self._last_report
        if not report:
            return 50  # Unknown state

        score = 0.0

        # System resources (30 points)
        system = report.get("system", {})
        cpu_pct = system.get("cpu_percent", 0)
        mem_pct = system.get("memory_percent", 0)
        gpu_util = system.get("gpu_utilization_pct", 0)

        # CPU: full score if under 80%, degrades after
        cpu_score = max(0, 10 * (1 - max(0, cpu_pct - 80) / 20))
        # Memory: full score if under 85%, degrades after
        mem_score = max(0, 10 * (1 - max(0, mem_pct - 85) / 15))
        # GPU: moderate usage is good (not 0, not maxed)
        if gpu_util > 0:
            gpu_score = min(10, max(0, 10 - abs(gpu_util - 50) / 5))
        else:
            gpu_score = 5  # Neutral if no GPU info
        score += cpu_score + mem_score + gpu_score

        # Model status (25 points)
        models = report.get("models", {})
        total = models.get("total", 0)
        loaded = models.get("loaded", 0)
        if total > 0:
            model_ratio = loaded / total
            score += 25 * model_ratio
        else:
            score += 12.5  # Neutral when no models

        # Request health (20 points)
        requests = report.get("requests", {})
        error_rate = requests.get("error_rate", 0)
        avg_latency_ms = requests.get("avg_latency_ms", 0)
        # Error rate penalty
        error_score = max(0, 10 * (1 - error_rate * 10))
        # Latency: full score if under 2s
        latency_score = max(0, 10 * (1 - avg_latency_ms / 5000))
        score += error_score + latency_score

        # Memory guard (15 points)
        mg = report.get("memory_guard", {})
        if mg.get("active", False):
            pressure = mg.get("max_pressure", "normal")
            if pressure == "normal":
                score += 15
            elif pressure == "warning":
                score += 10
            elif pressure == "critical":
                score += 3
            else:
                score += 7
        else:
            score += 12  # Neutral when no guard

        # KV cache (10 points)
        kv = report.get("kv_cache", {})
        if kv.get("active", False):
            hit_rate = kv.get("avg_hit_rate", 0)
            score += 10 * hit_rate
        else:
            score += 5  # Neutral

        return max(0, min(100, round(score)))

    def get_report(self) -> dict[str, Any]:
        """Return the latest health report, collecting if stale."""
        with self._lock:
            age = time.monotonic() - self._last_collect_time
            if age > 30 or not self._last_report:
                # Will collect outside lock
                pass
            else:
                return dict(self._last_report)

        return self.collect()

    # -- Subsystem collectors --

    def _collect_system(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        try:
            import psutil
            data["cpu_percent"] = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            data["memory_percent"] = mem.percent
            data["memory_available_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            data["cpu_percent"] = 0
            data["memory_percent"] = 0
            data["memory_available_gb"] = 0

        try:
            import mlx.core as mx
            active = mx.get_active_memory()
            data["gpu_active_bytes"] = active
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            )
            total = int(r.stdout.strip()) if r.returncode == 0 else 0
            data["gpu_utilization_pct"] = round(active / total * 100, 1) if total else 0
        except Exception:
            logger.debug("gpu stats collection failed", exc_info=True)
            data["gpu_active_bytes"] = 0
            data["gpu_utilization_pct"] = 0

        return data

    def _collect_models(self) -> dict[str, Any]:
        data: dict[str, Any] = {"total": 0, "loaded": 0, "details": []}
        try:
            from yunshu_gateway.engine import get_engine, get_model_manager
            manager = get_model_manager()
            if manager is not None:
                for entry in manager.list_entries():
                    data["total"] += 1
                    if entry.is_loaded:
                        data["loaded"] += 1
                    data["details"].append({
                        "model_id": entry.model_id,
                        "loaded": entry.is_loaded,
                    })
            else:
                engine = get_engine()
                if engine and getattr(engine, "is_loaded", False):
                    data["total"] = 1
                    data["loaded"] = 1
                    data["details"].append({
                        "model_id": getattr(engine, "model_name", "default"),
                        "loaded": True,
                    })
        except Exception:
            logger.debug("operation failed", exc_info=True)
        return data

    def _collect_requests(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "active": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
        }
        try:
            from yunshu_gateway.middleware.metrics_aggregator import get_metrics_aggregator
            agg = get_metrics_aggregator()
            summary = agg.get_summary(window_seconds=60)
            data["active"] = summary.get("total_requests", 0)
            data["error_rate"] = summary.get("error_rate", 0)
            data["avg_latency_ms"] = summary.get("avg_duration_ms", 0)
        except Exception:
            logger.debug("metrics collection failed", exc_info=True)
        return data

    def _collect_memory_guard(self) -> dict[str, Any]:
        data: dict[str, Any] = {"active": False}
        try:
            from yunshu_gateway.engine import get_engine, get_model_manager
            manager = get_model_manager()
            engines = []
            if manager is not None:
                for entry in manager.list_entries():
                    if entry.is_loaded and entry.engine:
                        engines.append((entry.model_id, entry.engine))
            else:
                engine = get_engine()
                if engine:
                    engines.append(("default", engine))

            max_pressure = "normal"
            for model_id, engine in engines:
                guard = getattr(engine, "_memory_guard", None)
                if guard is None:
                    core = getattr(engine, "_engine_core", None)
                    if core:
                        guard = getattr(core, "_memory_guard", None)
                if guard is not None:
                    data["active"] = True
                    pl = getattr(guard, "pressure_level", "normal")
                    pressure_order = {"normal": 0, "warning": 1, "critical": 2}
                    if pressure_order.get(pl, 0) > pressure_order.get(max_pressure, 0):
                        max_pressure = pl
                    data["eviction_count"] = data.get("eviction_count", 0) + getattr(
                        guard, "_eviction_count", 0
                    )
            data["max_pressure"] = max_pressure
        except Exception:
            logger.debug("operation failed", exc_info=True)
        return data

    def _collect_kv_cache(self) -> dict[str, Any]:
        data: dict[str, Any] = {"active": False}
        try:
            from yunshu_gateway.engine import get_engine, get_model_manager
            from yunshu_engine.batched_engine import BatchedEngine

            manager = get_model_manager()
            caches = []
            if manager is not None:
                for entry in manager.list_entries():
                    if entry.is_loaded and isinstance(
                        getattr(entry, "engine", None), BatchedEngine
                    ):
                        try:
                            stats = entry.engine.get_kv_cache_stats()
                            caches.append(stats)
                        except Exception:
                            logger.debug("operation failed", exc_info=True)
            else:
                engine = get_engine()
                if engine and isinstance(engine, BatchedEngine):
                    try:
                        caches.append(engine.get_kv_cache_stats())
                    except Exception:
                        logger.debug("operation failed", exc_info=True)

            if caches:
                data["active"] = True
                hits = sum(c.get("hits", 0) for c in caches)
                misses = sum(c.get("misses", 0) for c in caches)
                total = hits + misses
                data["avg_hit_rate"] = hits / total if total > 0 else 0
                data["total_caches"] = len(caches)
        except Exception:
            logger.debug("operation failed", exc_info=True)
        return data

    def _collect_spec_decode(self) -> dict[str, Any]:
        data: dict[str, Any] = {"active": False}
        try:
            from yunshu_gateway.engine import get_model_manager
            from yunshu_engine.batched_engine import BatchedEngine

            manager = get_model_manager()
            if manager is not None:
                for entry in manager.list_entries():
                    if entry.is_loaded and isinstance(
                        getattr(entry, "engine", None), BatchedEngine
                    ):
                        ngram = getattr(entry.engine, "_ngram_proposer", None)
                        if ngram is not None:
                            data["active"] = True
                            data["ngram_stats"] = getattr(
                                entry.engine, "_ngram_stats", {}
                            )
        except Exception:
            logger.debug("operation failed", exc_info=True)
        return data


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_tracer: Optional[InferenceTracer] = None
_structured_logger: Optional[StructuredLogger] = None
_metrics_v2: Optional[MetricsAggregatorV2] = None
_health_dashboard: Optional[HealthDashboard] = None
_singleton_lock = threading.Lock()


def get_inference_tracer() -> InferenceTracer:
    global _tracer
    if _tracer is None:
        with _singleton_lock:
            if _tracer is None:
                _tracer = InferenceTracer()
    return _tracer


def get_structured_logger() -> StructuredLogger:
    global _structured_logger
    if _structured_logger is None:
        with _singleton_lock:
            if _structured_logger is None:
                _structured_logger = StructuredLogger()
    return _structured_logger


def get_metrics_v2() -> MetricsAggregatorV2:
    global _metrics_v2
    if _metrics_v2 is None:
        with _singleton_lock:
            if _metrics_v2 is None:
                _metrics_v2 = MetricsAggregatorV2()
    return _metrics_v2


def get_health_dashboard() -> HealthDashboard:
    global _health_dashboard
    if _health_dashboard is None:
        with _singleton_lock:
            if _health_dashboard is None:
                _health_dashboard = HealthDashboard()
    return _health_dashboard


def reset_tracing() -> None:
    """Reset all singletons (for testing)."""
    global _tracer, _structured_logger, _metrics_v2, _health_dashboard
    with _singleton_lock:
        _tracer = None
        _structured_logger = None
        _metrics_v2 = None
        _health_dashboard = None
