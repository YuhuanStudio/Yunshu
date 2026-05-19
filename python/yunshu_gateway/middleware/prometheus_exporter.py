from __future__ import annotations
"""Yunshu Gateway — Prometheus-compatible metrics exporter.

Provides a richer Prometheus exposition format beyond what the basic
MetricsMiddleware produces.  This module defines typed counters and
histograms that can be incremented from anywhere in the gateway stack
and serialised to the standard Prometheus text exposition format.

Predefined metrics
------------------
- request_total           (counter)   — total HTTP requests
- request_duration_seconds (histogram) — request latency
- tokens_generated_total  (counter)   — completion tokens served
- active_requests         (gauge)     — currently in-flight requests
- inference_duration_seconds (histogram) — per-inference latency
- kv_cache_blocks_used    (gauge)     — KV cache blocks in use
- kv_cache_blocks_total   (gauge)     — KV cache blocks allocated
"""


import time
from collections import defaultdict
from threading import Lock
from typing import Optional


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

class _Counter:
    """Thread-safe labelled counter."""

    __slots__ = ("_name", "_help", "_lock", "_values")

    def __init__(self, name: str, help_text: str) -> None:
        self._name = name
        self._help = help_text
        self._lock = Lock()
        # key = frozenset of label pairs, value = int
        self._values: dict[frozenset[tuple[str, str]], int] = defaultdict(int)

    def inc(self, labels: Optional[dict[str, str]] = None, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("counter increment must be non-negative")
        key = frozenset((labels or {}).items())
        with self._lock:
            self._values[key] += amount

    def get(self, labels: Optional[dict[str, str]] = None) -> int:
        key = frozenset((labels or {}).items())
        with self._lock:
            return self._values.get(key, 0)

    def format(self) -> str:
        lines: list[str] = []
        lines.append(f"# HELP {self._name} {self._help}")
        lines.append(f"# TYPE {self._name} counter")
        with self._lock:
            for key in sorted(self._values, key=_label_sort_key):
                value = self._values[key]
                label_str = _format_labels(key)
                lines.append(f"{self._name}{label_str} {value}")
        return "\n".join(lines)


class _Gauge:
    """Thread-safe labelled gauge."""

    __slots__ = ("_name", "_help", "_lock", "_values")

    def __init__(self, name: str, help_text: str) -> None:
        self._name = name
        self._help = help_text
        self._lock = Lock()
        self._values: dict[frozenset[tuple[str, str]], float] = defaultdict(float)

    def set(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        key = frozenset((labels or {}).items())
        with self._lock:
            self._values[key] = value

    def inc(self, labels: Optional[dict[str, str]] = None, amount: float = 1.0) -> None:
        key = frozenset((labels or {}).items())
        with self._lock:
            self._values[key] += amount

    def dec(self, labels: Optional[dict[str, str]] = None, amount: float = 1.0) -> None:
        self.inc(labels, -amount)

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        key = frozenset((labels or {}).items())
        with self._lock:
            return self._values.get(key, 0.0)

    def format(self) -> str:
        lines: list[str] = []
        lines.append(f"# HELP {self._name} {self._help}")
        lines.append(f"# TYPE {self._name} gauge")
        with self._lock:
            for key in sorted(self._values, key=_label_sort_key):
                value = self._values[key]
                label_str = _format_labels(key)
                lines.append(f"{self._name}{label_str} {value}")
        return "\n".join(lines)


class _Histogram:
    """Thread-safe labelled histogram.

    Stores observations and maintains running bucket counters so that
    ``format()`` is O(B) instead of O(N*B) per label series.
    """

    __slots__ = (
        "_name", "_help", "_lock", "_observations",
        "_bucket_counts", "_sums", "_counts", "_buckets",
    )

    # Default Prometheus-style exponential buckets (seconds).
    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
        1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    )
    # Inference-specific buckets for sub-second latency (TTFT, ITL).
    INFERENCE_BUCKETS = (
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0,
    )

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Optional[tuple[float, ...]] = None,
    ) -> None:
        self._name = name
        self._help = help_text
        self._lock = Lock()
        self._buckets = buckets or self.DEFAULT_BUCKETS
        # key = frozenset of label pairs
        self._observations: dict[frozenset[tuple[str, str]], list[float]] = defaultdict(list)
        # Running bucket counters — updated incrementally in observe().
        # Each value is a list aligned with self._buckets.
        self._bucket_counts: dict[frozenset[tuple[str, str]], list[int]] = defaultdict(lambda: [0] * len(self._buckets))
        self._sums: dict[frozenset[tuple[str, str]], float] = defaultdict(float)
        self._counts: dict[frozenset[tuple[str, str]], int] = defaultdict(int)

    def observe(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        key = frozenset((labels or {}).items())
        with self._lock:
            lst = self._observations[key]
            lst.append(value)
            # Increment running counters.
            self._sums[key] += value
            self._counts[key] += 1
            bc = self._bucket_counts[key]
            for i, upper in enumerate(self._buckets):
                if value <= upper:
                    bc[i] += 1
            # Cap per-label-series to prevent unbounded growth.
            if len(lst) > 100_000:
                self._observations[key] = lst[-50_000:]
                # Recompute bucket_counts from remaining observations.
                remaining = self._observations[key]
                self._bucket_counts[key] = [
                    sum(1 for v in remaining if v <= upper)
                    for upper in self._buckets
                ]
                # Recompute sum/count to stay consistent with remaining data.
                self._sums[key] = sum(remaining)
                self._counts[key] = len(remaining)

    def format(self) -> str:
        lines: list[str] = []
        lines.append(f"# HELP {self._name} {self._help}")
        lines.append(f"# TYPE {self._name} histogram")
        with self._lock:
            for key in sorted(self._observations, key=_label_sort_key):
                count = self._counts.get(key, 0)
                if count == 0:
                    continue
                label_str = _format_labels(key)
                total = self._sums.get(key, 0.0)
                bc = self._bucket_counts.get(key, [0] * len(self._buckets))
                # Build the label portion for bucket lines.  Prometheus format
                # requires the le= label mixed with other labels, e.g.
                #   metric_bucket{le="0.1",method="POST"} 5
                extra = label_str[1:-1] if label_str else ""
                for i, upper in enumerate(self._buckets):
                    bucket_val = bc[i]
                    if extra:
                        lines.append(
                            f'{self._name}_bucket{{le="{upper}",{extra}}} {bucket_val}'
                        )
                    else:
                        lines.append(
                            f'{self._name}_bucket{{le="{upper}"}} {bucket_val}'
                        )
                # +Inf bucket.
                if extra:
                    lines.append(
                        f'{self._name}_bucket{{le="+Inf",{extra}}} {count}'
                    )
                else:
                    lines.append(
                        f'{self._name}_bucket{{le="+Inf"}} {count}'
                    )
                lines.append(f"{self._name}_sum{label_str} {total}")
                lines.append(f"{self._name}_count{label_str} {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _label_sort_key(key: frozenset[tuple[str, str]]) -> str:
    """Stable sort key for label sets."""
    return ",".join(f"{k}={v}" for k, v in sorted(key))


def _format_labels(key: frozenset[tuple[str, str]]) -> str:
    """Serialise label set to Prometheus label string, e.g. {a="b",c="d"}."""
    if not key:
        return ""
    def _esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    pairs = ",".join(f'{k}="{_esc(v)}"' for k, v in sorted(key))
    return f"{{{pairs}}}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PrometheusMetrics:
    """Central Prometheus-compatible metrics registry.

    Usage::

        pm = get_prometheus_metrics()
        pm.inc_counter("request_total", {"method": "POST", "status": "200"})
        pm.observe_histogram("request_duration_seconds", 0.42, {"endpoint": "/v1/chat"})
        text = pm.generate()  # Prometheus exposition format
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}

        # --- Predefined metrics ---
        self._counters["request_total"] = _Counter(
            "yunshu_request_total",
            "Total HTTP requests processed",
        )
        self._counters["tokens_generated_total"] = _Counter(
            "yunshu_tokens_generated_total",
            "Total completion tokens generated",
        )

        self._gauges["active_requests"] = _Gauge(
            "yunshu_active_requests",
            "Currently in-flight requests",
        )
        self._gauges["kv_cache_blocks_used"] = _Gauge(
            "yunshu_kv_cache_blocks_used",
            "KV cache blocks currently in use",
        )
        self._gauges["kv_cache_blocks_total"] = _Gauge(
            "yunshu_kv_cache_blocks_total",
            "KV cache blocks allocated",
        )

        self._histograms["request_duration_seconds"] = _Histogram(
            "yunshu_request_duration_seconds",
            "HTTP request duration in seconds",
        )
        self._histograms["inference_duration_seconds"] = _Histogram(
            "yunshu_inference_duration_seconds",
            "Inference step duration in seconds",
        )
        self._histograms["prefill_duration_seconds"] = _Histogram(
            "yunshu_prefill_duration_seconds",
            "Prefill phase duration in seconds",
            buckets=_Histogram.INFERENCE_BUCKETS,
        )
        self._histograms["ttft_seconds"] = _Histogram(
            "yunshu_ttft_seconds",
            "Time to first token in seconds",
            buckets=_Histogram.INFERENCE_BUCKETS,
        )
        self._histograms["itl_seconds"] = _Histogram(
            "yunshu_itl_seconds",
            "Inter-token latency in seconds",
            buckets=_Histogram.INFERENCE_BUCKETS,
        )

        # Spec decode gauges
        self._gauges["spec_ngram_proposals"] = _Gauge(
            "yunshu_spec_ngram_proposals_total",
            "Total N-gram spec decode proposals made",
        )
        self._gauges["spec_ngram_accepted"] = _Gauge(
            "yunshu_spec_ngram_accepted_total",
            "Total N-gram spec decode tokens accepted",
        )
        self._gauges["spec_ngram_draft"] = _Gauge(
            "yunshu_spec_ngram_draft_total",
            "Total N-gram draft tokens generated",
        )
        self._gauges["spec_enabled"] = _Gauge(
            "yunshu_spec_decode_enabled",
            "Whether speculative decoding is enabled (1=yes, 0=no)",
        )
        # Cross-model speculative decoder gauges (SpeculativeDecoder._stats)
        self._gauges["spec_draft_tokens"] = _Gauge(
            "yunshu_spec_draft_tokens_total",
            "Total draft tokens generated by cross-model speculative decoder",
        )
        self._gauges["spec_accepted_tokens"] = _Gauge(
            "yunshu_spec_accepted_tokens_total",
            "Total tokens accepted by cross-model speculative decoder",
        )
        self._gauges["spec_acceptance_rate"] = _Gauge(
            "yunshu_spec_acceptance_rate",
            "Acceptance rate of cross-model speculative decoder",
        )
        self._gauges["spec_bonus_tokens"] = _Gauge(
            "yunshu_spec_bonus_tokens_total",
            "Total bonus tokens from cross-model speculative decoder",
        )
        self._gauges["spec_effective_speedup"] = _Gauge(
            "yunshu_spec_effective_speedup",
            "Effective speedup ratio from speculative decoding",
        )
        # MTP speculative decoding gauges
        self._gauges["spec_mtp_accepts"] = _Gauge(
            "yunshu_spec_mtp_accepts_total",
            "Total MTP speculative decoding accepts",
        )
        self._gauges["spec_mtp_rejects"] = _Gauge(
            "yunshu_spec_mtp_rejects_total",
            "Total MTP speculative decoding rejects",
        )
        self._gauges["spec_mtp_acceptance_rate"] = _Gauge(
            "yunshu_spec_mtp_acceptance_rate",
            "MTP speculative decoding acceptance rate",
        )
        self._gauges["mtp_acceptance_rate"] = _Gauge(
            "yunshu_mtp_acceptance_rate",
            "MTP acceptance rate (alternate gauge set from batched_engine)",
        )
        self._gauges["mtp_total_cycles"] = _Gauge(
            "yunshu_mtp_total_cycles",
            "MTP total speculative decoding cycles",
        )

        # KV prefix cache gauges
        self._gauges["kv_prefix_cache_entries"] = _Gauge(
            "yunshu_kv_prefix_cache_entries",
            "Number of entries in the KV prefix cache",
        )
        self._gauges["kv_prefix_cache_hits"] = _Gauge(
            "yunshu_kv_prefix_cache_hits_total",
            "Total KV prefix cache hits",
        )
        self._gauges["kv_prefix_cache_misses"] = _Gauge(
            "yunshu_kv_prefix_cache_misses_total",
            "Total KV prefix cache misses",
        )

        # RadixTree eviction gauges
        self._gauges["radix_evictions_lru"] = _Gauge(
            "yunshu_radix_evictions_lru_total",
            "Total RadixTree LRU evictions",
        )
        self._gauges["radix_evictions_lfu"] = _Gauge(
            "yunshu_radix_evictions_lfu_total",
            "Total RadixTree LFU evictions",
        )
        self._gauges["radix_evictions_fifo"] = _Gauge(
            "yunshu_radix_evictions_fifo_total",
            "Total RadixTree FIFO evictions",
        )
        self._gauges["radix_evictions_freed_blocks"] = _Gauge(
            "yunshu_radix_evictions_freed_blocks_total",
            "Total KV blocks freed by RadixTree evictions",
        )
        self._gauges["radix_total_nodes"] = _Gauge(
            "yunshu_radix_total_nodes",
            "Total nodes in the RadixTree",
        )
        self._gauges["radix_total_tokens"] = _Gauge(
            "yunshu_radix_total_tokens",
            "Total tokens stored in the RadixTree",
        )

        # Chunked prefill gauges (Wave 108)
        self._gauges["chunked_prefill_active_chunks"] = _Gauge(
            "yunshu_chunked_prefill_active_chunks",
            "Number of requests currently in chunked prefill",
        )
        self._gauges["chunked_prefill_total_chunks_processed"] = _Gauge(
            "yunshu_chunked_prefill_total_chunks_processed",
            "Total chunked prefill chunks processed",
        )
        self._gauges["chunked_prefill_budget_used"] = _Gauge(
            "yunshu_chunked_prefill_budget_used",
            "Chunked prefill token budget used so far",
        )
        self._gauges["chunked_prefill_budget_limit"] = _Gauge(
            "yunshu_chunked_prefill_budget_limit",
            "Chunked prefill token budget limit",
        )

        # ITL gauges (set from monitoring.py prometheus_export endpoint)
        self._gauges["itl_p50_ms"] = _Gauge(
            "yunshu_itl_p50_ms",
            "Inter-token latency p50 in milliseconds",
        )
        self._gauges["itl_p99_ms"] = _Gauge(
            "yunshu_itl_p99_ms",
            "Inter-token latency p99 in milliseconds",
        )

        # Scheduler monitoring gauges (MON-2/4/5)
        self._gauges["scheduler_waiting_queue_depth"] = _Gauge(
            "yunshu_scheduler_waiting_queue_depth",
            "Number of requests waiting in the scheduler queue",
        )
        self._gauges["scheduler_batch_size"] = _Gauge(
            "yunshu_scheduler_batch_size",
            "Current scheduler step batch size",
        )
        self._gauges["compute_utilization_pct"] = _Gauge(
            "yunshu_compute_utilization_pct",
            "Percentage of time spent in active scheduler steps",
        )
        self._gauges["step_duration_ms"] = _Gauge(
            "yunshu_step_duration_ms",
            "Last scheduler step wall time in milliseconds",
        )
        self._gauges["attention_eviction_tracked_requests"] = _Gauge(
            "yunshu_attention_eviction_tracked_requests",
            "Number of requests tracked by H2O attention eviction",
        )
        self._gauges["attention_eviction_total_blocks"] = _Gauge(
            "yunshu_attention_eviction_total_blocks",
            "Total KV blocks scored by H2O attention eviction",
        )

    # --- Counter API ---

    def inc_counter(self, name: str, labels: Optional[dict[str, str]] = None, amount: int = 1) -> None:
        """Increment a named counter by *amount*."""
        with self._lock:
            if name not in self._counters:
                raise KeyError(f"Unknown counter: {name!r}")
            self._counters[name].inc(labels, amount)

    def get_counter(self, name: str, labels: Optional[dict[str, str]] = None) -> int:
        """Read current value of a counter."""
        with self._lock:
            if name not in self._counters:
                raise KeyError(f"Unknown counter: {name!r}")
            return self._counters[name].get(labels)

    # --- Gauge API ---

    def set_gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Set a named gauge to *value*."""
        with self._lock:
            if name not in self._gauges:
                raise KeyError(f"Unknown gauge: {name!r}")
            self._gauges[name].set(value, labels)

    def inc_gauge(self, name: str, labels: Optional[dict[str, str]] = None, amount: float = 1.0) -> None:
        with self._lock:
            if name not in self._gauges:
                raise KeyError(f"Unknown gauge: {name!r}")
            self._gauges[name].inc(labels, amount)

    def dec_gauge(self, name: str, labels: Optional[dict[str, str]] = None, amount: float = 1.0) -> None:
        with self._lock:
            if name not in self._gauges:
                raise KeyError(f"Unknown gauge: {name!r}")
            self._gauges[name].dec(labels, amount)

    # --- Histogram API ---

    def observe_histogram(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Observe *value* in a named histogram."""
        with self._lock:
            if name not in self._histograms:
                raise KeyError(f"Unknown histogram: {name!r}")
            self._histograms[name].observe(value, labels)

    # --- Serialisation ---

    def generate(self) -> str:
        """Generate Prometheus exposition format text."""
        sections: list[str] = []

        # Snapshot metric dicts under the outer lock, then release it
        # before calling format() (which acquires each metric's own lock).
        # This avoids holding the outer lock while doing I/O-heavy work
        # (histogram bucket computation), so writers are not blocked.
        with self._lock:
            counters = list(self._counters.values())
            gauges = list(self._gauges.values())
            histograms = list(self._histograms.values())

        # Counters first.
        for c in sorted(counters, key=lambda m: m._name):
            sections.append(c.format())
        # Gauges.
        for g in sorted(gauges, key=lambda m: m._name):
            sections.append(g.format())
        # Histograms.
        for h in sorted(histograms, key=lambda m: m._name):
            sections.append(h.format())

        # Append an uptime gauge.
        sections.append("# HELP yunshu_exporter_uptime_seconds Prometheus exporter uptime")
        sections.append("# TYPE yunshu_exporter_uptime_seconds gauge")
        sections.append(f"yunshu_exporter_uptime_seconds {time.time() - _BORN:.1f}")

        return "\n".join(sections) + "\n"


# Timestamp at module load for exporter uptime.
_BORN = time.time()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: PrometheusMetrics | None = None
_instance_lock = Lock()


def get_prometheus_metrics() -> PrometheusMetrics:
    """Return the global PrometheusMetrics singleton (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PrometheusMetrics()
    return _instance


def reset_prometheus_metrics() -> None:
    """Reset the singleton (for tests only)."""
    global _instance
    with _instance_lock:
        _instance = None
