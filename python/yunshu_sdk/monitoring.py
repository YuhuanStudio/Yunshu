"""Yunshu SDK — Monitoring namespace (system stats, model stats).

Provides real-time monitoring of system resources, engine internals,
per-model performance metrics, and request-level statistics.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class Monitoring:
    """Monitoring namespace (client.monitoring).

    Access system-level stats (CPU, memory, GPU), engine-level stats
    (batch size, cache utilization), per-model metrics (throughput,
    latency), and request-level statistics.
    """

    def __init__(self, http: httpx.Client):
        self._http = http

    def system(self) -> dict:
        """Get system-level stats.

        Returns:
            Dict with keys:
                - cpu_usage_pct: float — CPU utilization percentage
                - memory_total_bytes: int — Total unified memory
                - memory_active_bytes: int — Active MLX memory
                - memory_peak_bytes: int — Peak MLX memory
                - gpu_cores: int — Number of GPU cores
                - gpu_utilization_pct: float — GPU utilization (approximate)
                - uptime_seconds: float — Server uptime
        """
        resp = self._http.get("/api/v1/monitoring/system")
        resp.raise_for_status()
        return resp.json()

    def engine(self) -> dict:
        """Get engine-level stats.

        Returns:
            Dict with keys:
                - scheduler: dict — Waiting/running/finished request counts
                - batch_size: int — Current active batch size
                - kv_cache: dict — KV cache utilization, block counts
                - thinking_segments: dict — Thinking KV substore stats
                - step_counter: int — Total scheduler steps
                - total_prompt_tokens: int — Cumulative prompt tokens
                - total_completion_tokens: int — Cumulative completion tokens
        """
        resp = self._http.get("/api/v1/monitoring/engine")
        resp.raise_for_status()
        return resp.json()

    def requests(self) -> dict:
        """Get request-level stats.

        Returns:
            Dict with keys:
                - active_requests: int — Currently processing
                - waiting_requests: int — In queue
                - completed_requests: int — Finished total
                - avg_latency_ms: float — Average end-to-end latency
                - avg_ttft_ms: float — Average time to first token
                - avg_tpot_ms: float — Average time per output token
                - total_tokens_per_sec: float — Aggregate throughput
        """
        resp = self._http.get("/api/v1/monitoring/requests")
        resp.raise_for_status()
        return resp.json()

    def models(self) -> dict:
        """Get per-model stats.

        Returns:
            Dict with model_id → stats mapping, where each stats dict has:
                - status: str — 'loaded', 'loading', 'unloaded'
                - num_requests: int — Active requests for this model
                - tokens_per_sec: float — Model-specific throughput
                - avg_latency_ms: float — Model-specific latency
                - memory_bytes: int — Memory used by this model
                - kv_cache: dict — Model-specific KV cache stats
        """
        resp = self._http.get("/api/v1/monitoring/models")
        resp.raise_for_status()
        return resp.json()

    def kv_cache(self, model_id: Optional[str] = None) -> dict:
        """Get KV cache tier statistics.

        Args:
            model_id: Optional model filter. If None, returns aggregate stats.

        Returns:
            Dict with keys:
                - hot: dict — Hot tier stats (usage_pct, free_blocks, block_size)
                - warm: dict — Warm tier stats (num_blocks, hit_rate, utilization)
                - ssd: dict — SSD tier stats (num_entries, size_gb, utilization)
        """
        url = "/api/v1/monitoring/kv_cache"
        if model_id:
            url += f"?model_id={model_id}"
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    def timeline(
        self,
        duration_seconds: int = 60,
        resolution_seconds: int = 1,
    ) -> list[dict]:
        """Get time-series monitoring data.

        Args:
            duration_seconds: How far back to look (default 60s).
            resolution_seconds: Bucket size for aggregation (default 1s).

        Returns:
            List of timestamped data points with throughput, latency,
            and resource utilization metrics.
        """
        resp = self._http.get(
            "/api/v1/monitoring/timeline",
            params={
                "duration": duration_seconds,
                "resolution": resolution_seconds,
            },
        )
        resp.raise_for_status()
        return resp.json().get("points", [])
