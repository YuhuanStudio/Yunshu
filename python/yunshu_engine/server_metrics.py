"""Server metrics — thread-safe request tracking with per-model breakdown.

Based on oMLX's ServerMetrics pattern: session + all-time scopes,
per-model counters, periodic JSON persistence.
"""


import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SAVE_INTERVAL = 300  # seconds


class ServerMetrics:
    """Thread-safe server-level metrics aggregator.

    Scopes:
    - session: resets on server restart
    - alltime: persisted across restarts via stats_path JSON
    """

    def __init__(self, stats_path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._stats_path = stats_path

        # Session totals
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_requests: int = 0
        self.total_prefill_duration: float = 0.0
        self.total_generation_duration: float = 0.0
        self._per_model: dict[str, dict[str, Any]] = {}

        # All-time totals
        self._alltime_prompt_tokens: int = 0
        self._alltime_completion_tokens: int = 0
        self._alltime_cached_tokens: int = 0
        self._alltime_requests: int = 0
        self._alltime_prefill_duration: float = 0.0
        self._alltime_generation_duration: float = 0.0
        self._alltime_per_model: dict[str, dict[str, Any]] = {}

        # ITL histogram (ITL-1: inter-token latency tracking)
        self._itl_samples: list[float] = []
        self._itl_p50: float = 0.0
        self._itl_p99: float = 0.0

        # Batch size distribution tracking (MON-2)
        self._batch_size_samples: list[int] = []
        self._batch_size_p50: int = 0
        self._batch_size_p99: int = 0

        self._start_time = time.time()
        self._last_save_time = time.time()

        if stats_path:
            self._load_alltime()

    @staticmethod
    def _new_model_counters() -> dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "requests": 0,
            "prefill_duration": 0.0,
            "generation_duration": 0.0,
        }

    def _load_alltime(self) -> None:
        if not self._stats_path or not self._stats_path.exists():
            return
        try:
            with open(self._stats_path) as f:
                data = json.load(f)
            self._alltime_prompt_tokens = int(data.get("total_prompt_tokens", 0))
            self._alltime_completion_tokens = int(data.get("total_completion_tokens", 0))
            self._alltime_cached_tokens = int(data.get("total_cached_tokens", 0))
            self._alltime_requests = int(data.get("total_requests", 0))
            self._alltime_prefill_duration = float(data.get("total_prefill_duration", 0.0))
            self._alltime_generation_duration = float(data.get("total_generation_duration", 0.0))
            for model_id, counters in data.get("per_model", {}).items():
                self._alltime_per_model[model_id] = {
                    k: float(v) if "duration" in k else int(v)
                    for k, v in counters.items()
                }
            logger.info("Loaded all-time stats from %s", self._stats_path)
        except Exception as e:
            logger.warning("Failed to load all-time stats: %s", e)

    def save_alltime(self) -> None:
        if not self._stats_path:
            return
        with self._lock:
            data = {
                "total_prompt_tokens": self._alltime_prompt_tokens,
                "total_completion_tokens": self._alltime_completion_tokens,
                "total_cached_tokens": self._alltime_cached_tokens,
                "total_requests": self._alltime_requests,
                "total_prefill_duration": self._alltime_prefill_duration,
                "total_generation_duration": self._alltime_generation_duration,
                "per_model": dict(self._alltime_per_model),
            }
            self._last_save_time = time.time()
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._stats_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self._stats_path)
        except OSError as e:
            logger.warning("Failed to save all-time stats: %s", e)

    def record_request_complete(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        prefill_duration: float = 0.0,
        generation_duration: float = 0.0,
        model_id: str = "",
    ) -> None:
        with self._lock:
            # Session
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.total_requests += 1
            self.total_prefill_duration += prefill_duration
            self.total_generation_duration += generation_duration

            # All-time
            self._alltime_prompt_tokens += prompt_tokens
            self._alltime_completion_tokens += completion_tokens
            self._alltime_cached_tokens += cached_tokens
            self._alltime_requests += 1
            self._alltime_prefill_duration += prefill_duration
            self._alltime_generation_duration += generation_duration

            # Per-model
            if model_id:
                for store in (self._per_model, self._alltime_per_model):
                    if model_id not in store:
                        store[model_id] = self._new_model_counters()
                    m = store[model_id]
                    m["prompt_tokens"] += prompt_tokens
                    m["completion_tokens"] += completion_tokens
                    m["cached_tokens"] += cached_tokens
                    m["requests"] += 1
                    m["prefill_duration"] += prefill_duration
                    m["generation_duration"] += generation_duration

            # Periodic save (flag-based to avoid lock reentrance)
            needs_save = (
                self._stats_path
                and time.time() - self._last_save_time >= _SAVE_INTERVAL
            )

        if needs_save:
            self.save_alltime()

    def record_itl(self, itl_seconds: float) -> None:
        """Record an inter-token latency sample (ITL-1).

        Called from Scheduler._process_responses for each generated token.
        Maintains a bounded buffer and computes percentiles on flush.
        """
        self._itl_samples.append(itl_seconds)
        # Keep buffer bounded (flush and compute percentiles every 1000 samples)
        if len(self._itl_samples) >= 1000:
            self._compute_itl_percentiles()

    def _compute_itl_percentiles(self) -> None:
        """Compute ITL percentiles from collected samples."""
        if not self._itl_samples:
            return
        import bisect
        samples = sorted(self._itl_samples)
        n = len(samples)
        self._itl_p50 = samples[n // 2]
        self._itl_p99 = samples[min(int(n * 0.99), n - 1)]
        self._itl_samples = samples[-100:]  # Keep last 100 for rolling stats

    def get_itl_stats(self) -> dict[str, Any]:
        """Return ITL statistics."""
        if self._itl_samples:
            self._compute_itl_percentiles()
        return {
            "itl_p50_ms": round(self._itl_p50 * 1000, 2),
            "itl_p99_ms": round(self._itl_p99 * 1000, 2),
            "itl_samples_buffered": len(self._itl_samples),
        }

    def record_batch_size(self, batch_size: int) -> None:
        """Record a scheduler batch size sample (MON-2).

        Called from engine_core._engine_loop after each scheduler step.
        Maintains a bounded buffer and computes percentiles on flush.
        """
        self._batch_size_samples.append(batch_size)
        if len(self._batch_size_samples) >= 1000:
            self._compute_batch_size_percentiles()

    def _compute_batch_size_percentiles(self) -> None:
        """Compute batch size percentiles from collected samples."""
        if not self._batch_size_samples:
            return
        samples = sorted(self._batch_size_samples)
        n = len(samples)
        self._batch_size_p50 = samples[n // 2]
        self._batch_size_p99 = samples[min(int(n * 0.99), n - 1)]
        self._batch_size_samples = samples[-100:]  # Keep last 100 for rolling stats

    def get_batch_size_stats(self) -> dict[str, Any]:
        """Return batch size distribution statistics."""
        if self._batch_size_samples:
            self._compute_batch_size_percentiles()
        return {
            "batch_size_p50": self._batch_size_p50,
            "batch_size_p99": self._batch_size_p99,
            "batch_size_samples_buffered": len(self._batch_size_samples),
        }

    def _build_snapshot(
        self,
        prompt: int,
        completion: int,
        cached: int,
        requests: int,
        prefill_dur: float,
        gen_dur: float,
        uptime: float,
    ) -> dict[str, Any]:
        actual = prompt - cached
        avg_prefill_tps = actual / prefill_dur if prefill_dur > 0 else 0.0
        avg_gen_tps = completion / gen_dur if gen_dur > 0 else 0.0
        cache_eff = (cached / prompt * 100) if prompt > 0 else 0.0

        return {
            "total_tokens_served": prompt + completion,
            "total_cached_tokens": cached,
            "cache_efficiency_pct": round(cache_eff, 1),
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_requests": requests,
            "avg_prefill_tps": round(avg_prefill_tps, 1),
            "avg_generation_tps": round(avg_gen_tps, 1),
            "uptime_seconds": round(uptime, 1),
        }

    def get_snapshot(
        self, model_id: str = "", scope: str = "session"
    ) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time

            if scope == "alltime":
                src = self._alltime_per_model.get(model_id, {}) if model_id else {
                    "prompt_tokens": self._alltime_prompt_tokens,
                    "completion_tokens": self._alltime_completion_tokens,
                    "cached_tokens": self._alltime_cached_tokens,
                    "requests": self._alltime_requests,
                    "prefill_duration": self._alltime_prefill_duration,
                    "generation_duration": self._alltime_generation_duration,
                }
                if model_id and model_id in self._alltime_per_model:
                    src = self._alltime_per_model[model_id]
                elif model_id:
                    return self._build_snapshot(0, 0, 0, 0, 0, 0, uptime)
            else:
                src = self._per_model.get(model_id, {}) if model_id else {
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                    "cached_tokens": self.total_cached_tokens,
                    "requests": self.total_requests,
                    "prefill_duration": self.total_prefill_duration,
                    "generation_duration": self.total_generation_duration,
                }
                if model_id and model_id in self._per_model:
                    src = self._per_model[model_id]
                elif model_id:
                    return self._build_snapshot(0, 0, 0, 0, 0, 0, uptime)

            return self._build_snapshot(
                src.get("prompt_tokens", 0),
                src.get("completion_tokens", 0),
                src.get("cached_tokens", 0),
                src.get("requests", 0),
                src.get("prefill_duration", 0),
                src.get("generation_duration", 0),
                uptime,
            )

    def clear_session(self) -> None:
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cached_tokens = 0
            self.total_requests = 0
            self.total_prefill_duration = 0.0
            self.total_generation_duration = 0.0
            self._per_model.clear()


# Global singleton
_server_metrics: Optional[ServerMetrics] = None


def get_server_metrics() -> ServerMetrics:
    global _server_metrics
    if _server_metrics is None:
        _server_metrics = ServerMetrics()
    return _server_metrics


def reset_server_metrics(stats_path: Optional[Path] = None) -> None:
    global _server_metrics
    if _server_metrics is not None:
        _server_metrics.save_alltime()
    _server_metrics = ServerMetrics(stats_path=stats_path)
