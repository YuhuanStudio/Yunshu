from __future__ import annotations
"""Yunshu Memory Monitor — Apple Silicon unified memory tracking.

- GPU memory utilization tracking via MLX Metal API
- Block memory estimation for KV cache management
- Prefill peak memory estimation

ProcessMemoryEnforcer lives in process_memory_enforcer.py (full version
with TTL support, model eviction logic).

Key MLX Metal APIs:
- mx.metal.is_available() — check Metal support
- mx.get_active_memory() — current active Metal memory
- mx.get_peak_memory() — peak Metal memory since last reset
- mx.metal.get_memory_info() — detailed memory breakdown
"""


import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Check MLX Metal availability
try:
    import mlx.core as mx

    HAS_MLX_METAL = mx.metal.is_available()
except (ImportError, AttributeError):
    HAS_MLX_METAL = False
    mx = None

from .utils.hardware import format_bytes


def get_system_memory() -> int:
    """Return total system RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        logger.debug("sysctl hw.memsize detection failed, using 16GB default", exc_info=True)
        return 16 * 1024 ** 3


def get_max_working_set_bytes() -> int:
    """Get max_recommended_working_set_size from MLX Metal.

    Falls back to system memory heuristic if MLX Metal unavailable.
    """
    if HAS_MLX_METAL:
        try:
            info = mx.metal.get_memory_info()
            if hasattr(info, 'max_recommended_working_set_size'):
                return int(info.max_recommended_working_set_size)
        except Exception:
            logger.debug("MLX max_working_set_size detection failed", exc_info=True)

    # Fallback: use 75% of system memory
    return int(get_system_memory() * 0.75)


@dataclass
class MemoryInfo:
    """Current GPU memory state."""

    total_bytes: int
    active_bytes: int
    peak_bytes: int
    cache_bytes: int
    available_bytes: int
    utilization_pct: float


class MemoryMonitor:
    """Memory monitor for GPU KV cache management on Apple Silicon.

    Tracks MLX Metal memory usage for:
    - KV cache memory estimation
    - Memory pressure detection
    - Block memory calculations

    Adapted from oMLX's MemoryMonitor pattern.
    """

    def __init__(
        self,
        max_kv_cache_memory: int = 0,
        check_interval: float = 1.0,
    ):
        self._max_kv_cache_memory = max_kv_cache_memory
        self._check_interval = check_interval
        self._max_memory = get_max_working_set_bytes()

        self._last_check_time: float = 0.0
        self._last_info: Optional[MemoryInfo] = None
        self._lock = threading.Lock()

        # Model architecture info (set after model load)
        self._num_layers: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._dtype_size: int = 2  # float16
        self._num_attention_heads: Optional[int] = None

        # Baseline memory (model weights) — set after model load
        self._baseline_memory: int = 0

        logger.info(
            f"MemoryMonitor initialized: "
            f"max_working_set={format_bytes(self._max_memory)}, "
            f"max_kv_cache={format_bytes(max_kv_cache_memory) if max_kv_cache_memory else 'auto'}"
        )

    def set_baseline_memory(self) -> None:
        """Set baseline memory after model load (oMLX pattern).

        Captures model weight memory so KV cache growth can be
        accurately tracked relative to baseline.
        """
        if HAS_MLX_METAL:
            try:
                self._baseline_memory = mx.get_active_memory()
                logger.info(f"Baseline memory set: {format_bytes(self._baseline_memory)}")
            except Exception as e:
                logger.warning(f"Failed to set baseline memory: {e}")

    def set_model_info(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_size: int = 2,
        num_attention_heads: Optional[int] = None,
    ) -> None:
        """Set model architecture info for memory estimation."""
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype_size = dtype_size
        self._num_attention_heads = num_attention_heads or num_kv_heads

        if num_layers and num_kv_heads and head_dim:
            sample = self.estimate_block_memory(64)
            logger.info(
                f"Model info: {num_layers}L, {num_kv_heads} KV heads, "
                f"{num_attention_heads} Q heads, {head_dim}d. "
                f"Block(64tok): {format_bytes(sample)}"
            )

    def get_memory_info(self) -> MemoryInfo:
        """Get current memory state from MLX Metal."""
        with self._lock:
            now = time.time()
            if (
                self._last_info is not None
                and now - self._last_check_time < self._check_interval
            ):
                return self._last_info

            total = self._max_memory
            active = 0
            peak = 0
            cache = 0

            if HAS_MLX_METAL:
                try:
                    active = mx.get_active_memory()
                    peak = mx.get_peak_memory()
                    info = mx.metal.get_memory_info()
                    if hasattr(info, 'cache_memory'):
                        cache = int(info.cache_memory)
                except Exception:
                    logger.debug("MLX memory info read failed", exc_info=True)

            available = max(0, total - active)
            util = (active / total * 100) if total > 0 else 0.0

            self._last_info = MemoryInfo(
                total_bytes=total,
                active_bytes=active,
                peak_bytes=peak,
                cache_bytes=cache,
                available_bytes=available,
                utilization_pct=util,
            )
            self._last_check_time = now
            return self._last_info

    def is_under_pressure(self, threshold_pct: float = 90.0) -> bool:
        """Check if memory usage exceeds threshold."""
        info = self.get_memory_info()
        return info.utilization_pct >= threshold_pct

    def estimate_block_memory(self, block_size: int) -> int:
        """Estimate memory usage for one KV cache block."""
        layers = self._num_layers or 32
        kv_heads = self._num_kv_heads or 8
        dim = self._head_dim or 128
        dtype = self._dtype_size

        # Per layer: keys + values, shape (1, kv_heads, block_size, head_dim)
        per_layer = block_size * kv_heads * dim * dtype * 2
        return per_layer * layers

    def estimate_prompt_kv_bytes(self, num_tokens: int) -> int:
        """Estimate KV cache memory for a prompt of given length."""
        layers = self._num_layers or 0
        kv_heads = self._num_kv_heads or 0
        dim = self._head_dim or 0
        if not (layers and kv_heads and dim):
            return 0
        per_token = layers * kv_heads * dim * self._dtype_size * 2
        return num_tokens * per_token

    def estimate_prefill_peak_bytes(
        self, total_prompt_tokens: int, chunk_size: int,
    ) -> int:
        """Estimate worst-case peak memory during prefill (oMLX pattern).

        MLX SDPA internals:
        - head_dim > 128: full attention matrix materialized in float32
        - head_dim <= 128: fused kernel, tiled, O(n) memory
        """
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0
        if n_q == 0 or hd == 0:
            return 0

        if hd > 128:
            attn = n_q * chunk_size * total_prompt_tokens * 4
            attn += n_q * chunk_size * hd * 4
        else:
            attn = n_q * chunk_size * hd * 4

        kv = self.estimate_prompt_kv_bytes(total_prompt_tokens)
        return attn + kv

    def get_stats(self) -> dict:
        """Return memory stats for monitoring endpoints."""
        info = self.get_memory_info()
        return {
            "total_bytes": info.total_bytes,
            "active_bytes": info.active_bytes,
            "peak_bytes": info.peak_bytes,
            "cache_bytes": info.cache_bytes,
            "available_bytes": info.available_bytes,
            "utilization_pct": round(info.utilization_pct, 1),
            "max_kv_cache_memory": self._max_kv_cache_memory,
            "baseline_memory": self._baseline_memory,
        }


