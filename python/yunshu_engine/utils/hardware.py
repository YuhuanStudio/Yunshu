"""Unified hardware detection for Apple Silicon.

oMLX pattern: single source of truth for chip identification,
memory detection, MLX availability, and version info.

Fallback chain for each detection: sysctl → MLX Metal → heuristic → default.
"""
from __future__ import annotations

import logging
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_BYTES = 8 * 1024 ** 3


@dataclass
class HardwareInfo:
    chip_name: str
    total_memory_gb: float
    max_working_set_bytes: int
    gpu_cores: Optional[int] = None
    mlx_device_name: Optional[str] = None
    os_version: str = ""


def get_chip_name() -> str:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        logger.debug("sysctl chip name detection failed", exc_info=True)
        return "Apple Silicon"


def get_total_memory_bytes() -> int:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True,
        )
        return int(r.stdout.strip())
    except Exception:
        logger.debug("failed", exc_info=True)
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                info = mx.device_info()
                if "memory_size" in info:
                    return int(info["memory_size"])
        except Exception:
            logger.debug("failed", exc_info=True)
    return DEFAULT_MEMORY_BYTES


def get_system_memory_gb() -> float:
    return get_total_memory_bytes() / (1024 ** 3)


def get_max_working_set_bytes() -> int:
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                info = mx.device_info()
                ws = info.get("max_recommended_working_set_size", 0)
                if ws > 0:
                    return ws
        except Exception:
            logger.debug("failed", exc_info=True)
    try:
        import psutil
        return int(psutil.virtual_memory().total * 0.75)
    except ImportError:
        pass
    return DEFAULT_MEMORY_BYTES


def get_gpu_core_count() -> Optional[int]:
    try:
        r = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "Total Number of Cores" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    return int(m.group(1))
    except Exception:
        logger.debug("failed", exc_info=True)
    return None


def get_mlx_device_name() -> Optional[str]:
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                return mx.device_info().get("device_name")
        except Exception:
            logger.debug("failed", exc_info=True)
    return None


def get_os_version() -> str:
    try:
        ver = platform.mac_ver()[0]
        if ver:
            return f"macOS {ver}"
    except Exception:
        logger.debug("failed", exc_info=True)
    return "macOS"


def get_mlx_version() -> str:
    try:
        import mlx
        return getattr(mlx, "__version__", "unknown")
    except Exception:
        logger.debug("mlx version detection failed", exc_info=True)
        return "unavailable"


def get_mlx_lm_version() -> str:
    try:
        import mlx_lm
        return getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        logger.debug("mlx-lm version detection failed", exc_info=True)
        return "unavailable"


def parse_chip_info(chip_string: str) -> tuple[str, str]:
    m = re.search(r"M(\d+)\s*(Pro|Max|Ultra)?", chip_string)
    if not m:
        return ("M1", "")
    return (f"M{m.group(1)}", m.group(2) or "")


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_mlx_available() -> bool:
    if not is_apple_silicon():
        return False
    if not HAS_MLX:
        return False
    try:
        mx.array([1.0])
        return True
    except Exception:
        logger.debug("MLX array test failed", exc_info=True)
        return False


def format_bytes(b: int) -> str:
    if b == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(
        chip_name=get_chip_name(),
        total_memory_gb=get_system_memory_gb(),
        max_working_set_bytes=get_max_working_set_bytes(),
        gpu_cores=get_gpu_core_count(),
        mlx_device_name=get_mlx_device_name(),
        os_version=get_os_version(),
    )


def compute_adaptive_defaults(hw: HardwareInfo | None = None) -> dict:
    """Compute optimal ModelSettings overrides based on detected hardware.

    oMLX pattern: auto-tune batch size, KV cache limits, prefill chunks,
    and prefix cache size based on chip generation and memory.

    Returns a dict of ModelSettings field overrides (only non-default values).
    """
    if hw is None:
        hw = detect_hardware()

    overrides: dict = {}

    chip_gen, chip_tier = parse_chip_info(hw.chip_name)
    mem_gb = hw.total_memory_gb
    ws_bytes = hw.max_working_set_bytes
    ws_gb = ws_bytes / (1024 ** 3)

    # ── Max KV cache memory ──
    # Reserve 40% of working set for model weights + activations, rest for KV
    max_kv = int(ws_bytes * 0.60)
    overrides["max_kv_cache_memory"] = max_kv

    # ── Memory pressure threshold ──
    # Lower threshold on memory-constrained devices (8GB)
    if mem_gb <= 8:
        overrides["memory_pressure_threshold"] = 70.0
    elif mem_gb <= 16:
        overrides["memory_pressure_threshold"] = 80.0
    else:
        overrides["memory_pressure_threshold"] = 85.0

    # ── Batch size ──
    # M1/M2 base: 1, Pro: 2, Max: 4, Ultra: 8
    tier_batch = {"": 1, "Pro": 2, "Max": 4, "Ultra": 8}
    base_batch = tier_batch.get(chip_tier, 1)
    # Scale with memory: 8GB → base, 16GB → base*2, 32+ → base*3
    if mem_gb <= 8:
        overrides["batch_size"] = base_batch
    elif mem_gb <= 16:
        overrides["batch_size"] = base_batch * 2
    elif mem_gb <= 32:
        overrides["batch_size"] = base_batch * 3
    else:
        overrides["batch_size"] = base_batch * 4

    # ── Prefill chunk size ──
    # Larger chunks on higher-bandwidth chips
    if chip_tier in ("Max", "Ultra"):
        overrides["prefill_chunk_size"] = 4096
    elif chip_tier == "Pro":
        overrides["prefill_chunk_size"] = 2048
    else:
        overrides["prefill_chunk_size"] = 1024

    # ── Prefix cache ──
    # More entries on larger memory systems
    if mem_gb <= 8:
        overrides["prefix_cache_max_entries"] = 16
        overrides["prefix_cache_min_prefix"] = 64
    elif mem_gb <= 16:
        overrides["prefix_cache_max_entries"] = 32
        overrides["prefix_cache_min_prefix"] = 48
    else:
        overrides["prefix_cache_max_entries"] = 64
        overrides["prefix_cache_min_prefix"] = 32

    # ── KV quantization ──
    # Auto-enable 4-bit KV quant on 8GB devices
    if mem_gb <= 8:
        overrides["kv_cache_quant_bits"] = 4
        overrides["kv_cache_quant_group_size"] = 64

    # ── Spec decode ──
    # N-gram spec decode is cheap — enable on 16GB+
    if mem_gb >= 16:
        overrides["ngram_spec_enabled"] = True

    # ── SSD cache ──
    if mem_gb <= 16:
        overrides["ssd_cache_enabled"] = True
        overrides["ssd_cache_max_gb"] = min(int(mem_gb), 10)

    # ── Streaming ──
    # Shorter keepalive on memory-constrained devices
    if mem_gb <= 8:
        overrides["stream_keepalive_interval"] = 10.0

    return overrides


def get_hardware_profile() -> dict:
    """Return a full hardware profile summary for diagnostics."""
    hw = detect_hardware()
    chip_gen, chip_tier = parse_chip_info(hw.chip_name)
    adaptive = compute_adaptive_defaults(hw)
    return {
        "chip_name": hw.chip_name,
        "chip_generation": chip_gen,
        "chip_tier": chip_tier,
        "total_memory_gb": round(hw.total_memory_gb, 1),
        "working_set_gb": round(hw.max_working_set_bytes / (1024 ** 3), 1),
        "gpu_cores": hw.gpu_cores,
        "mlx_device": hw.mlx_device_name,
        "os_version": hw.os_version,
        "mlx_version": get_mlx_version(),
        "mlx_lm_version": get_mlx_lm_version(),
        "adaptive_defaults": adaptive,
    }
