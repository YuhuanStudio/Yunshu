from __future__ import annotations
"""Unified hardware detection for Apple Silicon.

oMLX pattern: single source of truth for chip identification,
memory detection, MLX availability, and version info.

Fallback chain for each detection: sysctl → MLX Metal → heuristic → default.
"""

import logging
import platform
import re
import subprocess
import sys
import threading
import time
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
    total_memory_bytes: int = 0
    gpu_cores: Optional[int] = None
    mlx_device_name: Optional[str] = None
    os_version: str = ""
    gpu_family: Optional[str] = None
    memory_bandwidth_gb: Optional[float] = None
    ane_available: Optional[bool] = None


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


def get_gpu_family() -> Optional[str]:
    """Detect GPU family name (e.g. "Apple M2 Max").

    Tries system_profiler first (most accurate on macOS), then falls back
    to sysctl hw.model, then to the existing chip_name from CPU brand.
    """
    try:
        r = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            # Match lines like "Chipset Model: Apple M2 Max"
            if "Chipset Model" in line:
                m = re.search(r":\s*(Apple\s+\S.*)", line)
                if m:
                    return m.group(1).strip()
            # Also try "Chipset:" on newer macOS
            if "Chipset:" in line:
                m = re.search(r":\s*(Apple\s+\S.*)", line)
                if m:
                    return m.group(1).strip()
    except Exception:
        logger.debug("system_profiler GPU family detection failed", exc_info=True)

    # Fallback: sysctl hw.model (e.g. "Mac15,12")
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, check=True,
        )
        hw_model = r.stdout.strip()
        if hw_model:
            # hw.model gives board IDs like "Mac15,12", not human-readable.
            # Use the CPU brand string as a more useful fallback.
            cpu_brand = get_chip_name()
            if "Apple" in cpu_brand:
                return cpu_brand
    except Exception:
        logger.debug("sysctl hw.model fallback failed", exc_info=True)

    # Last resort: use CPU brand string
    chip = get_chip_name()
    if "Apple" in chip:
        return chip
    return None


# Known Apple Silicon memory bandwidth (GB/s) per chip family.
# Sources: Apple specs, AnandTech, Chipwise measurements.
_MEMORY_BANDWIDTH: dict[str, float] = {
    # M1 family
    "M1": 68.0,
    "M1 Pro": 200.0,
    "M1 Max": 400.0,
    "M1 Ultra": 800.0,
    # M2 family
    "M2": 100.0,
    "M2 Pro": 200.0,
    "M2 Max": 400.0,
    "M2 Ultra": 800.0,
    # M3 family
    "M3": 100.0,
    "M3 Pro": 150.0,
    "M3 Max": 400.0,
    "M3 Ultra": 800.0,
    # M4 family
    "M4": 120.0,
    "M4 Pro": 273.0,
    "M4 Max": 546.0,
    "M4 Ultra": 819.0,
}


def get_memory_bandwidth_gb() -> Optional[float]:
    """Return memory bandwidth in GB/s for the detected chip.

    Uses a lookup table of known Apple Silicon bandwidth values.  Falls back
    to None if the chip cannot be identified.
    """
    chip = get_chip_name()
    if not chip:
        return None

    # Try exact match first (e.g. "Apple M3 Max")
    for key, bw in _MEMORY_BANDWIDTH.items():
        if key in chip:
            return bw

    # Parse chip generation + tier for partial match
    gen, tier = parse_chip_info(chip)
    lookup_key = f"{gen} {tier}".strip()
    if lookup_key in _MEMORY_BANDWIDTH:
        return _MEMORY_BANDWIDTH[lookup_key]
    if gen in _MEMORY_BANDWIDTH:
        return _MEMORY_BANDWIDTH[gen]

    return None


def get_ane_available() -> Optional[bool]:
    """Check if Apple Neural Engine (ANE) is present.

    Detection strategy:
    1. Check for ANE device in IOKit registry (most reliable)
    2. Check for CoreML / NeuralEngine framework
    3. Default: True on Apple Silicon (ANE is present on all M-series)
    """
    if not is_apple_silicon():
        return False

    # Method 1: IOKit registry lookup
    try:
        r = subprocess.run(
            ["ioreg", "-l", "-w0"],
            capture_output=True, text=True, timeout=3,
        )
        if "AppleNeuralEngine" in r.stdout:
            return True
    except Exception:
        logger.debug("ioreg ANE detection failed", exc_info=True)

    # Method 2: CoreML availability
    try:
        import coremltools  # noqa: F401
        return True
    except ImportError:
        pass

    # Method 3: All Apple Silicon chips have ANE — assume True on arm64
    return True


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
    total_mem_bytes = get_total_memory_bytes()
    return HardwareInfo(
        chip_name=get_chip_name(),
        total_memory_gb=total_mem_bytes / (1024 ** 3),
        total_memory_bytes=total_mem_bytes,
        max_working_set_bytes=get_max_working_set_bytes(),
        gpu_cores=get_gpu_core_count(),
        mlx_device_name=get_mlx_device_name(),
        os_version=get_os_version(),
        gpu_family=get_gpu_family(),
        memory_bandwidth_gb=get_memory_bandwidth_gb(),
        ane_available=get_ane_available(),
    )


_cached_hw: HardwareInfo | None = None
_cached_hw_time: float = 0.0
_HW_CACHE_TTL = 30.0
_hw_cache_lock = threading.Lock()


def get_hardware_info() -> HardwareInfo:
    """Cached hardware info singleton (refreshes every 30s).

    Used by engine_core, scheduler, and kv_offload for memory checks.
    Caching avoids sysctl overhead on every engine loop step.
    Thread-safe: protects check-and-update with a lock since it is
    called from both the MLX executor thread and the asyncio loop.
    """
    global _cached_hw, _cached_hw_time
    with _hw_cache_lock:
        now = time.monotonic()
        if _cached_hw is None or (now - _cached_hw_time) > _HW_CACHE_TTL:
            _cached_hw = detect_hardware()
            _cached_hw_time = now
        return _cached_hw


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
        "gpu_family": hw.gpu_family,
        "memory_bandwidth_gb": hw.memory_bandwidth_gb,
        "ane_available": hw.ane_available,
        "mlx_device": hw.mlx_device_name,
        "os_version": hw.os_version,
        "mlx_version": get_mlx_version(),
        "mlx_lm_version": get_mlx_lm_version(),
        "adaptive_defaults": adaptive,
    }
