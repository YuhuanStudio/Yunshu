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
        return "Apple Silicon"


def get_total_memory_bytes() -> int:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True,
        )
        return int(r.stdout.strip())
    except Exception:
        pass
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                info = mx.device_info()
                if "memory_size" in info:
                    return int(info["memory_size"])
        except Exception:
            pass
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
            pass
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
        pass
    return None


def get_mlx_device_name() -> Optional[str]:
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                return mx.device_info().get("device_name")
        except Exception:
            pass
    return None


def get_os_version() -> str:
    try:
        ver = platform.mac_ver()[0]
        if ver:
            return f"macOS {ver}"
    except Exception:
        pass
    return "macOS"


def get_mlx_version() -> str:
    try:
        import mlx
        return getattr(mlx, "__version__", "unknown")
    except Exception:
        return "unavailable"


def get_mlx_lm_version() -> str:
    try:
        import mlx_lm
        return getattr(mlx_lm, "__version__", "unknown")
    except Exception:
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
