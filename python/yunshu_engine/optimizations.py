"""Hardware detection and optimization status (oMLX pattern).

Re-exports hardware detection and provides get_optimization_status()
for the admin dashboard. MLX already includes optimized SDPA and
Metal kernels — no additional runtime patching needed.
"""

import logging

from .utils.hardware import (
    HardwareInfo,
    detect_hardware,
    format_bytes,
    get_system_memory_gb,
)

try:
    import mlx.core as mx
except ImportError:
    mx = None

logger = logging.getLogger(__name__)

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "get_system_memory_gb",
    "get_optimization_status",
]


def get_optimization_status() -> dict:
    """Get current hardware and MLX status for dashboard."""
    hw = detect_hardware()

    mlx_mem = {}
    if mx is not None:
        try:
            mlx_mem = {
                "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }
        except Exception:
            logger.debug("MLX memory stats read failed", exc_info=True)

    flash_available = hasattr(mx, "fast") and hasattr(
        mx.fast, "scaled_dot_product_attention"
    ) if mx else False

    return {
        "hardware": {
            "chip": hw.chip_name,
            "total_memory_gb": hw.total_memory_gb,
            "gpu_cores": hw.gpu_cores,
            "device_name": hw.mlx_device_name,
            "os_version": hw.os_version,
        },
        "mlx_memory": mlx_mem,
        "mlx_lm_features": {
            "flash_attention": "built-in" if flash_available else "not available",
            "metal_kernels": "optimized for Apple Silicon",
            "kv_cache": "managed by mlx-lm",
            "quantization": "4-bit and 8-bit supported",
        },
    }
