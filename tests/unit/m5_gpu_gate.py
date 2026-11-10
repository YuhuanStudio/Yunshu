"""Decides whether ``m5_gpu`` tests run on this machine (used by conftest.py and test_m5_gpu_gate.py).

M5-class means the GPU generation ``_generation`` reads from the Metal architecture string is 17 or 18
(the MetalPerformancePrimitives tensor-op kernels and their numerics exist only there). ``YUNSHU_TEST_M5_GPU``
(1 or 0) overrides the probe. Kept out of conftest.py so the unit test can import the decision directly."""

from __future__ import annotations

import os

M5_GPU_ENV = "YUNSHU_TEST_M5_GPU"
_M5_GENERATIONS = (17, 18)


def m5_gpu_allowed(architecture: str, override: str | None = None) -> bool:
    """True when the ``m5_gpu`` tests should run: an explicit override wins, else the GPU generation decides."""
    if override == "1":
        return True
    if override == "0":
        return False
    from yunshu_engine.kernels.tensorfold.lane_qmm import _generation

    return _generation(architecture) in _M5_GENERATIONS


def probe_architecture() -> str:
    """The GPU architecture string; any failure reads as "unknown", which is not M5."""
    try:
        from yunshu_engine.kernels import gpu_probe

        return str(gpu_probe.device_info().get("architecture", ""))
    except Exception:
        return ""


def m5_gpu_run_here() -> bool:
    override = os.environ.get(M5_GPU_ENV)
    if override in ("0", "1"):
        return m5_gpu_allowed("", override)
    return m5_gpu_allowed(probe_architecture())
