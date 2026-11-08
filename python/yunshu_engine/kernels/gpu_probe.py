"""Hardware probes that always ask the GPU, whatever device the caller has made the default.

Kernel modules decide once per process what the GPU can do (lane matmul variant and self-test, NAX / M5
gates, thread limits) and cache the answer in module state. ``device_info()`` and the self-test kernels read
the *default* device, so a probe first made while the default device was the CPU (a CPU unit test, a CPU-side
tool in the same process) cached "portable / not ready" for the whole process and later GPU work silently ran
another arithmetic. Every such probe goes through here: it runs under the GPU stream when Metal exists, so
the cached answer is the hardware's, not the caller's."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import mlx.core as mx


@contextlib.contextmanager
def gpu_scope() -> Iterator[None]:
    """Default device and stream are the GPU inside the block (a no-op without Metal)."""
    try:
        has_gpu = bool(mx.metal.is_available())
    except Exception:  # pragma: no cover - an mlx without the metal module
        has_gpu = False
    if has_gpu:
        with mx.stream(mx.gpu):
            yield
    else:
        yield


def device_info() -> dict[str, Any]:
    """The GPU's ``device_info``."""
    with gpu_scope():
        if hasattr(mx, "device_info"):
            return dict(mx.device_info())
        return dict(mx.metal.device_info())  # pragma: no cover - older mlx
