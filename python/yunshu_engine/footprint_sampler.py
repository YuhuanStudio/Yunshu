"""In-process process-footprint sampler: peak phys_footprint at a 20 ms cadence.

A harness that polls from outside (ps + proc_pid_rusage every 200 ms) misses short spikes
(prefill activations, a checkpoint copy) and is noisy between runs. This thread reads the
server's own ``phys_footprint`` through ``proc_pid_rusage`` (one syscall, no subprocess) and
keeps the peak since start; ``/metrics`` exports it as ``yunshu_process_footprint_bytes``.
Enabled by ``YUNSHU_FOOTPRINT_SAMPLE_MS`` (0 = off).
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Callable


class _RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in [
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        ]
    ]


def self_footprint_bytes() -> int:
    """phys_footprint of this process in bytes (0 when unavailable, e.g. not macOS)."""
    try:
        fn = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        info = _RusageV2()
        if fn(os.getpid(), 2, ctypes.byref(info)) != 0:
            return 0
        return int(info.phys_footprint)
    except (OSError, AttributeError):
        return 0


class FootprintSampler:
    def __init__(
        self, interval_s: float = 0.02, read: Callable[[], int] = self_footprint_bytes
    ):
        self.interval_s = interval_s
        self._read = read
        self.peak = 0
        self.current = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> int:
        v = int(self._read())
        self.current = v
        self.samples += 1
        if v > self.peak:
            self.peak = v
        return v

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> FootprintSampler:
        if self._thread is None:
            self.sample_once()
            self._thread = threading.Thread(
                target=self._run, name="footprint-sampler", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None


_SAMPLER: FootprintSampler | None = None


def start_from_settings() -> FootprintSampler | None:
    """Start the process-wide sampler when YUNSHU_FOOTPRINT_SAMPLE_MS > 0 (idempotent)."""
    global _SAMPLER
    from . import settings

    ms = int(settings.get("YUNSHU_FOOTPRINT_SAMPLE_MS") or 0)
    if ms > 0 and _SAMPLER is None:
        _SAMPLER = FootprintSampler(ms / 1000.0).start()
    return _SAMPLER


def current() -> FootprintSampler | None:
    return _SAMPLER


def metric_lines() -> list[str]:
    s = _SAMPLER
    if s is None:
        return []
    return [
        "",
        "# HELP yunshu_process_footprint_bytes Process phys_footprint (in-process sampler)",
        "# TYPE yunshu_process_footprint_bytes gauge",
        f'yunshu_process_footprint_bytes{{type="current"}} {s.current}',
        f'yunshu_process_footprint_bytes{{type="peak"}} {s.peak}',
        f"yunshu_process_footprint_samples_total {s.samples}",
    ]
