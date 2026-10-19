"""Darwin job-local swap evidence; no MLX imports and no GPU activity.

footprint --swapped attributes compressed/swapped dirty bytes to each process,
not clean file-backed mmap faults. ri_pageins is diagnostic only: loading
mmap weights legitimately increments it. All sizes in bytes unless named *_mb.
"""

from __future__ import annotations

import ctypes
import math
import re
import subprocess


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 10)]


def process_memory(pid):
    """Read one live process. Errors are evidence failures, never zero swap."""
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    usage = Usage()
    if lib.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage {pid}")
    out = subprocess.run(
        ["footprint", "--swapped", "-f", "bytes", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout
    # TOTAL columns: dirty (swapped) clean reclaimable regions TOTAL. Require
    # exact units and a complete row; permission failures/partial output fail.
    match = re.search(
        r"^\s*\d+ B\s+(\d+) B\s+\d+ B\s+\d+ B\s+(\d+)\s+TOTAL\s*$", out, re.MULTILINE
    )
    if match is None:
        raise ValueError(f"incomplete footprint for {pid}")
    after = Usage()
    if lib.proc_pid_rusage(pid, 0, ctypes.byref(after)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage after footprint {pid}")
    if usage.values[8] != after.values[8]:
        raise ValueError(f"PID reused during footprint {pid}")
    return dict(
        pid=pid,
        start=usage.values[8],
        pageins=after.values[4],
        swapped_bytes=int(match[1]),
        regions=int(match[2]),
    )


def host_available_bytes():
    """Free + inactive (not speculative/compressor); unreadable => None."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        return page * sum(
            int(re.search(r"Pages " + key + r":\s+(\d+)", out).group(1))
            for key in ("free", "inactive")
        )
    except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
        return None


def memory_verdict(host_swap_mb, job_swap_mb, available_mb, cfg):
    """Pure policy, also used by offline review. Missing evidence fails closed."""
    reasons = []
    for value, budget, reason, below in (
        (host_swap_mb, cfg["host_swap_mb"], "swap", False),
        (job_swap_mb, cfg["swap_mb"], "job_swap", False),
        (available_mb, cfg["memory_floor_mb"], "memory_pressure", True),
    ):
        if value is None or not math.isfinite(value) or value < 0:
            reasons.append(reason + "_unreadable")
        elif value < budget if below else value > budget:
            reasons.append(reason)
    return reasons
