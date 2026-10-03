"""macOS per-process physical footprint (Metal may be absent from ps RSS).

Layout follows sys/resource.h rusage_info_v2 in the installed macOS SDK.
Process-tree sums are accounting sums, not deduplicated physical RAM.
"""

import ctypes
import subprocess


class RusageInfoV2(ctypes.Structure):
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


def process_tree_memory(root_pid):
    rows = [
        list(map(int, line.split()))
        for line in subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss="], text=True
        ).splitlines()
    ]
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, ppid, _ in rows if ppid in selected}
        if expanded == selected:
            break
        selected = expanded
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    fn = lib.proc_pid_rusage
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    processes = []
    for pid, ppid, rss in rows:
        if pid not in selected:
            continue
        info = RusageInfoV2()
        status = fn(pid, 2, ctypes.byref(info))
        row = dict(pid=pid, ppid=ppid, rss_bytes=rss * 1024)
        if status == 0:
            row.update(
                physical_footprint_bytes=info.phys_footprint,
                resident_bytes=info.resident_size,
                wired_bytes=info.wired_size,
            )
        else:
            row["rusage_errno"] = ctypes.get_errno()
        processes.append(row)
    return dict(
        processes=processes,
        rss_sum_bytes=sum(x["rss_bytes"] for x in processes),
        physical_footprint_sum_bytes=sum(
            x.get("physical_footprint_bytes", 0) for x in processes
        ),
    )


def apc_resident_gib(url):
    """Prefix-cache (APC) bytes held in RAM, from the server's /metrics; None when
    unavailable. The prefix cache legitimately holds RAM, so a "memory returns"
    check compares footprints with it subtracted."""
    import urllib.request

    try:
        text = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10).read()
        for line in text.decode().splitlines():
            if "apc_resident_bytes" in line and not line.startswith("#"):
                return round(float(line.rsplit(" ", 1)[1]) / 2**30, 3)
    except Exception:  # noqa: BLE001
        pass
    return None


def allocator_pool_gib(url):
    """Freed MLX buffers the allocator keeps for reuse (bounded by the engine's cache limit,
    ~6 GiB on 128 GiB), from the server's /metrics; None when unavailable."""
    import urllib.request

    try:
        text = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10).read()
        for line in text.decode().splitlines():
            if line.startswith('yunshu_gpu_memory_bytes{type="cache"}'):
                return round(float(line.rsplit(" ", 1)[1]) / 2**30, 3)
    except Exception:  # noqa: BLE001
        pass
    return None


def retained_gib(url):
    """Memory the server holds on purpose and bounds itself: prefix-cache checkpoints in RAM plus
    the allocator's reuse pool. A "memory returns" check subtracts this; None without the APC
    figure (the pool counts as 0 when the server does not report it)."""
    apc = apc_resident_gib(url)
    if apc is None:
        return None
    return round(apc + (allocator_pool_gib(url) or 0.0), 3)
