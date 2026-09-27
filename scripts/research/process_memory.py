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
