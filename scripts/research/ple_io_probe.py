"""Cold read throughput of the Flash-Next PLE table: what one prompt span's row reads achieve against the drive.

Reads random rows (weight 80 B + scales 10 B + biases 10 B each, as ``ple_parallel`` does) from the
checkpoint's own files with ``pread`` from T threads, over fresh random row ids each trial (rows drawn
from 32 GB: first touch is cold unless the page cache already holds them), and compares with the raw
device: the same number of random 16 KiB reads, and one sequential read. Output: JSON lines.

Usage: ple_io_probe.py MANIFEST --rows 23000 --threads 8,16,32,64,128 --trials 3 [--nocache] [--seq-gb 1]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
PAGE = 16384


def load_layout(manifest: Path) -> list[dict]:
    """Per shard: row_start, row_count and (file, offset, row_bytes) of weight / scales / biases."""
    m = json.loads(manifest.read_text())
    root = (manifest.parent / m["source_root"]).resolve()
    width = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4}
    shards = []
    for s in m["shards"]:
        tensors = {}
        for name in ("weight", "scales", "biases"):
            t = s[name]
            tensors[name] = (
                str(root / t["file"]),
                int(t["offset"]),
                int(t["shape"][1]) * width[t["dtype"]],
            )
        shards.append(
            dict(
                row_start=int(s["row_start"]), row_count=int(s["row_count"]), t=tensors
            )
        )
    return shards


def plan_reads(shards: list[dict], row_ids: np.ndarray) -> list[tuple[str, int, int]]:
    """(file, offset, bytes) of every tensor row of ``row_ids``, in row-id order."""
    starts = np.asarray([s["row_start"] for s in shards], dtype=np.int64)
    out = []
    for row_id in row_ids.tolist():
        k = int(np.searchsorted(starts, row_id, side="right")) - 1
        s = shards[k]
        local = row_id - s["row_start"]
        if not 0 <= local < s["row_count"]:
            raise IndexError(row_id)
        for path, offset, row_bytes in s["t"].values():
            out.append((path, offset + local * row_bytes, row_bytes))
    return out


def run_reads(reads, threads: int, nocache: bool, interleave: bool = True) -> float:
    """Wall seconds to pread every (file, offset, bytes) with ``threads`` threads."""
    fds: dict[str, int] = {}
    for path, _, _ in reads:
        if path not in fds:
            fds[path] = os.open(path, os.O_RDONLY)
            if nocache:
                fcntl.fcntl(fds[path], F_NOCACHE, 1)
    lanes = max(1, min(threads, len(reads)))
    chunks = (
        [reads[i::lanes] for i in range(lanes)]
        if interleave
        else [
            reads[i * len(reads) // lanes : (i + 1) * len(reads) // lanes]
            for i in range(lanes)
        ]
    )

    def work(chunk):
        buf = bytearray(max((n for _, _, n in chunk), default=1))
        view = memoryview(buf)
        for path, offset, n in chunk:
            if os.preadv(fds[path], [view[:n]], offset) != n:
                raise OSError(f"short read at {path}:{offset}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(lanes) as pool:
        for f in [pool.submit(work, c) for c in chunks]:
            f.result()
    wall = time.perf_counter() - t0
    for fd in fds.values():
        os.close(fd)
    return wall


def random_page_reads(path: str, count: int, rng) -> list[tuple[str, int, int]]:
    size = os.path.getsize(path)
    pages = rng.integers(0, size // PAGE - 1, size=count)
    return [(path, int(p) * PAGE, PAGE) for p in pages]


def sequential_mbps(path: str, gb: float, nocache: bool, skip_gb: float) -> float:
    fd = os.open(path, os.O_RDONLY)
    if nocache:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    block = 8 << 20
    total = int(gb * (1 << 30))
    offset = int(skip_gb * (1 << 30))
    buf = bytearray(block)
    t0 = time.perf_counter()
    done = 0
    while done < total:
        n = os.preadv(fd, [buf], offset + done)
        if n <= 0:
            break
        done += n
    wall = time.perf_counter() - t0
    os.close(fd)
    return done / wall / 1e6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--rows", type=int, default=23000)
    ap.add_argument("--threads", default="8,16,32,64,128")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--nocache", action="store_true")
    ap.add_argument("--seq-gb", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)
    shards = load_layout(a.manifest)
    total_rows = shards[-1]["row_start"] + shards[-1]["row_count"]
    rng = np.random.default_rng(a.seed)
    weight_file = shards[0]["t"]["weight"][0]

    def emit(**row):
        print(json.dumps(row), flush=True)

    emit(ev="seq", mbps=sequential_mbps(weight_file, a.seq_gb, a.nocache, skip_gb=0.5))
    for threads in [int(x) for x in a.threads.split(",")]:
        for trial in range(a.trials):
            ids = np.unique(rng.integers(0, total_rows, size=a.rows))
            reads = plan_reads(shards, ids)
            useful = sum(n for _, _, n in reads)
            wall = run_reads(reads, threads, a.nocache)
            emit(
                ev="rows",
                threads=threads,
                trial=trial,
                rows=len(ids),
                reads=len(reads),
                wall_s=wall,
                iops=len(reads) / wall,
                useful_mbps=useful / wall / 1e6,
            )
            raw = random_page_reads(weight_file, len(reads), rng)
            wall = run_reads(raw, threads, a.nocache)
            emit(
                ev="raw16k",
                threads=threads,
                trial=trial,
                reads=len(raw),
                wall_s=wall,
                iops=len(raw) / wall,
                mbps=len(raw) * PAGE / wall / 1e6,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
