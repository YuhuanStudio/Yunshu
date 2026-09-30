"""Sequential write / read speed of a directory's volume, page cache bypassed (F_NOCACHE).

    disk_speed.py DIR [--gib 2]

Writes one file of the given size, reads it back, deletes it. The APC SSD tier reads a
checkpoint as a few large safetensors files, so sequential throughput is what matters.
"""

import argparse
import fcntl
import os
import time
from pathlib import Path

F_NOCACHE = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--gib", type=float, default=2.0)
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "disk_speed.bin"
    chunk = os.urandom(8 << 20)
    n = int(a.gib * (1 << 30) / len(chunk))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    t0 = time.time()
    for _ in range(n):
        os.write(fd, chunk)
    os.fsync(fd)
    w = time.time() - t0
    os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    t0 = time.time()
    total = 0
    while True:
        b = os.read(fd, 8 << 20)
        if not b:
            break
        total += len(b)
    r = time.time() - t0
    os.close(fd)
    path.unlink()
    gib = total / (1 << 30)
    print(f"{d}: write {gib / w:.2f} GiB/s  read {gib / r:.2f} GiB/s  ({gib:.1f} GiB)")


if __name__ == "__main__":
    main()
