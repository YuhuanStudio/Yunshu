"""Bytes/s of the APC disk writer for one synthetic exact checkpoint (idle time needed per size)."""

import shutil
import sys
import tempfile
import time

import mlx.core as mx
from mlx_vlm.models.cache import KVCache

from yunshu_engine.apc_manager import SpillDiskStore, YunshuAPCManager


def main(mb: int, reps: int = 2) -> None:
    n = 2048
    heads, dim = 4, 128
    layers = max(1, mb * (1 << 20) // (2 * 2 * heads * dim * n))
    cache = []
    for _ in range(layers):
        kv = KVCache()
        kv.update_and_fetch(
            mx.ones((1, heads, n, dim), dtype=mx.bfloat16),
            mx.ones((1, heads, n, dim), dtype=mx.bfloat16),
        )
        cache.append(kv)
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    size = layers * 2 * 2 * heads * dim * n
    root = tempfile.mkdtemp(prefix="apcrate-", dir=".")
    try:
        disk = SpillDiskStore(root, namespace="t", num_workers=1, max_bytes=1 << 40)
        m = YunshuAPCManager(
            num_blocks=8,
            block_size=16,
            disk=disk,
            overrides={"memory_max_gb": 64},
            head_marker=(900, 901),
            persist_on_store=True,
        )
        for r in range(reps):
            ids = list(range(r * 10_000, r * 10_000 + n))
            m.begin_request()
            t = time.perf_counter()
            m.store_exact_cache(ids, cache)
            disk.flush()
            dt = time.perf_counter() - t
            print(
                f"checkpoint {size / 2**20:.0f} MiB: {dt:.3f} s = {size / dt / 2**20:.0f} MiB/s"
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
