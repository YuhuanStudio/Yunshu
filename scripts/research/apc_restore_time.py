"""Wall time of restoring a checkpoint into a cache with room to grow: upstream clone vs ours.

Builds a hybrid prompt cache the size of the 27B's (``--layers`` K/V layers of (1, 4, T, 256)
bf16 plus 3 recurrent layers per K/V layer), then times ``_clone_prompt_cache_for_apc`` with
``min_capacity_tokens = T + 300`` through the upstream function and through the one
``yunshu_engine.apc_manager`` installs, interleaved, and prints the median milliseconds.
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx_vlm.apc as upstream
from mlx_vlm.models.cache import ArraysCache, KVCache

from yunshu_engine import apc_manager  # noqa: F401  (installs the clone)


def build(layers, tokens):
    caches = []
    for _ in range(layers):
        for _ in range(3):
            rec = ArraysCache(2)
            rec.cache = [
                mx.random.normal((1, 3, 10240)).astype(mx.bfloat16),
                mx.random.normal((1, 48, 128, 128)).astype(mx.float32),
            ]
            caches.append(rec)
        kv = KVCache()
        k = mx.random.normal((1, 4, tokens, 256)).astype(mx.bfloat16)
        kv.keys, kv.values, kv.offset = k, k + 1, tokens
        caches.append(kv)
    mx.eval([a for c in caches for a in (c.state if hasattr(c, "state") else ())])
    return caches


def timed(fn, caches, capacity):
    mx.synchronize()
    t = time.perf_counter()
    out = fn(caches, min_capacity_tokens=capacity)
    mx.eval([a for c in out for a in (c.state if hasattr(c, "state") else ())])
    mx.synchronize()
    ms = (time.perf_counter() - t) * 1000
    del out
    return ms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args(argv)
    mine = upstream._clone_prompt_cache_for_apc
    theirs = mine._upstream
    for tokens in a.tokens:
        caches = build(a.layers, tokens)
        capacity = tokens + 300
        for _ in range(3):
            timed(theirs, caches, capacity)
            timed(mine, caches, capacity)
        t_up, t_mine = [], []
        for _ in range(a.reps):
            t_up.append(timed(theirs, caches, capacity))
            t_mine.append(timed(mine, caches, capacity))
        print(
            json.dumps(
                dict(
                    tokens=tokens,
                    upstream_ms=round(statistics.median(t_up), 2),
                    ours_ms=round(statistics.median(t_mine), 2),
                    upstream_all=[round(x, 1) for x in t_up],
                    ours_all=[round(x, 1) for x in t_mine],
                )
            ),
            flush=True,
        )
        del caches
    print(json.dumps(dict(complete=True)), flush=True)


if __name__ == "__main__":
    main()
