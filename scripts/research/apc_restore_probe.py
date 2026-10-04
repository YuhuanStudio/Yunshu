"""Memory of one follow-up turn's prefix-cache path, stage by stage, at server scale.

A 125K-token conversation turn leaves its checkpoint in the APC; the next turn restores it
(lookup copy), merges it into the batch cache, appends the new tokens, then captures and
stores the new checkpoint. This replays those steps with real mlx-vlm cache classes and the
Yunshu APC manager (attention layers only; GDN states are small) and prints active / peak
MLX memory after each stage so the stage that holds the transient copies is named.
"""

import argparse
import json

import mlx.core as mx
from mlx_vlm.apc_adapters import clone_cache_entry
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

from yunshu_engine.apc_manager import YunshuAPCManager, materialize

GIB = 1 << 30


def mem():
    mx.synchronize()
    return mx.get_active_memory() / GIB, mx.get_peak_memory() / GIB


def stage(rows, name):
    active, peak = mem()
    rows.append(dict(stage=name, active=round(active, 2), peak=round(peak, 2)))
    mx.reset_peak_memory()


def build(layers, tokens):
    # A recurrent layer first: hybrid models are never "dense trimmable" checkpoints.
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 8)), mx.ones((1, 8))]
    out = [rec]
    for _ in range(layers):
        c = KVCache()
        k = mx.random.normal((1, 4, tokens, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 4, tokens, 256)).astype(mx.bfloat16)
        mx.eval(k, v)
        c.keys, c.values, c.offset = k, v, tokens
        out.append(c)
    return out


def run(layers, tokens, new_tokens, owned):
    rows = []
    mgr = YunshuAPCManager(num_blocks=8, block_size=16, overrides={"memory_max_gb": 48})
    ids = list(range(tokens))
    live = build(layers, tokens)
    stage(rows, "live_cache_built")
    mx.reset_peak_memory()
    assert mgr.store_exact_cache(ids, live, extra_hash=0)
    stage(rows, "store_checkpoint_1")
    del live
    stage(rows, "live_released")
    ids2 = ids + list(range(10**6, 10**6 + new_tokens))
    mgr.begin_request()
    restored, prefix = mgr.lookup_exact_cache(ids2, 0, max_prefix_tokens=len(ids2) - 1)
    mx.eval([c.keys for c in restored if type(c) is KVCache])
    stage(rows, f"lookup_restore_{prefix}")
    batch = [c for c in restored if type(c) is ArraysCache]
    for c in restored:
        if type(c) is not KVCache:
            continue
        b = BatchKVCache([0])
        b.keys = c.keys.view(c.keys.dtype)
        b.values = c.values.view(c.values.dtype)
        b._idx = prefix
        b.offset += prefix
        batch.append(b)
    del restored
    stage(rows, "merged_into_batch_cache")
    for b in batch:
        if type(b) is ArraysCache:
            continue
        k = mx.random.normal((1, 4, new_tokens, 256)).astype(mx.bfloat16)
        b.update_and_fetch(k, k)
    mx.eval([b.keys for b in batch if type(b) is BatchKVCache])
    stage(rows, "suffix_appended")
    targets = []
    snap = [
        clone_cache_entry(b, min_capacity_tokens=None, eval_targets=targets)
        for b in batch
    ]
    stage(rows, "snapshot_captured_lazy")
    mgr.release_superseded(ids2, 0)
    stage(rows, "superseded_released")
    materialize([targets])
    stage(rows, "snapshot_materialized")
    mgr.store_exact_cache(ids2, snap, extra_hash=0, _owned=owned)
    stage(rows, "store_checkpoint_2")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=125000)
    ap.add_argument("--new-tokens", type=int, default=23)
    ap.add_argument("--no-owned", action="store_true")
    a = ap.parse_args(argv)
    rows = run(a.layers, a.tokens, a.new_tokens, not a.no_owned)
    for r in rows:
        print(json.dumps(r), flush=True)
    print(json.dumps(dict(complete=True)), flush=True)


if __name__ == "__main__":
    main()
