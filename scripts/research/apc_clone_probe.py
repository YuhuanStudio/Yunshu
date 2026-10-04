"""How many physical copies does a prefix-cache checkpoint cost on its way into the APC?

Builds a KVCache-style prompt cache (``--layers`` x (1, 4, ``--tokens``, 256) bf16 K and V),
then measures MLX active memory after each stage the VLM runner's deferred-checkpoint path
runs: ``clone_cache_entry`` (the snapshot, evaluated), ``_clone_prompt_cache_for_apc`` on
that snapshot (what ``APCManager.store_exact_cache`` does again) and the release of the
snapshot. Prints one JSON line with the sizes in MiB.
"""

import argparse
import json

import mlx.core as mx
from mlx_vlm.apc import _clone_prompt_cache_for_apc
from mlx_vlm.apc_adapters import clone_cache_entry
from mlx_vlm.models.cache import BatchKVCache, KVCache

MIB = 1 << 20


def active():
    mx.synchronize()
    return mx.get_active_memory() / MIB


def build(layers, tokens):
    caches = []
    for _ in range(layers):
        c = KVCache()
        k = mx.random.normal((1, 4, tokens, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 4, tokens, 256)).astype(mx.bfloat16)
        mx.eval(k, v)
        c.keys, c.values, c.offset = k, v, tokens
        caches.append(c)
    return caches


def batch_of(caches):
    out = []
    for c in caches:
        b = BatchKVCache([0])
        b.keys, b.values = c.keys, c.values
        b._idx = c.offset
        b.offset = b.offset + c.offset
        out.append(b)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--batch", action="store_true", help="live cache is a BatchKVCache")
    a = ap.parse_args(argv)
    live = build(a.layers, a.tokens)
    if a.batch:
        live = batch_of(live)
    base = active()
    one = sum(c.keys.nbytes + c.values.nbytes for c in live) / MIB
    targets = []
    snapshot = [
        clone_cache_entry(c, min_capacity_tokens=None, eval_targets=targets)
        for c in live
    ]
    mx.eval(targets)
    after_snapshot = active()
    stored = _clone_prompt_cache_for_apc(snapshot)
    after_store = active()
    del snapshot, targets
    after_release = active()
    print(
        json.dumps(
            dict(
                layers=a.layers,
                tokens=a.tokens,
                batch=a.batch,
                cache_mib=round(one, 1),
                snapshot_mib=round(after_snapshot - base, 1),
                second_clone_mib=round(after_store - after_snapshot, 1),
                after_release_mib=round(after_release - base, 1),
                kept=len(stored),
            )
        )
    )


if __name__ == "__main__":
    main()
