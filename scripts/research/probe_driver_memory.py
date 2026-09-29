"""Memory of the round driver by stage: engine load (driver on / off), then
N rows decoding at a given context, MLX active / cache / peak in GiB.

    YUNSHU_ROUND_DRIVER=1 python scripts/research/probe_driver_memory.py <ckpt> --rows 8 --context 1024
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

G = 2**30


def mem(tag):
    mx.synchronize()
    row = {
        "stage": tag,
        "active": round(mx.get_active_memory() / G, 2),
        "cache": round(mx.get_cache_memory() / G, 2),
        "peak": round(mx.get_peak_memory() / G, 2),
    }
    print(json.dumps(row), flush=True)


def parts(eng, d):
    """Resident bytes by owner (GiB): model parameters, the decode batch's
    slot buffers / GDN state, the MTP head's slots and weights."""
    from mlx.utils import tree_flatten

    def nbytes(tree):
        return sum(a.nbytes for _, a in tree_flatten(tree) if hasattr(a, "nbytes"))

    lm = eng._model.language_model
    row = {
        "stage": "parts",
        "lm_params": round(nbytes(lm.parameters()) / G, 2),
        "model_params": round(nbytes(eng._model.parameters()) / G, 2),
        "batch_slots": round(sum(a.nbytes for a in d.batch.slots.arrays()) / G, 2),
        "batch_state": round(
            sum(a.nbytes for a in d.batch.state + d.batch.conv if a is not None) / G, 2
        ),
        "S": d.batch.slots.S,
        "cap": d.batch.slots.cap,
    }
    if d.head is not None:
        row["head_slots"] = round(sum(a.nbytes for a in d.head.slots.arrays()) / G, 2)
        row["drafter"] = round(nbytes(d.head.drafter.parameters()) / G, 2)
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    from yunshu_engine.round_driver.driver import Request
    from yunshu_engine.vlm_engine import VLMEngine

    eng = VLMEngine(a.ckpt)
    eng.load()
    mem("loaded")
    runner = eng._batch_runner
    d = runner.driver
    if d is None:
        print("driver off")
        return
    rnd = random.Random(0)
    for r in range(a.rows):
        ids = [rnd.randrange(1000, 20000) for _ in range(a.context)]
        d.add(Request(ids, a.tokens, handle=r))
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    steps = 0
    while d.busy():
        d.step()
        steps += 1
        if steps == 10:
            mem("after 10 steps")
            parts(eng, d)
    mem("done")
    parts(eng, d)
    print("wall", round(time.perf_counter() - t0, 1))
    mx.clear_cache()
    mem("after clear_cache")


main()
