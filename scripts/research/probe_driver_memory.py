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


def sequence(eng, d):
    """The bench_context_batch order: 1K, 32K, then 2/4/8 concurrent 1K, with
    the APC's resident bytes after each phase."""
    import uuid

    from bench_context_batch import CORPUS
    from yunshu_engine.round_driver.driver import Request

    tok = eng._processor.tokenizer
    base = tok.encode(CORPUS.read_text() * 4)
    apc = eng._apc_backend

    def prompt(n):
        return (tok.encode(f"BENCH-{uuid.uuid4().hex} ") + base)[:n]

    def phase(name, n_rows, ctx):
        mx.reset_peak_memory()
        for r in range(n_rows):
            d.add(Request(prompt(ctx), 64, handle=r))
        while d.busy():
            d.step()
        mx.synchronize()
        print(
            json.dumps(
                {
                    "phase": name,
                    "active": round(mx.get_active_memory() / G, 2),
                    "cache": round(mx.get_cache_memory() / G, 2),
                    "peak": round(mx.get_peak_memory() / G, 2),
                    "apc_resident": round(apc.resident_bytes() / G, 2) if apc else None,
                }
            ),
            flush=True,
        )

    phase("1k", 1, 1024)
    phase("32k", 1, 32768)
    for n in (2, 4, 8):
        phase(f"b{n}", n, 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--sequence", action="store_true")
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
    if a.sequence:
        sequence(eng, d)
        return
    rnd = random.Random(0)
    tok = eng._processor.tokenizer
    if a.text:
        from bench_context_batch import CORPUS

        corpus = CORPUS.read_text()
        base = tok.encode(corpus * 4)
    for r in range(a.rows):
        if a.text:  # distinct prefix per row, then the code corpus
            head = tok.encode(f"BENCH-{r}-{rnd.random()} ")
            ids = (head + base)[: a.context]
        else:
            ids = [rnd.randrange(1000, 20000) for _ in range(a.context)]
        d.add(Request(ids, a.tokens, handle=r))
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    steps = 0
    top = 0.0
    while d.busy():
        d.step()
        steps += 1
        cur = (mx.get_active_memory() + mx.get_cache_memory()) / G
        if cur > top + 0.5:
            top = cur
            print(
                json.dumps(
                    {
                        "step": steps,
                        "rows_in_batch": len(d.batch.rows),
                        "active+cache": round(cur, 2),
                        "peak": round(mx.get_peak_memory() / G, 2),
                    }
                ),
                flush=True,
            )
        if steps == 40:
            mem("after 40 steps")
            parts(eng, d)
    mem("done")
    parts(eng, d)
    print("wall", round(time.perf_counter() - t0, 1))
    mx.clear_cache()
    mem("after clear_cache")


main()
