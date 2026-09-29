"""Driver prefill tok/s at one length under prefill knobs: MLX cache bound,
evaluation cadence (layers), idle budget (tokens a prefill step takes when no
row decodes), chunk. One model load, one run per config.

    python scripts/research/probe_prefill_cfg.py <ckpt> --length 32768
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

G = 2**30
CONFIGS = [  # cache GiB, eval every, idle budget, chunk
    (8, 1, 2048, 512),
    (2, 4, 4096, 512),
    (8, 4, 4096, 512),
    (8, 8, 4096, 512),
    (8, 4, 2048, 512),
    (8, 4, 4096, 1024),
    (8, 4, 4096, 2048),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--length", type=int, default=32768)
    a = ap.parse_args()
    from mlx_vlm import load as vlm_load

    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.round_driver import driver as drv
    from yunshu_engine.round_driver import forward as fwd

    model, _ = vlm_load(a.ckpt)
    lm = model.language_model
    lane_linear.convert(lm)
    if lm.args.tie_word_embeddings:
        lm._yunshu_lane_head = lane_linear.lane_head(lm.model.embed_tokens)
    rng = random.Random(0)
    ids = [rng.randrange(1000, 20000) for _ in range(a.length)]
    for cache, every, budget, chunk in CONFIGS:
        mx.set_cache_limit(64 * G)
        mx.clear_cache()
        drv.CACHE_LIMIT = cache * G
        fwd.EVAL_EVERY = every
        drv.IDLE_BUDGET = budget
        d = drv.RoundDriver(model, chunk=chunk)
        d.add(drv.Request(ids, 1, handle=0))
        mx.synchronize()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        while d.busy():
            d.step()
        mx.synchronize()
        dt = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "cache_gib": cache,
                    "eval_every": every,
                    "idle_budget": budget,
                    "chunk": chunk,
                    "tok_s": round(a.length / dt, 1),
                    "peak_gib": round(mx.get_peak_memory() / G, 1),
                }
            ),
            flush=True,
        )


main()
