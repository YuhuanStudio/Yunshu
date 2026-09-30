"""Round driver prefill cost vs context length (one row, one output token).

Times ``RoundDriver`` prefilling a single prompt of N tokens to its first
token, next to a plain chunked prefill (the decoder over its own cache,
2048-token chunks, evaluated per chunk) of the same model with stock
quantized projections and with lane projections. Linear growth in N is
expected; the driver growing faster than the plain prefill (or its peak memory
growing with N beyond the KV cache) points at unevaluated graphs carried
across steps, and lane vs stock plain isolates the projection kernel.

    python scripts/research/probe_driver_prefill.py \
        ~/models/Qwen3.5-0.8B-MLX-bf16 --quantize \
        --lengths 2048 8192 16384 32768
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def driver_prefill(model, ids):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(model)
    d.add(Request(ids, 1, handle=0))
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    steps = 0
    while d.busy():
        d.step()
        steps += 1
    mx.synchronize()
    return time.perf_counter() - t0, steps, mx.get_peak_memory() / 2**30


def plain_prefill(model, ids, chunk=2048):
    lm = model.language_model
    cache = lm.make_cache()
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for s in range(0, len(ids), chunk):
        out = lm(mx.array(ids[s : s + chunk])[None], cache=cache)
        logits = getattr(out, "logits", out)
        mx.eval(logits[:, -1], [c.state for c in cache])
    mx.synchronize()
    return time.perf_counter() - t0, mx.get_peak_memory() / 2**30


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="*", default=[2048, 8192, 16384])
    ap.add_argument("--chunks", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    import mlx.nn as nn
    from mlx_vlm import load as vlm_load

    from yunshu_engine.kernels import lane_linear

    model, _ = vlm_load(a.ckpt)
    lm = model.language_model
    if a.quantize:
        nn.quantize(
            lm,
            group_size=64,
            bits=4,
            class_predicate=lambda _p, mod: (
                isinstance(mod, nn.Linear) and mod.weight.shape[-1] % 64 == 0
            ),
        )
    rng = __import__("random").Random(0)
    prompts = {n: [rng.randrange(1000, 20000) for _ in range(n)] for n in a.lengths}
    chunks = a.chunks
    stock = {
        (n, c): plain_prefill(model, ids, c)
        for n, ids in prompts.items()
        for c in chunks
    }
    lane_linear.convert(lm)
    if lm.args.tie_word_embeddings:
        lm._yunshu_lane_head = lane_linear.lane_head(lm.model.embed_tokens)
    for n, ids in prompts.items():
        dt, steps, peak = driver_prefill(model, ids)
        lane = {c: plain_prefill(model, ids, c) for c in chunks}
        pt, ppeak = lane[chunks[-1]]
        st, speak = stock[(n, chunks[-1])]
        row = {
            "tokens": n,
            "stock_plain_s": round(st, 2),
            "stock_plain_tok_s": round(n / st, 1),
            "stock_plain_peak_gib": round(speak, 2),
            "driver_s": round(dt, 2),
            "driver_steps": steps,
            "driver_tok_s": round(n / dt, 1),
            "driver_peak_gib": round(peak, 2),
            "lane_plain_s": round(pt, 2),
            "lane_plain_tok_s": round(n / pt, 1),
            "lane_plain_peak_gib": round(ppeak, 2),
            "by_chunk_tok_s": {
                str(c): {
                    "stock": round(n / stock[(n, c)][0], 1),
                    "lane": round(n / lane[c][0], 1),
                }
                for c in chunks
            },
        }
        print(json.dumps(row), flush=True)
        if a.output:
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
