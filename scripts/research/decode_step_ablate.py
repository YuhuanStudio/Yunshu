"""Whole-model decode-step cost of the lane matmul's parts (run only via gpuq).

One 27B-class Qwen3.5 checkpoint, a prefilled cache, T-row decode steps pipelined the way BatchGenerator
runs them (async_eval on the next step before reading the previous). Arms change ONE thing per run, so
the difference to ``lane`` is that thing's share of the step; outputs of the ablated arms are not
meaningful (they only time the kernels):

  lane         the served configuration
  no_xsum      the per-input group-sum kernel replaced by a cached zero buffer
  sk4 / sk8    every projection's K split forced to 4 / 8 slices (changes bits)

    python scripts/research/decode_step_ablate.py MODEL_DIR --rows 1 4 8 --output out.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ARMS = ("lane", "no_xsum", "sk4", "sk8")


def summarize(samples: list[float]) -> dict:
    """Median / min / max in milliseconds of per-step times."""
    ms = [s * 1e3 for s in samples]
    return {
        "median_ms": statistics.median(ms),
        "min_ms": min(ms),
        "max_ms": max(ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_vlm import load

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from yunshu_engine.kernels import lane_linear, omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active
    from yunshu_engine.kernels.tensorfold import lane_qmm

    model, processor = load(a.model_dir)
    lm = model.language_model
    omlx.apply(row_exact=False)
    lane_linear.convert(lm)
    install_invariant(lm, model=model, packed=False)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    tok = processor.tokenizer
    filler = "".join(
        f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(4000)
    )
    ids_all = tok.encode(filler, add_special_tokens=False)

    real_kernel = lane_qmm._kernel
    real_split = lane_qmm.split_k
    zeros: dict = {}

    def fake_kernel(name):
        k = real_kernel(name)
        if name != "xsum":
            return k

        def call(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            shape = tuple(output_shapes[0])
            if shape not in zeros:
                zeros[shape] = mx.zeros(shape, dtype=mx.float32)
                mx.eval(zeros[shape])
            return [zeros[shape]]

        return call

    def configure(arm: str) -> None:
        lane_qmm._kernel = fake_kernel if arm == "no_xsum" else real_kernel
        lane_qmm.split_k = (
            (lambda n, k, s=int(arm[2:]): s) if arm in ("sk4", "sk8") else real_split
        )
        lane_qmm._xs_cache.clear()
        set_active(True)

    def fresh_cache(ctx):
        cache = lm.make_cache()
        ids = mx.array(ids_all[:ctx])[None]
        for s in range(0, ctx, 2048):
            lm(ids[:, s : s + 2048], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")
    for rep in range(a.reps):
        for t in a.rows:
            for arm in a.arms:
                configure(arm)
                cache = fresh_cache(a.context)
                toks = mx.array(ids_all[a.context : a.context + t])[None]

                def step():
                    return mx.argmax(lm(toks, cache=cache).logits[:, -1], axis=-1)

                for _ in range(3):
                    mx.eval(step())
                times = []
                nxt = step()
                mx.async_eval(nxt)
                for _ in range(a.steps):
                    t0 = time.perf_counter()
                    cur = nxt
                    nxt = step()
                    mx.async_eval(nxt)
                    mx.eval(cur)
                    times.append(time.perf_counter() - t0)
                mx.eval(nxt)
                row = {
                    "rep": rep,
                    "rows": t,
                    "arm": arm,
                    "context": a.context,
                    **summarize(times),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(row, flush=True)
    print("complete: true")


if __name__ == "__main__":
    main()
