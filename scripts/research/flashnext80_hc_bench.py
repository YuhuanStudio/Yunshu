"""TensorFold's fused hyper-connection chain vs Yunshu's, on the real weights of one layer.

Per call the chain is: write the previous block's branch back into the 4 residual streams (inject), then
norm -> down projection -> SiLU -> up projection -> sigmoid mix (the next block's input) and the inject gates.
Yunshu runs that as ~14 small kernels; TensorFold's hc_norm + hc_project (derived files under
python/yunshu_engine/kernels/tf_flash_next) run it as 3.

Reports (a) how close the TF outputs are to Yunshu's (exact-equal fraction, max abs diff) and (b) the time of a
chain of ``--calls`` dependent HC calls (decode runs 97 per step) for each implementation.
Run through gpuq only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path


def summarize_diff(ours, theirs):
    """(exact fraction, max abs diff, max relative-to-scale diff) between two arrays of equal shape."""
    import mlx.core as mx

    a = ours.astype(mx.float32).reshape(-1)
    b = theirs.astype(mx.float32).reshape(-1)
    diff = mx.abs(a - b)
    scale = float(mx.max(mx.abs(a)).item()) or 1.0
    return {
        "exact_fraction": float(mx.mean((a == b).astype(mx.float32)).item()),
        "max_abs": float(mx.max(diff).item()),
        "max_rel_to_max": float(mx.max(diff).item()) / scale,
    }


async def run(args):
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import language as ql

    from yunshu_engine import settings
    from yunshu_engine.kernels.tf_flash_next import hc as tf_hc
    from yunshu_engine.kernels.tf_flash_next.base import QWeights
    from yunshu_engine.kernels.tf_flash_next.inputs import floats
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
    engine = VLMEngine(
        str(args.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
    )
    await engine.start()
    lm = engine._model.language_model
    fwd = ql._QWEN4_BATCH_INVARIANT_FORWARD
    out = args.out.open("w")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    layer = lm.model.layers[args.layer]
    hc = layer.attn_hyper_connection
    streams, dims, low = (
        hc.hc_count,
        hc.hidden_size,
        hc.input_mix_weight_down.weight.shape[0],
    )
    down = hc.input_mix_weight_down
    emit(
        {
            "kind": "weights",
            "layer": args.layer,
            "down": [down.bits, down.group_size, list(down.weight.shape)],
            "up": [
                hc.input_mix_weight_up.bits,
                hc.input_mix_weight_up.group_size,
                list(hc.input_mix_weight_up.weight.shape),
            ],
            "inject": [
                hc.block_inject_weight.bits,
                hc.block_inject_weight.group_size,
                list(hc.block_inject_weight.weight.shape),
            ],
        }
    )
    down_q = QWeights.of(hc.input_mix_weight_down, hc.block_inject_weight)
    up_q = QWeights.of(hc.input_mix_weight_up)
    nw = (1.0 + hc.hc_norm.weight.astype(mx.float32)).reshape(-1)
    eps = floats([hc.hc_norm.eps])
    mx.random.seed(11)

    def tf_step(h, branch, inj):
        hn, ssp = tf_hc.hc_norm(
            h, streams=streams, write_back="plain", branch=(branch,), inject=inj
        )
        mixed, inj2 = tf_hc.hc_project(
            hn, ssp, down_q, up_q, nw, eps=eps, streams=streams, low=low
        )
        return hn, mixed, inj2

    def our_step(h, branch, inj):
        hidden = fwd._inject(
            branch.reshape(1, 1, -1), h.reshape(1, 1, -1), inj.reshape(1, 1, -1)
        )
        mixed, hyper_input, injection = fwd._hyper_connection(hc, hidden)
        return hidden.reshape(1, -1), mixed.reshape(1, -1), injection.reshape(1, -1)

    for rows in args.rows:
        h = (mx.random.normal((rows, streams * dims)) * 0.5).astype(mx.bfloat16)
        branch = (mx.random.normal((rows, dims)) * 0.2).astype(mx.bfloat16)
        inj = (mx.random.uniform(shape=(rows, streams)) * 2.0).astype(mx.bfloat16)
        if rows == 1:
            ours = our_step(h, branch, inj)
            theirs = tf_step(h, branch, inj)
            mx.eval(ours, theirs)
            emit(
                {
                    "kind": "numerics",
                    "rows": rows,
                    "hidden": summarize_diff(ours[0], theirs[0]),
                    "mixed": summarize_diff(ours[1], theirs[1]),
                    "inject": summarize_diff(ours[2], theirs[2][:rows]),
                }
            )

        def chain(step):
            hh, br, ij = h, branch, inj
            for _ in range(args.calls):
                hh, mixed, ij = step(hh, br, ij)
                ij = ij[:rows]
                br = mixed
            return hh, br

        results = {}
        for name, step in (
            ("yunshu", our_step if rows == 1 else None),
            ("tf", tf_step),
        ) * args.rounds:
            if step is None:
                continue
            mx.eval(chain(step))  # warm
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                mx.eval(chain(step))
                times.append((time.perf_counter() - t0) * 1e3)
            results.setdefault(name, []).append(statistics.median(times))
        emit(
            {
                "kind": "chain",
                "rows": rows,
                "calls": args.calls,
                "ms_per_chain": {
                    k: [round(x, 3) for x in v] for k, v in results.items()
                },
                "us_per_call": {
                    k: round(statistics.median(v) / args.calls * 1e3, 1)
                    for k, v in results.items()
                },
            }
        )
    emit({"complete": True})
    out.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--calls", type=int, default=97)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
