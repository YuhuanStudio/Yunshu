"""Bit identity of the oMLX Qwen4 GDN plan on a real layer: window rows vs serial rows, and deferred rollback.

For several GDN layers and window sizes S = 2..8:
  serial  : S one-row calls through the plan, states carried
  window  : one S-row call under a speculative transaction, then commit keeping m = 0..S rows
  compare : outputs row by row, and the committed recurrent + conv state after m rows against the serial
            state after m rows (m = S full accept, 0 < m < S deferred replay, 0 rejects everything).
Also reports closeness to the upstream per-op layer (information only).  Run through gpuq only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--layers", type=int, default=3, help="number of GDN layers to test"
    )
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 3, 4, 8])
    ap.add_argument("--out", type=Path, required=True)
    return ap


async def run(args):
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import language as ql

    from yunshu_engine import settings
    from yunshu_engine.kernels import qwen4_gdn
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
    engine = VLMEngine(
        str(args.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
    )
    await engine.start()
    lm = engine._model.language_model
    fwd = ql._QWEN4_BATCH_INVARIANT_FORWARD
    original = type(fwd)._gated_delta
    qwen4_gdn.install_cache_patch()
    out = args.out.open("w")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    gdn_layers = [i for i, ly in enumerate(lm.model.layers) if ly.is_linear][
        : args.layers
    ]
    ok = True
    mx.random.seed(7)
    for li in gdn_layers:
        layer = lm.model.layers[li].linear_attn

        def fresh_cache():
            cache = lm.make_cache()[li]
            warm = (mx.random.normal((1, 16, 2560)) * 0.5).astype(mx.bfloat16)
            original(fwd, layer, warm, None, cache)
            mx.eval(cache[0], cache[1])
            return cache

        def plan_call(x, cache):
            return qwen4_gdn.gated_delta(fwd, layer, x, None, cache, original)

        for s in args.rows:
            x = (mx.random.normal((1, s, 2560)) * 0.5).astype(mx.bfloat16)
            serial_cache = fresh_cache()
            start_conv, start_ssm = serial_cache[0], serial_cache[1]
            serial_out, serial_states = [], []
            for r in range(s):
                serial_out.append(plan_call(x[:, r : r + 1], serial_cache))
                mx.eval(serial_out[-1], serial_cache[0], serial_cache[1])
                serial_states.append((serial_cache[0], serial_cache[1]))
            before = qwen4_gdn.counters()["engaged"]
            for keep in range(0, s + 1):
                cache = lm.make_cache()[li]
                cache[0], cache[1] = start_conv, start_ssm
                cache.start_speculation(s)
                win = plan_call(x, cache)
                mx.eval(win)
                row_equal = all(
                    bool(mx.array_equal(win[:, r : r + 1], serial_out[r]).item())
                    for r in range(s)
                )
                cache.commit_speculation([keep])
                mx.eval(cache[0], cache[1])
                if keep == 0:
                    ref_conv, ref_ssm = start_conv, start_ssm
                else:
                    ref_conv, ref_ssm = serial_states[keep - 1]
                state_equal = bool(mx.array_equal(cache[1], ref_ssm).item())
                conv_equal = bool(mx.array_equal(cache[0], ref_conv).item())
                good = row_equal and state_equal and conv_equal
                ok = ok and good
                emit(
                    {
                        "kind": "identity",
                        "layer": li,
                        "rows": s,
                        "keep": keep,
                        "rows_equal": row_equal,
                        "ssm_equal": state_equal,
                        "conv_equal": conv_equal,
                    }
                )
            plan_calls = qwen4_gdn.counters()["engaged"] - before
            # closeness to the upstream per-op layer, information only
            up_cache = lm.make_cache()[li]
            up_cache[0], up_cache[1] = start_conv, start_ssm
            up = original(fwd, layer, x, None, up_cache)
            mx.eval(up)
            diff = mx.abs(
                up.astype(mx.float32)
                - mx.concatenate(serial_out, axis=1).astype(mx.float32)
            )
            emit(
                {
                    "kind": "vs_upstream",
                    "layer": li,
                    "rows": s,
                    "max_abs": float(mx.max(diff).item()),
                    "exact_fraction": float(
                        mx.mean((diff == 0).astype(mx.float32)).item()
                    ),
                    "plan_calls": plan_calls,
                }
            )
    counters = qwen4_gdn.counters()
    emit({"kind": "counters", **counters})
    ok = ok and counters["engaged"] > 0
    emit({"complete": ok})
    out.close()
    return 0 if ok else 1


def main(argv=None):
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
