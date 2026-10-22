"""Compare native QMM M tiles without changing reduction order (gpuq only)."""

import argparse
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
if a.dry_run:
    print("532 rows; M128/640 vs M64/576; 4/5/8 bits; 3 projection shapes")
    raise SystemExit(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.kernels import nax_prefill as np
from yunshu_engine.kernels.lane_linear import LaneLinear

mx.random.seed(1022)
assert np.enable()
for k, n in ((5120, 17408), (17408, 5120), (5120, 12288)):
    for bits in (4, 5, 8):
        linear = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
        lane = LaneLinear.from_quantized(linear)
        x = mx.random.normal((532, k)).astype(mx.bfloat16)
        mx.eval(lane.parameters(), x)
        funcs = []
        for bm, m in ((128, 640), (64, 576)):
            xp = mx.pad(x, [(0, m - 532), (0, 0)])
            mx.eval(xp)
            funcs.append(
                np.make(
                    mx,
                    xp,
                    lane.weight,
                    None,
                    None,
                    bits=bits,
                    bm=bm,
                    bn=64,
                    sbt=lane.sbt,
                )
            )
        ref, got = [f()[:532] for f in funcs]
        mx.eval(ref, got)
        times = [[], []]
        for _ in range(6):
            for i in (0, 1, 1, 0):
                t = time.perf_counter()
                mx.eval(funcs[i]())
                mx.synchronize()
                times[i].append(time.perf_counter() - t)
        print(
            json.dumps(
                dict(
                    k=k,
                    n=n,
                    bits=bits,
                    equal=bool(mx.array_equal(ref, got).item()),
                    ms=[1000 * sum(v) / len(v) for v in times],
                )
            ),
            flush=True,
        )
        del lane, linear, funcs, ref, got, x
        mx.clear_cache()
print(json.dumps(dict(complete=True)), flush=True)
