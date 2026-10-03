import argparse
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
if a.dry_run:
    print(json.dumps(dict(dry_run=True, cases=36)))
    sys.exit()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.kernels.lane_group import grouped_linears
from yunshu_engine.kernels.lane_linear import LaneLinear
from yunshu_engine.kernels.tensorfold import lane_qmm

mx.random.seed(5)
rows = []
for bits in (4, 8):
    for gs in (32, 64):
        if not lane_qmm.readable(bits, gs):
            continue
        for k, n in ((512, 64), (2048, 512), (4096, 1024)):
            layers = [nn.Linear(k, n, bias=False), nn.Linear(k, n, bias=False)]
            for l in layers:
                l.weight = l.weight.astype(mx.bfloat16)
            layers = [
                LaneLinear.from_quantized(
                    nn.QuantizedLinear.from_linear(l, group_size=gs, bits=bits)
                )
                for l in layers
            ]
            v = SimpleNamespace(_linears=lambda ms, x: tuple(m(x) for m in ms))
            v._linears = lambda ms, x: grouped_linears(ms, x)
            for m in (1, 8, 16, 32):
                x = mx.random.normal((1, m, k)).astype(mx.bfloat16)
                ref = tuple(l(x) for l in layers)
                mx.eval(ref)
                got = v._linears(layers, x)
                assert got is not None
                mx.eval(got)
                assert all(bool(mx.array_equal(r, g)) for r, g in zip(ref, got, strict=True)), (
                    bits,
                    gs,
                    n,
                    m,
                )
                rows.append(dict(bits=bits, gs=gs, k=k, n=n, m=m, bit_equal=True))
a.output.write_text(
    "".join(json.dumps(r) + "\n" for r in rows + [dict(complete=True, success=True)])
)
print(json.dumps(dict(complete=True, success=True, cases=len(rows))), flush=True)
