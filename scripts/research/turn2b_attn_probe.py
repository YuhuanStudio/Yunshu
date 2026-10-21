"""Diagnostic short-tail attention parity; run through gpuq."""

import argparse
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
if a.dry_run:
    print("rows=532,535,640,1023; keys=33300; stock vs BQ32")
    raise SystemExit(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import mlx.core as mx

from yunshu_engine.kernels import attn_prefill as ap

mx.random.seed(1022)
for rows in (532, 535, 640, 1023):
    q = mx.random.normal((1, rows, 24, 256)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    k = mx.random.normal((1, 4, 33300, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=256**-0.5, mask="causal")
    got = ap.attend(q, k, v)
    mx.eval(ref, got)
    print(
        json.dumps(
            dict(
                rows=rows,
                equal=bool(mx.array_equal(ref, got).item()),
                mismatch=int(mx.sum(ref != got).item()),
            )
        ),
        flush=True,
    )
print(json.dumps(dict(complete=True)), flush=True)
