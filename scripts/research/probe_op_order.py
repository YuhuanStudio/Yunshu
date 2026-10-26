"""Which summation order does the M5 tensor op (matmul2d, bf16 x uint4 -> fp32, K = 64) use? (gpuq only)

The lane matmul's bits are defined by that op: a group's 64 products are summed inside the matrix unit, then
``fma(scale, P, fma(bias, xsum, C))``. A scalar kernel that reproduces P bit for bit would let rows 1-4 stream
weights like MLX's qmv does while 16-row verify keeps the tensor op, with spec on == spec off intact. This runs
the op on random data, writes P in fp32, and compares it with CPU emulations of candidate orders (all products
are exact in fp32: bf16 x 4-bit integer).

    probe_op_order.py --out result.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  constexpr auto desc = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(64, 16));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b((device uchar*)Wq, dextents<int32_t, 2>(64, 32));
  auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  op.run(a, b, P);
  for (int i = 0; i < 16 * 32 / 32; i++) {
    auto ids = P.get_multidimensional_index(i);
    Out[ids[1] * 32 + ids[0]] = P[i];
  }
"""

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""


def pack_codes(codes):
    """codes (32 cols, 64) of 0..15 -> the tile layout's bytes: a column's 32 bytes, low nibble first."""
    low, high = codes[:, 0::2], codes[:, 1::2]
    return (low | (high << 4)).astype(np.uint8).reshape(-1)


def products(x, codes):
    """Exact fp32 products x[r, k] * q[c, k]: array (16, 32, 64)."""
    return (
        x[:, None, :].astype(np.float32) * codes[None, :, :].astype(np.float32)
    ).astype(np.float32)


def seq(p):
    acc = np.zeros(p.shape[:2], dtype=np.float32)
    for k in range(p.shape[2]):
        acc = (acc + p[:, :, k]).astype(np.float32)
    return acc


def pairwise(p):
    cur = p
    while cur.shape[2] > 1:
        cur = (cur[:, :, 0::2] + cur[:, :, 1::2]).astype(np.float32)
    return cur[:, :, 0]


def blocks(p, size, inner):
    parts = []
    for s in range(0, p.shape[2], size):
        parts.append(inner(p[:, :, s : s + size]))
    acc = parts[0]
    for q in parts[1:]:
        acc = (acc + q).astype(np.float32)
    return acc


def exact(p):
    """Correctly rounded sum of the exact products (fp64 holds 64 of these without error for the data used)."""
    return p.astype(np.float64).sum(axis=2).astype(np.float32)


def strided(p, lanes):
    """`lanes` interleaved partial sums (k % lanes), then added in order."""
    parts = [seq(p[:, :, i::lanes]) for i in range(lanes)]
    acc = parts[0]
    for q in parts[1:]:
        acc = (acc + q).astype(np.float32)
    return acc


HYPOTHESES = {
    "sequential": seq,
    "pairwise": pairwise,
    "exact_rounded": exact,
    **{f"blocks{s}_seq": (lambda p, s=s: blocks(p, s, seq)) for s in (4, 8, 16, 32)},
    **{
        f"blocks{s}_pair": (lambda p, s=s: blocks(p, s, pairwise))
        for s in (4, 8, 16, 32)
    },
    **{f"strided{n}": (lambda p, n=n: strided(p, n)) for n in (2, 4, 8, 16)},
}


def compare(p_hw, x, codes):
    """Fraction of the 512 outputs each hypothesis reproduces bit for bit."""
    prod = products(x, codes)
    return {
        name: float(np.mean(fn(prod).view(np.uint32) == p_hw.view(np.uint32)))
        for name, fn in HYPOTHESES.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    kern = mx.fast.metal_kernel(
        name="op_order",
        input_names=["X", "Wq"],
        output_names=["Out"],
        source=SOURCE,
        header=HEADER,
    )
    rng = np.random.default_rng(0)
    totals = {name: 0.0 for name in HYPOTHESES}
    kinds = ("normal", "wide", "int")
    for trial in range(a.trials):
        kind = kinds[trial % len(kinds)]
        if kind == "normal":
            x = rng.standard_normal((16, 64)).astype(np.float32)
        elif kind == "wide":
            x = (
                rng.standard_normal((16, 64)) * np.exp(rng.uniform(-6, 6, (16, 64)))
            ).astype(np.float32)
        else:
            x = rng.integers(-8, 9, (16, 64)).astype(np.float32)
        xb = np.asarray(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
        codes = rng.integers(0, 16, (32, 64))
        wbytes = mx.array(pack_codes(codes))
        out = kern(
            inputs=[mx.array(x).astype(mx.bfloat16), wbytes],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(16 * 32,)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(out)
        p_hw = np.asarray(out).reshape(16, 32)
        scores = compare(p_hw, xb, codes)
        for name, v in scores.items():
            totals[name] += v / a.trials
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    row = {"trials": a.trials, "match_fraction": dict(ranked)}
    with a.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    for name, v in ranked:
        print(f"{name:16s} {v:.4f}")


if __name__ == "__main__":
    main()
