"""Memory-side ceiling of the lane kernel's access pattern, without the tensor op (gpuq only).

The lane matmul gives each simdgroup one K slice of one 32-column tile: per group it reads a 1 KB contiguous
tile block (lane l: its column's 32 bytes), then multiplies. This kernel keeps exactly that grid, slice split
and addressing but only loads (``--depth`` groups in flight per simdgroup) and xors, so its GB/s is what the
memory system gives that pattern. Compare with the lane kernel's own GB/s at the same shape: the gap is what
the op, the epilogue and the dependent load-multiply chain cost.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int tile = threadgroup_position_in_grid.x;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  const device uint4* W4 = (const device uint4*)W + (int64_t)tile * KG * (NT * CB / 16) + (int)lane * NV;
  uint4 acc = uint4(0);
  int g = g_begin;
  for (; g + DEPTH <= g_end; g += DEPTH) {
    uint4 v[DEPTH][NV];
    for (int d = 0; d < DEPTH; d++)
      for (int j = 0; j < NV; j++) v[d][j] = W4[(int64_t)(g + d) * (NT * CB / 16) + j];
    for (int d = 0; d < DEPTH; d++)
      for (int j = 0; j < NV; j++) acc ^= v[d][j];
  }
  for (; g < g_end; g++)
    for (int j = 0; j < NV; j++) acc ^= W4[(int64_t)g * (NT * CB / 16) + j];
  out[tile * SK * 32 + sg * 32 + lane] = acc.x ^ acc.y ^ acc.z ^ acc.w;
"""

SHAPES = [
    # name, n, k, sk (the lane kernel's split for the shape)
    ("gate", 17408, 5120, 2),
    ("down", 5120, 17408, 8),
    ("qkv", 10240, 5120, 4),
    ("z", 6144, 5120, 8),
    ("q", 12288, 5120, 4),
]


def geometry(n, k, sk, group=64, nt=32):
    """Tiles, groups, bytes per tile group for a 4-bit tiled weight."""
    kg = k // group
    cb = group // 2
    return {
        "tiles": n // nt,
        "kg": kg,
        "cb": cb,
        "nv": max(1, cb // 16),
        "bytes": n * k // 2,
        "sk": sk,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--target-mb", type=int, default=768)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    for name, n, k, sk in SHAPES:
        g = geometry(n, k, sk)
        copies = max(2, min(64, a.target_mb * 2**20 // g["bytes"]))
        ws = [
            mx.random.randint(0, 2**31 - 1, (g["bytes"] // 4,), dtype=mx.int32).astype(
                mx.uint32
            )
            for _ in range(copies)
        ]
        mx.eval(ws)
        row = {"shape": name, "n": n, "k": k, "sk": sk, "copies": copies}
        for depth in a.depths:
            src = (
                f"constexpr int KG = {g['kg']}, SK = {sk}, NT = 32, CB = {g['cb']}, NV = {g['nv']}, DEPTH = {depth};\n"
                + SOURCE
            )
            kern = mx.fast.metal_kernel(
                name=f"lane_stream_{name}_{depth}",
                input_names=["W"],
                output_names=["out"],
                source=src,
            )

            def graph(kern=kern):
                outs, prev = [], None
                for w in ws:
                    wi = w if prev is None else mx.depends(w, prev)
                    prev = kern(
                        inputs=[wi],
                        grid=(g["tiles"] * 32 * sk, 1, 1),
                        threadgroup=(32 * sk, 1, 1),
                        output_shapes=[(g["tiles"] * sk * 32,)],
                        output_dtypes=[mx.uint32],
                    )[0]
                    outs.append(prev)
                return outs

            for _ in range(2):
                mx.eval(graph())
            ts = []
            for _ in range(7):
                gr = graph()
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(gr)
                ts.append((time.perf_counter() - t0) / copies)
            row[f"depth{depth}"] = g["bytes"] / statistics.median(ts) / 1e9
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {k_: (round(v) if isinstance(v, float) else v) for k_, v in row.items()}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
