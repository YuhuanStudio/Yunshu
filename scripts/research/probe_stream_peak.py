"""Achievable DRAM read bandwidth of this GPU (the ceiling for the projection kernels; gpuq only).

A plain streaming read of a buffer larger than the system cache, 16-byte loads, several
independent loads per thread, over a few launch shapes. The best value is the practical
peak a weight-streaming kernel can approach; the datasheet figure is 614 GB/s for the
40-core M5 Max.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SOURCE = r"""
  const uint tid = thread_position_in_grid.x;
  const uint nth = threads_per_grid.x;
  const device uint4* p = (const device uint4*)src;
  uint4 a0 = uint4(0), a1 = uint4(0), a2 = uint4(0), a3 = uint4(0);
  uint i = tid;
  for (; i + 3 * nth < N; i += 4 * nth) {
    a0 ^= p[i]; a1 ^= p[i + nth]; a2 ^= p[i + 2 * nth]; a3 ^= p[i + 3 * nth];
  }
  for (; i < N; i += nth) a0 ^= p[i];
  a0 ^= a1 ^ a2 ^ a3;
  out[tid] = a0.x ^ a0.y ^ a0.z ^ a0.w;
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=int, default=1024)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    n16 = a.mb * 2**20 // 16
    src = mx.random.randint(0, 2**31 - 1, (n16 * 4,), dtype=mx.int32).astype(mx.uint32)
    mx.eval(src)
    rows = []
    for threads_total in (1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19):
        for tg in (256, 1024):
            kern = mx.fast.metal_kernel(
                name=f"stream_{threads_total}_{tg}",
                input_names=["src"],
                output_names=["out"],
                source=f"constexpr uint N = {n16};\n" + SOURCE,
            )

            def run(kern=kern, threads_total=threads_total, tg=tg):
                return kern(
                    inputs=[src],
                    grid=(threads_total, 1, 1),
                    threadgroup=(tg, 1, 1),
                    output_shapes=[(threads_total,)],
                    output_dtypes=[mx.uint32],
                )[0]

            for _ in range(2):
                mx.eval(run())
            ts = []
            for _ in range(7):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(run())
                ts.append(time.perf_counter() - t0)
            rows.append(
                {
                    "threads": threads_total,
                    "tg": tg,
                    "GBps": a.mb * 2**20 / statistics.median(ts) / 1e9,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    with a.out.open("a") as f:
        f.write(
            json.dumps({"best_GBps": max(r["GBps"] for r in rows), "rows": rows}) + "\n"
        )


if __name__ == "__main__":
    main()
