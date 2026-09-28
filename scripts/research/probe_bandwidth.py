"""Attainable GPU read bandwidth on this machine (the decode-attention roofline).

A Metal kernel streams a large buffer with 16-byte vector loads (uint4) and
reduces it, for several threadgroup sizes and loads-per-thread; reports GB/s.
Also times MLX's fast SDPA for one decode query over B=1 Qwen3.8-shaped K/V
(24 q heads, 4 KV heads, head_dim 256) at a few lengths — what the stock
kernel reaches against the same line.

    PYTHONPATH=python python scripts/research/probe_bandwidth.py --gib 1 --iters 20
"""

import argparse
import json
import time

import mlx.core as mx

_SRC = r"""
  const uint tid = thread_position_in_grid.x;
  const uint nthreads = threads_per_grid.x;
  const device uint4* p = (const device uint4*)src;
  const uint n = N;
  uint4 acc = uint4(0);
  for (uint i = tid; i < n; i += nthreads) {
    acc ^= p[i];
  }
  out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
"""


def timeit(fn, iters):
    for _ in range(3):
        mx.eval(fn())
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    nbytes = int(a.gib * 2**30) // 16 * 16
    buf = mx.random.randint(0, 2**31, (nbytes // 4,), dtype=mx.uint32)
    mx.eval(buf)
    k = mx.fast.metal_kernel(
        name="yunshu_bw_probe", input_names=["src"], output_names=["out"], source=_SRC
    )
    best = 0.0
    for tg in (256, 512, 1024):
        for total in (2**16, 2**18, 2**20):

            def run(tg=tg, total=total):
                return k(
                    inputs=[buf],
                    template=[("N", nbytes // 16)],
                    grid=(total, 1, 1),
                    threadgroup=(tg, 1, 1),
                    output_shapes=[(total,)],
                    output_dtypes=[mx.uint32],
                )[0]

            s = timeit(run, a.iters)
            gbs = nbytes / s / 1e9
            best = max(best, gbs)
            print(
                json.dumps(
                    {
                        "probe": "read",
                        "threadgroup": tg,
                        "threads": total,
                        "GB/s": round(gbs, 1),
                    }
                ),
                flush=True,
            )
    print(json.dumps({"probe": "read_best", "GB/s": round(best, 1)}), flush=True)

    H, HKV, D = 24, 4, 256
    for L in (8192, 32768, 131072):
        kk = mx.random.normal((1, HKV, L, D)).astype(mx.bfloat16)
        vv = mx.random.normal((1, HKV, L, D)).astype(mx.bfloat16)
        q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
        mx.eval(kk, vv, q)
        s = timeit(
            lambda q=q, kk=kk, vv=vv: mx.fast.scaled_dot_product_attention(
                q, kk, vv, scale=D**-0.5
            ),
            a.iters,
        )
        kv = 2 * L * HKV * D * 2
        print(
            json.dumps(
                {
                    "probe": "mlx_sdpa_b1",
                    "L": L,
                    "us": round(s * 1e6, 1),
                    "GB/s": round(kv / s / 1e9, 1),
                    "pct_of_best_read": round(100 * kv / s / 1e9 / best, 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
