"""Unified-memory read / write / copy bandwidth vs footprint (1 MB .. 8 GB).

Shows the SLC / cache regime and the DRAM plateau. Read uses a Metal kernel with
16-byte vector loads and many threadgroups; write and copy use custom kernels.
Theoretical peak: M5 Max 40-core GPU = 614 GB/s (LPDDR5X-9600, 512-bit).

    PYTHONPATH=python:scripts/research/hw python scripts/research/hw/bw_unified_memory.py
"""

import mlx.core as mx
from _common import Out, timeit

READ = r"""
  const uint tid = thread_position_in_grid.x;
  const uint nth = threads_per_grid.x;
  const device uint4* p = (const device uint4*)src;
  uint4 acc = uint4(0);
  for (uint i = tid; i < N; i += nth) acc ^= p[i];
  out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
"""
# four independent loads in flight per thread
READ4 = r"""
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
WRITE = r"""
  const uint tid = thread_position_in_grid.x;
  const uint nth = threads_per_grid.x;
  device uint4* p = (device uint4*)dst;
  for (uint i = tid; i < N; i += nth) p[i] = uint4(i, i, i, i);
  out[tid] = 0;
"""
COPY = r"""
  const uint tid = thread_position_in_grid.x;
  const uint nth = threads_per_grid.x;
  const device uint4* s = (const device uint4*)src;
  device uint4* d = (device uint4*)dst;
  for (uint i = tid; i < N; i += nth) d[i] = s[i];
  out[tid] = 0;
"""


def kern(name, src, ins):
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def main():
    out = Out("bw_unified_memory")
    kr, kr4 = kern("bw_r", READ, ["src"]), kern("bw_r4", READ4, ["src"])
    kw, kc = kern("bw_w", WRITE, ["dst"]), kern("bw_c", COPY, ["src", "dst"])
    for mb in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        nb = mb * 2**20
        n16 = nb // 16
        src = mx.random.randint(0, 2**31 - 1, (nb // 4 // 1024, 1024)).astype(mx.uint32)
        dst = mx.zeros((nb // 4 // 1024, 1024), dtype=mx.uint32)
        mx.eval(src, dst)
        iters = 30 if mb <= 512 else 8
        best = {}
        for tg in (256, 1024):
            for total in (2**16, 2**18, 2**20):
                total = min(total, max(tg, n16 // tg * tg))

                def call(k, ins):
                    return lambda: mx.eval(
                        k(inputs=ins, template=[("N", n16)], grid=(total, 1, 1),
                          threadgroup=(tg, 1, 1), output_shapes=[(total,)],
                          output_dtypes=[mx.uint32])[0]
                    )

                for nm, fn, mult in (
                    ("read", call(kr, [src]), 1),
                    ("read4", call(kr4, [src]), 1),
                    ("write", call(kw, [dst]), 1),
                    ("copy", call(kc, [src, dst]), 2),
                ):
                    t, tmin = timeit(fn, iters)
                    gbs = mult * nb / t / 1e9
                    if gbs > best.get(nm, (0,))[0]:
                        best[nm] = (gbs, tg, total, tmin)
        # Batched: 32 back-to-back read dispatches per eval, so launch latency amortizes and a
        # buffer that fits the SLC is re-read from cache.
        tg, total = 1024, min(2**18, max(1024, n16 // 1024 * 1024))
        outs = lambda: [  # noqa: E731
            kr(inputs=[src], template=[("N", n16)], grid=(total, 1, 1),
               threadgroup=(tg, 1, 1), output_shapes=[(total,)],
               output_dtypes=[mx.uint32])[0]
            for _ in range(32)
        ]
        t, _ = timeit(lambda: mx.eval(outs()), max(4, iters // 3))
        best["read_batched32"] = (32 * nb / t / 1e9, tg, total, t / 32)
        t, _ = timeit(lambda: mx.eval(src + 1), iters)
        best["mlx_add_rw"] = (2 * nb / t / 1e9, 0, 0, 0)
        out(mb=mb, **{k: {"GBps": round(v[0], 1), "tg": v[1], "threads": v[2],
                          "min_ms": round(v[3] * 1e3, 4)} for k, v in best.items()})
        src = dst = None


if __name__ == "__main__":
    main()
