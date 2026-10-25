"""Kernel-boundary loss of a chain of dependent weight-streaming kernels, and cross-kernel prefetch (gpuq only).

A decode round is ~1800 dependent kernels; each boundary drains the GPU's memory pipes and ramps them up again.
This streams SLICES distinct buffers of ``--mb`` MB (like consecutive projections' weights):

  big      one kernel over all slices (no boundary)
  chain    one kernel per slice, each depending on the previous (what a round does)
  prefetch chain, plus extra threadgroups at the end of kernel k that read the first ``pf`` MB of slice k + 1
           (they fill cores idle in k's last wave; if the system cache keeps those lines, k + 1's first tiles hit)
  early    like prefetch, but the extra threadgroups come first (control: contention instead of filling idle cores)

GB/s counts the main slices only.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SOURCE = r"""
  const uint tid = thread_position_in_grid.x;
  const uint main_threads = P[4];
  const uint pf_threads = P[5];
  const uint early = P[6];
  uint4 acc = uint4(0);
  const device uint4* s = (const device uint4*)src;
  uint pf_tid = tid;
  uint main_tid = tid;
  bool is_pf = false;
  if (pf_threads) {
    if (early) { is_pf = tid < pf_threads; pf_tid = tid; main_tid = tid - pf_threads; }
    else { is_pf = tid >= main_threads; pf_tid = tid - main_threads; main_tid = tid; }
  }
  if (is_pf) {
    for (uint i = pf_tid; i < P[3]; i += pf_threads) acc ^= s[P[2] + i];
  } else {
    uint i = main_tid;
    const uint start = P[0], count = P[1];
    for (; i + 3 * main_threads < count; i += 4 * main_threads) {
      acc ^= s[start + i] ^ s[start + i + main_threads] ^ s[start + i + 2 * main_threads] ^ s[start + i + 3 * main_threads];
    }
    for (; i < count; i += main_threads) acc ^= s[start + i];
  }
  out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
"""


def params_list(k, slices, n16, pf, early, threads, pf_threads, big=False):
    """The kernel's P array: [start, count, next_start, pf_count, main_threads, pf_threads, early, 0] in 16-byte units."""
    if big:
        return [0, n16 * slices, 0, 0, threads, 0, 0, 0]
    nxt = min(k + 1, slices - 1)
    pfn = (pf * 2**20 // 16) if (pf and k + 1 < slices) else 0
    return [k * n16, n16, nxt * n16, pfn, threads, pf_threads if pfn else 0, early, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mbs", type=int, nargs="+", default=[12, 25, 50, 100])
    p.add_argument("--slices", type=int, default=48)
    p.add_argument("--pfs", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--threads", type=int, default=1 << 18)
    p.add_argument("--pf-threads", type=int, default=1 << 14)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    kern = mx.fast.metal_kernel(
        name="boundary_stream",
        input_names=["src", "P"],
        output_names=["out"],
        source=SOURCE,
    )

    def run(src, params, threads):
        return kern(
            inputs=[src, params],
            grid=(threads, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(threads,)],
            output_dtypes=[mx.uint32],
        )[0]

    for mb in a.mbs:
        n16 = mb * 2**20 // 16
        total16 = n16 * a.slices
        src = mx.random.randint(0, 2**31 - 1, (total16 * 4,), dtype=mx.int32).astype(
            mx.uint32
        )
        mx.eval(src)

        def params(k, pf, early=0, big=False):
            return mx.array(
                params_list(k, a.slices, n16, pf, early, a.threads, a.pf_threads, big),
                dtype=mx.uint32,
            )

        def chain(pf, early=0):
            outs, prev = [], None
            for k in range(a.slices):
                pr = params(k, pf, early)
                if prev is not None:
                    pr = mx.depends(pr, prev)
                total_threads = a.threads + (
                    a.pf_threads if (pf and k + 1 < a.slices) else 0
                )
                prev = run(src, pr, total_threads)
                outs.append(prev)
            return outs

        arms = {
            "big": lambda: [run(src, params(0, 0, big=True), a.threads)],
            "chain": lambda: chain(0),
        }
        for pf in a.pfs:
            arms[f"prefetch{pf}"] = lambda pf=pf: chain(pf)
            arms[f"early{pf}"] = lambda pf=pf: chain(pf, 1)
        times = {k: [] for k in arms}
        for fn in arms.values():
            mx.eval(fn())
        for rep in range(7):
            order = list(arms) if rep % 2 == 0 else list(arms)[::-1]
            for k in order:
                g = arms[k]()
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(g)
                times[k].append(time.perf_counter() - t0)
        row = {"mb": mb, "slices": a.slices}
        for k, ts in times.items():
            row[k] = a.slices * mb * 2**20 / statistics.median(ts) / 1e9
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {k: (round(v) if isinstance(v, float) else v) for k, v in row.items()}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
