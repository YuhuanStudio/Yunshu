"""Decode attention over a left-padded batch KV: where does the time go?

Qwen3.8 attention shape (24 query heads, 4 KV heads, head_dim 256), B rows with
different lengths inside one left-padded BatchKVCache-style buffer. Compares:

- dense: stock SDPA over the whole padded length with a padding mask
- upstream_view: mlx-vlm's ragged decode kernel on the cache's strided view
  (what the model runs today: it makes the view contiguous first)
- upstream_contig: the same kernel on already-contiguous arrays (no copy)
- upstream_fallback: mlx-vlm's per-pad-group path (taken when rows fall into
  different vector plans)
- ragged: Yunshu's per-row-length kernel (right-aligned per-row storage), the
  key-parallel variant launched over the rows' own chunks (what RaggedKVCache
  runs)
- ragged_int8: the same kernel over int8 K/V + fp16 32-dim group scales
  (YUNSHU_RAGGED_KV=int8)
- ideal_rows: per-row SDPA on exact slices (B launches; lower bound on bytes).
  At B=1 this is plain MLX fast SDPA over the row.
- ragged_per_head / ragged_int8_per_head: the per-query-head kernel (one
  simdgroup per head walks a chunk key by key), kept as the A/B baseline and as
  the fallback for head_dim != 256.

``--ab``: time the selected paths round-robin (median over ``--iters`` rounds)
so a shared GPU affects every path alike. Each sample chains ``--chain`` calls,
each call's output being the next call's queries (successive layers of one
decode step: no overlap between calls), and divides by the chain length. A
16-byte-vector read of the rows' bf16 K/V byte count, chained the same way, is
the roofline for the run (small sizes can read from the system cache, so it is
the attainable line for that working set, not DRAM alone).

GB/s is the K/V bytes each path must read (dense: the padded length; the rest:
each row's own keys; int8: codes + scales) over the measured time.

    PYTHONPATH=python python scripts/research/bench_ragged_decode_attention.py \
        --lengths 16000 500 500 500 500 500 500 500 --iters 30
    PYTHONPATH=python python scripts/research/bench_ragged_decode_attention.py \
        --lengths 131072 --iters 30 --paths ragged ragged_int8 ideal_rows
"""

import argparse
import json
import time

import mlx.core as mx

H, HKV, D = 24, 4, 256

_READ_SRC = r"""
  const uint tid = thread_position_in_grid.x;
  const uint nthreads = threads_per_grid.x;
  const device uint4* p = (const device uint4*)src;
  uint4 acc = uint4(0);
  for (uint i = tid; i < N; i += nthreads) acc ^= p[i];
  out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w ^ dep[0];
"""


def timeit(fn, iters):
    for _ in range(3):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lengths", type=int, nargs="+", default=[16000] + [500] * 7)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--paths", nargs="+", default=None, help="subset of paths")
    ap.add_argument("--ab", action="store_true", help="interleaved medians + roofline")
    ap.add_argument("--chain", type=int, default=16, help="--ab: calls per sample")
    ap.add_argument(
        "--step", type=int, default=256, help="BatchKVCache allocation step"
    )
    a = ap.parse_args()
    from mlx_vlm.models.qwen3_5 import language as lang

    from yunshu_engine.kernels import ragged_attention as ra

    lengths = a.lengths
    B = len(lengths)
    L = max(lengths)
    cap = -(-L // a.step) * a.step + a.step  # buffer larger than _idx, like the cache
    pads = [L - n for n in lengths]
    mx.random.seed(0)
    kbuf = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    vbuf = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    q = mx.random.normal((B, H, 1, D)).astype(mx.bfloat16)
    scale = D**-0.5
    kview, vview = kbuf[..., :L, :], vbuf[..., :L, :]
    kc, vc = mx.contiguous(kview), mx.contiguous(vview)
    mask = mx.arange(L)[None, :] >= mx.array(pads)[:, None]
    mask = mask[:, None, None, :]
    # right-aligned per-row storage for the ragged kernel
    rk = mx.stack(
        [
            mx.concatenate(
                [
                    kview[b, :, pads[b] :, :],
                    mx.zeros((HKV, cap - lengths[b], D), mx.bfloat16),
                ],
                axis=1,
            )
            for b in range(B)
        ]
    )
    rv = mx.stack(
        [
            mx.concatenate(
                [
                    vview[b, :, pads[b] :, :],
                    mx.zeros((HKV, cap - lengths[b], D), mx.bfloat16),
                ],
                axis=1,
            )
            for b in range(B)
        ]
    )
    lens = mx.array(lengths, dtype=mx.int32)
    rkq, rks = ra.quantize_kv(rk)
    rvq, rvs = ra.quantize_kv(rv)
    mx.eval(kbuf, vbuf, q, kc, vc, rk, rv, rkq, rks, rvq, rvs)
    row_bytes = sum(lengths) * HKV * D * 2 * 2  # bf16 K + V, rows' own keys
    kv_bytes = {
        "dense": B * L * HKV * D * 2 * 2,
        "ragged_int8": sum(lengths) * HKV * (D + D // ra.GROUP * 2) * 2,
    }

    class Stub:
        pass

    stub = Stub()
    stub._qwen3_5_decode_left_padding = pads

    def fallback():
        # force the per-pad-group path (what runs when plans differ)
        out = {}
        for pad in sorted(set(pads)):
            rows = [i for i, p in enumerate(pads) if p == pad]
            idx = mx.array(rows, dtype=mx.int32)
            o = mx.fast.scaled_dot_product_attention(
                mx.take(q, idx, axis=0),
                mx.take(kview, idx, axis=0)[:, :, pad:, :],
                mx.take(vview, idx, axis=0)[:, :, pad:, :],
                scale=scale,
            )
            for j, r in enumerate(rows):
                out[r] = o[j : j + 1]
        return mx.concatenate([out[i] for i in range(B)], axis=0)

    ref = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[b : b + 1],
                kview[b : b + 1, :, pads[b] :, :],
                vview[b : b + 1, :, pads[b] :, :],
                scale=scale,
            )
            for b in range(B)
        ],
        axis=0,
    )

    def ideal(qq):
        return mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    qq[b : b + 1],
                    kview[b : b + 1, :, pads[b] :, :],
                    vview[b : b + 1, :, pads[b] :, :],
                    scale=scale,
                )
                for b in range(B)
            ],
            axis=0,
        )

    qpaths = {
        "dense": lambda qq: mx.fast.scaled_dot_product_attention(
            qq, kview, vview, scale=scale, mask=mask
        ),
        "upstream_view": lambda qq: lang._qwen3_5_ragged_decode_attention(
            qq, kview, vview, pads, scale
        ),
        "upstream_contig": lambda qq: lang._qwen3_5_ragged_decode_attention(
            qq, kc, vc, pads, scale
        ),
        "ragged": lambda qq: ra.ragged_decode_attention(
            qq, rk, rv, lens, scale, row_lengths=lengths
        ),
        "ragged_int8": lambda qq: ra.ragged_decode_attention(
            qq, rkq, rvq, lens, scale, None, rks, rvs, row_lengths=lengths
        ),
        "ragged_per_head": lambda qq: ra.ragged_decode_attention(
            qq, rk, rv, lens, scale, impl="per_head"
        ),
        "ragged_int8_per_head": lambda qq: ra.ragged_decode_attention(
            qq, rkq, rvq, lens, scale, None, rks, rvs, impl="per_head"
        ),
        "ideal_rows": ideal,
    }
    paths = {n: (lambda f=f: f(q)) for n, f in qpaths.items()}
    paths["upstream_fallback"] = fallback
    kv_bytes["ragged_int8_per_head"] = kv_bytes["ragged_int8"]
    row = {"lengths": lengths, "cap": cap}
    if a.ab:
        names = [
            n
            for n in qpaths
            if (not a.paths or n in a.paths) and qpaths[n](q) is not None
        ]
        # roofline: stream the same number of bytes as the bf16 K/V of the rows
        nbytes = row_bytes // 16 * 16
        buf = mx.random.randint(0, 2**31, (nbytes // 4,), dtype=mx.uint32)
        rd = mx.fast.metal_kernel(
            name="yunshu_bw_probe_ab",
            input_names=["src", "dep"],
            output_names=["out"],
            source=_READ_SRC,
        )

        def read(dep):
            return rd(
                inputs=[buf, dep],
                template=[("N", nbytes // 16)],
                grid=(65536, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(65536,)],
                output_dtypes=[mx.uint32],
            )[0]

        fns = {n: qpaths[n] for n in names}
        fns["read_roofline"] = read
        seed = {n: q for n in names}
        seed["read_roofline"] = mx.zeros((1,), mx.uint32)

        def sample(n):
            o = seed[n]
            mx.eval(o)
            t0 = time.perf_counter()
            for _ in range(a.chain):
                o = fns[n](o)
            mx.eval(o)
            return (time.perf_counter() - t0) / a.chain * 1e6

        for n in fns:
            sample(n)
        samples = {n: [] for n in fns}
        for _ in range(a.iters):
            for n in fns:
                samples[n].append(sample(n))
        roof = nbytes / sorted(samples["read_roofline"])[a.iters // 2] / 1e3
        row["read_roofline_GB/s"] = round(roof, 1)
        for n in names:
            us = sorted(samples[n])[a.iters // 2]
            gbs = kv_bytes.get(n, row_bytes) / us / 1e3
            err = float(
                mx.max(
                    mx.abs(fns[n](q).astype(mx.float32) - ref.astype(mx.float32))
                ).item()
            )
            row[n] = {
                "us": round(us, 1),
                "GB/s": round(gbs, 1),
                "pct_roof": round(100 * gbs / roof, 1),
                "max_abs_err": round(err, 5),
            }
        print(json.dumps(row))
        return
    for name, fn in paths.items():
        if a.paths and name not in a.paths:
            continue
        out = fn()
        if out is None:
            row[name] = "not launched (plans differ)"
            continue
        err = float(
            mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
        )
        us = timeit(fn, a.iters)
        gbs = kv_bytes.get(name, row_bytes) / us / 1e3
        row[name] = {
            "us": round(us, 1),
            "GB/s": round(gbs, 1),
            "max_abs_err": round(err, 5),
        }
    print(json.dumps(row))


if __name__ == "__main__":
    main()
