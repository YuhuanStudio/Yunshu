"""Decode attention over a left-padded batch KV: where does the time go?

Qwen3.8 attention shape (24 query heads, 4 KV heads, head_dim 256), B rows with
different lengths inside one left-padded BatchKVCache-style buffer. Compares:

- dense: stock SDPA over the whole padded length with a padding mask
- upstream_view: mlx-vlm's ragged decode kernel on the cache's strided view
  (what the model runs today: it makes the view contiguous first)
- upstream_contig: the same kernel on already-contiguous arrays (no copy)
- upstream_fallback: mlx-vlm's per-pad-group path (taken when rows fall into
  different vector plans)
- ragged: Yunshu's per-row-length kernel (right-aligned per-row storage)
- ragged_int8: the same kernel over int8 K/V + fp16 32-dim group scales
  (YUNSHU_RAGGED_KV=int8)
- ideal_rows: per-row SDPA on exact slices (B launches; lower bound on bytes).
  At B=1 this is plain MLX fast SDPA over the row.

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
    paths = {
        "dense": lambda: mx.fast.scaled_dot_product_attention(
            q, kview, vview, scale=scale, mask=mask
        ),
        "upstream_view": lambda: lang._qwen3_5_ragged_decode_attention(
            q, kview, vview, pads, scale
        ),
        "upstream_contig": lambda: lang._qwen3_5_ragged_decode_attention(
            q, kc, vc, pads, scale
        ),
        "upstream_fallback": fallback,
        "ragged": lambda: ra.ragged_decode_attention(q, rk, rv, lens, scale),
        "ragged_int8": lambda: ra.ragged_decode_attention(
            q, rkq, rvq, lens, scale, None, rks, rvs
        ),
        "ideal_rows": lambda: mx.concatenate(
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
        ),
    }
    row = {"lengths": lengths, "cap": cap}
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
