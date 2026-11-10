"""Can stock MLX run R rows as R independent M=1 dispatches in ONE call (batch axis, weight broadcast)?

For each real E4B / E2B shape: bitwise equality of row r against its own stock M=1 call, and time per
projection of (a) per-row loop, (b) stock M=R (the non-invariant batched call), (c) batched call with the
weight broadcast along a new batch axis (stride 0), which dispatches the M=1 kernel once per row.
Usage: rows_stock_probe.py dense|q4g64|q8g64|q4g32
"""

import sys
import time

import mlx.core as mx

mode = sys.argv[1]
L = 12
mx.random.seed(0)
if mode == "dense":
    shapes = [(2560, 2048), (2560, 512), (2048, 2560), (2560, 10240), (10240, 2560), (2560, 262144)]
    bits = group = 0
else:
    shapes = [(1536, 2048), (1536, 256), (2048, 1536), (1536, 6144), (6144, 1536), (1536, 262144)]
    bits, group = int(mode[1]), int(mode.split("g")[1])


def bench(fn, count, n=15):
    for _ in range(3):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n / count * 1e6


for k, n in shapes:
    count = 2 if n > 100000 else L
    x8 = mx.random.normal((8, k)).astype(mx.bfloat16)
    if mode == "dense":
        ws = [(mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16) for _ in range(count)]
        mx.eval(ws, x8)

        def one(x, w):
            return x @ w.T

        def stock_rows(x, w):
            return x @ w.T

        def bcast(x, w):
            r = x.shape[0]
            return mx.matmul(x[:, None, :], mx.broadcast_to(w.T, (r, k, n)))[:, 0, :]

        args = ws
    else:
        qs = [mx.quantize((mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=group, bits=bits) for _ in range(count)]
        mx.eval([list(q) for q in qs], x8)

        def one(x, q):
            return mx.quantized_matmul(x, q[0], q[1], q[2], transpose=True, group_size=group, bits=bits)

        def stock_rows(x, q):
            return one(x, q)

        def bcast(x, q):
            r = x.shape[0]
            bw, bs, bb = (mx.broadcast_to(a, (r, *a.shape)) for a in q)
            return mx.quantized_matmul(
                x[:, None, :], bw, bs, bb, transpose=True, group_size=group, bits=bits
            )[:, 0, :]

        args = qs
    a0 = args[0]
    ref = [one(x8[r : r + 1], a0) for r in range(8)]
    mx.eval(ref)
    res = {}
    for r in (2, 4, 8):
        try:
            got = bcast(x8[:r], a0)
            mx.eval(got)
            res[r] = all(bool(mx.array_equal(got[i], ref[i][0]).item()) for i in range(r))
        except Exception as e:  # noqa: BLE001
            res[r] = f"ERR {type(e).__name__}: {str(e)[:60]}"
    stock_eq = {
        r: all(bool(mx.array_equal(stock_rows(x8[:r], a0)[i], ref[i][0]).item()) for i in range(r))
        for r in (2, 4, 8)
    }
    print(f"{mode} K={k} N={n} bcast_equal_M1={res} stockMR_equal_M1={stock_eq}", flush=True)
    for r in (1, 2, 4, 8):
        x = x8[:r]
        try:
            t_loop = bench(lambda: [mx.concatenate([one(x[i : i + 1], a) for i in range(r)]) for a in args], count)
            t_stock = bench(lambda: [stock_rows(x, a) for a in args], count)
            t_b = bench(lambda: [bcast(x, a) for a in args], count)
            print(f"  R={r}: per-row loop {t_loop:.0f}  stock M=R {t_stock:.0f}  bcast {t_b:.0f} us/proj", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  R={r}: ERR {e}", flush=True)
