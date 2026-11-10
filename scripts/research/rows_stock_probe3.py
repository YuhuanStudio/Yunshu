"""Equality matrix between stock M=1 call forms and batch-axis forms, dense bf16 (the shapes that failed)."""

import mlx.core as mx

mx.random.seed(0)
for k, n in [(2560, 2048), (2560, 10240), (2560, 4096), (2048, 2560)]:
    x = mx.random.normal((8, 1, k)).astype(mx.bfloat16)
    w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    wt = w.T
    forms = {
        "3D [1,1,K]@wT": lambda r: mx.matmul(x[r : r + 1], wt)[0],
        "2D [1,K]@wT": lambda r: mx.matmul(x[r, :, :], wt),
        "w@x (gemv non-T)": lambda r: mx.matmul(w, x[r, 0, :][:, None]).T,
    }
    for rows in (2, 4, 8):
        forms[f"bcast R={rows}"] = (lambda rows: (lambda r: mx.matmul(x[:rows], mx.broadcast_to(wt, (rows, k, n)))[r]))(rows)
        forms[f"bcast-contig R={rows}"] = (lambda rows: (lambda r: mx.matmul(x[:rows], mx.contiguous(mx.broadcast_to(wt, (rows, k, n))))[r]))(rows)
        forms[f"einsum-like R={rows}"] = (lambda rows: (lambda r: mx.matmul(x[:rows, :, :], wt[None])[r]))(rows)
    out = {name: [f(r) for r in range(2)] for name, f in forms.items()}
    mx.eval(list(out.values()))
    ref = out["3D [1,1,K]@wT"]
    print(f"K={k} N={n}", flush=True)
    for name, vals in out.items():
        eq = [bool(mx.array_equal(vals[r].reshape(-1), ref[r].reshape(-1)).item()) for r in range(2)]
        print(f"   {name:24s} equals 3D stock: {eq}", flush=True)
