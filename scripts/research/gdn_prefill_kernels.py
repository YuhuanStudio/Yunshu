"""GDN prefill core: mlx-vlm's step kernel vs mx.fast.gated_delta_update (equivalence + time) at Qwen3.8-27B shapes."""

import json
import time

import mlx.core as mx
from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_kernel

Hk, Hv, Dk, Dv = 16, 48, 128, 128
mx.random.seed(3)
out = {}
for T in (256, 2048, 4096):
    q = (mx.random.normal((1, T, Hk, Dk)) * 0.05).astype(mx.bfloat16)
    k = (mx.random.normal((1, T, Hk, Dk)) * 0.09).astype(mx.bfloat16)
    v = mx.random.normal((1, T, Hv, Dv)).astype(mx.bfloat16)
    a = mx.random.normal((1, T, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((1, T, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.float32) * 0.5
    dt_bias = mx.random.normal((Hv,)).astype(mx.bfloat16)
    st = mx.random.normal((1, Hv, Dv, Dk)).astype(mx.float32) * 0.1
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    mx.eval(q, k, v, g, beta, st)

    def old():
        return gated_delta_kernel(q, k, v, g, beta, st)

    def new():
        return mx.fast.gated_delta_update(q, k, v, g, beta, st)

    def bench(f):
        for _ in range(2):
            mx.eval(f())
        t0 = time.perf_counter()
        for _ in range(8):
            mx.eval(f())
        return (time.perf_counter() - t0) / 8 * 1e3

    yo, so = old()
    yn, sn = new()
    mx.eval(yo, so, yn, sn)
    f32 = mx.float32
    out[T] = dict(
        old_ms=round(bench(old), 2),
        new_ms=round(bench(new), 2),
        max_abs_y=float(mx.max(mx.abs(yo.astype(f32) - yn.astype(f32)))),
        ref_abs_y=float(mx.max(mx.abs(yo.astype(f32)))),
        max_abs_state=float(mx.max(mx.abs(so - sn))),
        ref_abs_state=float(mx.max(mx.abs(so))),
        shapes=[list(yn.shape), list(sn.shape), str(yn.dtype), str(sn.dtype)],
    )
print(json.dumps(out, indent=1))
