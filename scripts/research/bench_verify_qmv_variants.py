"""Micro-benchmark mlx-vlm's exact verify matmul kernels per row count and bit width.

mlx-vlm picks the kernel for a T-row verify forward: T<=5 -> base kernel
(per-row register arrays), 4-bit 6<=T<=8 -> streamed, 4-bit T>8 -> token_tiled;
5-bit always base. This forces each variant for Qwen3.8-27B projection shapes and
reports microseconds per call plus whether every row is bit-identical to the
single-row (T=1, i.e. decode) result — the property MTP needs to stay lossless.

    .venv/bin/python scripts/research/bench_verify_qmv_variants.py
"""

import json
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import quantized_verifier as qv

SHAPES = {"mlp_down": (17408, 5120), "mlp_up": (5120, 17408), "attn_q": (5120, 12288)}
VARIANTS = {
    "base": (qv._target_verify_qmv_kernel, 4, False),
    "streamed": (qv._target_verify_qmv_streamed_kernel, 1, False),
    "token_tiled": (qv._target_verify_qmv_token_tiled_kernel, 4, True),
}


def run(linear, x, variant):
    factory, rps, tiled = VARIANTS[variant]
    B, T, K = x.shape
    N = linear.weight.shape[0]
    kernel = factory(linear.bits, linear.group_size, x.dtype, T, K, N)
    return kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // (2 * rps)), B * ((T + 1) // 2) if tiled else B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]


def timeit(fn, iters=50):
    for _ in range(5):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters * 1e6


mx.random.seed(0)
for bits in (4, 5):
    for name, (k_in, n_out) in SHAPES.items():
        lin = nn.QuantizedLinear(k_in, n_out, bias=False, group_size=64, bits=bits)
        lin.scales = lin.scales.astype(mx.bfloat16)
        lin.biases = lin.biases.astype(mx.bfloat16)
        x_all = (mx.random.normal((1, 8, k_in)) * 0.5).astype(mx.bfloat16)
        ref_rows = [run(lin, x_all[:, t : t + 1], "base") for t in range(8)]
        stock_rows = [lin(x_all[:, t : t + 1]) for t in range(8)]
        mx.eval(ref_rows, stock_rows)
        base_eq_stock = all(
            bool(mx.array_equal(a, b).item())
            for a, b in zip(ref_rows, stock_rows, strict=True)
        )
        print(
            json.dumps(
                {
                    "bits": bits,
                    "shape": name,
                    "base_T1_equals_stock_decode": base_eq_stock,
                }
            ),
            flush=True,
        )
        for T in range(1, 9):
            x = x_all[:, :T]
            for variant in VARIANTS:
                try:
                    out = run(lin, x, variant)
                    mx.eval(out)
                    exact = all(
                        bool(mx.array_equal(out[:, t], ref_rows[t][:, 0]).item())
                        for t in range(T)
                    )
                    us = timeit(lambda x=x, v=variant: run(lin, x, v))
                except Exception as e:  # noqa: BLE001
                    exact, us = None, repr(e)[:80]
                print(
                    json.dumps(
                        {
                            "bits": bits,
                            "shape": name,
                            "T": T,
                            "variant": variant,
                            "us": round(us, 1) if isinstance(us, float) else us,
                            "exact_vs_decode": exact,
                        }
                    ),
                    flush=True,
                )
