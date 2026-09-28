"""Faster exact kernel choice for 5-bit layers in mlx-vlm's multi-row verify.

mlx-vlm's exact verify matmul picks its kernel by row count: rows <= 5 use the
base kernel (per-row register accumulators), and only 4-bit layers get the
``streamed`` / ``token_tiled`` kernels for >= 6 rows. Mixed 4/5-bit checkpoints
(the Qwen3.8-27B oQ4e build has 166 5-bit projections, incl. every GDN
``out_proj`` and 27 ``down_proj``) therefore run the base kernel at 6-8 rows,
where it collapses. Measured on M5 Max for a 17408->5120 5-bit projection
(docs/research/runs/2026-09-28-matrix/verify-qmv-variants.jsonl):

    rows      6        7        8
    base   1172 us  2204 us  2829 us
    tiled   746 us   823 us  1008 us   (bit-identical to single-row decode)

This routes 5-bit layers with >= 6 verify rows to ``token_tiled`` and keeps
5-bit grouped projections off the fused base kernel at those widths. Output is
unchanged (every row still matches the decode kernel bit for bit); only the
kernel choice differs. ``streamed`` is not used for 5-bit: it is not exact there.
"""

from __future__ import annotations

import mlx.core as mx

MIN_TILED_ROWS = 6


def _tiled_5bit(qv, linear, x: mx.array) -> mx.array | None:
    B, T, K = x.shape
    N = linear.weight.shape[0]
    x = mx.contiguous(x)
    kernel = qv._target_verify_qmv_token_tiled_kernel(
        linear.bits, linear.group_size, x.dtype, T, K, N
    )
    out = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // 8), B * ((T + 1) // 2)),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        out = out + linear["bias"]
    return out


def install() -> bool:
    from mlx_vlm.models import quantized_verifier as qv
    from mlx_vlm.speculative.ops import linear as ops

    single = qv.optimized_affine_linear
    grouped = qv.optimized_affine_linears
    if getattr(single, "_yunshu_tiled5", False):
        return True

    def optimized_affine_linear(linear, x):
        if (
            getattr(linear, "bits", None) == 5
            and x.ndim == 3
            and x.shape[1] >= MIN_TILED_ROWS
            and qv._can_optimized_affine_linear(linear, x)
        ):
            return _tiled_5bit(qv, linear, x)
        return single(linear, x)

    def optimized_affine_linears(linears, x):
        if (
            linears
            and getattr(linears[0], "bits", None) == 5
            and x.ndim == 3
            and x.shape[1] >= MIN_TILED_ROWS
        ):
            return None  # per-projection tiled path beats the fused base kernel
        return grouped(linears, x)

    optimized_affine_linear._yunshu_tiled5 = True
    qv.optimized_affine_linear = optimized_affine_linear
    qv.optimized_affine_linears = optimized_affine_linears
    ops._target_verify_optimized_affine_linear = optimized_affine_linear
    ops._target_verify_quantized_linears = optimized_affine_linears
    return True
