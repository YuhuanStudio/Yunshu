# Upstream (derived): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py, src/tensorfold/kernels/qwen/dense/v1/lane_widen.py @ 34bae79a
"""5/6/8-bit projections on the M5 tensor units via integer codes (TensorFold's method).

The tensor unit (MetalPerformancePrimitives ``matmul2d``) has no 5-bit format,
and dequantizing to bf16 in threadgroup memory (our first 5-bit attempt) is
slower than MLX for 1..8 rows. TensorFold's lane matmul instead widens each
64-value group's codes to uint8 in threadgroup memory and multiplies bf16
activations by the raw codes, then applies the affine terms per group:

    y = sum_g  s_g * (x_g . q_g)  +  b_g * sum(x_g)

(``kernels.tensorfold.lane_qmm.lane_matmul`` with its ``bytes`` kernel). Rows
1..16 share one 16-row op, so a row's result does not depend on how many rows
share the call — the property ``batch_invariant`` needs.

``IntCodeLinear`` keeps MLX's weight layout (so prompt chunks above
``lane_qmm.MAX_ROWS`` rows use stock ``quantized_matmul`` without re-laying out
weights) and adds the group-major (scale, bias) pairs the kernel reads.
Used for every 5/6/8-bit projection when projections are packed (see
``qwen35_packed_linear``); one row outside the speculative lane stays on MLX qmv.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .tensorfold import lane_qmm

INT_BITS = (5, 6, 8)


def eligible(linear: Any) -> bool:
    """A bias-free affine 5/6/8-bit group-64 QuantizedLinear the kernel reads."""
    if type(linear) is not nn.QuantizedLinear:
        return False
    if getattr(linear, "mode", "affine") != "affine" or "bias" in linear:
        return False
    if int(linear.bits) not in INT_BITS or int(linear.group_size) != 64:
        return False
    if linear.get("biases") is None or linear.scales.dtype != mx.bfloat16:
        return False
    n = int(linear.weight.shape[0])
    k = int(linear.weight.shape[1]) * 32 // int(linear.bits)
    return n % 4 == 0 and k % 64 == 0


class IntCodeLinear(nn.Module):
    """A 5/6/8-bit linear whose small-row calls run on the tensor units."""

    def __init__(self, linear: nn.QuantizedLinear):
        super().__init__()
        bits = int(linear.bits)
        self.bits = bits
        self.group_size = 64
        self.output_dims = int(linear.weight.shape[0])
        self.input_dims = int(linear.weight.shape[1]) * 32 // bits
        self.weight = linear.weight
        self.scales = linear.scales
        self.biases = linear.biases
        # Group-major (scale, bias) bf16 pairs: (K/64, N, 2).
        self.lane_sbt = lane_qmm.pack_scales(linear.scales, linear.biases)
        self.freeze()

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bits={self.bits}, int-code lane matmul"
        )

    def __call__(self, x: mx.array) -> mx.array:
        rows = 1
        for d in x.shape[:-1]:
            rows *= int(d)
        # One row outside the speculative lane: MLX's qmv is faster than a
        # 16-row tensor-unit tile (256 vs 331 us on 17408x5120). Inside the
        # lane (batch-invariant active) one row must match a verify row, so it
        # stays on the lane kernel.
        from .batch_invariant import _STATE as INVARIANT_STATE

        single = rows == 1 and not INVARIANT_STATE.get("active", False)
        if rows > lane_qmm.MAX_ROWS or single:
            return mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=64,
                bits=self.bits,
            )
        dtype = x.dtype
        x2 = x.reshape(-1, self.input_dims).astype(mx.bfloat16)
        y = lane_qmm.lane_matmul(x2, self.weight, self.lane_sbt, tiled=False)
        return y.reshape(*x.shape[:-1], self.output_dims).astype(dtype)

    def quantized_rows(self, ids: mx.array):
        """``QuantizedLinear`` weight, scales and biases of output rows ``ids``."""
        return (
            mx.take(self.weight, ids, axis=0),
            mx.take(self.scales, ids, axis=0),
            mx.take(self.biases, ids, axis=0),
        )
