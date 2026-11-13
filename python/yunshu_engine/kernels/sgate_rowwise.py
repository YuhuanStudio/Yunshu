"""Row-invariant ``shared_expert_gate`` for Flash-Next prefill (N=1 quantized linear).

Stock quantized matmul gives a row different bits for M=3..16 than for M=512, so a prefill
segment's gate depends on its length. A float32 multiply-and-sum over the bf16 dequantized
weight (the decode router's rows) is the same for any M (probe 11-12, M=1..16/512/4096)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class RowwiseGate:
    """Callable replacing a stock 1-output QuantizedLinear; per-row f32 dot product."""

    def __init__(self, linear: Any) -> None:
        self.weight = (
            mx.dequantize(
                linear.weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
            )
            .astype(mx.bfloat16)
            .astype(mx.float32)
        )  # [1, K]

    def __call__(self, x: mx.array) -> mx.array:
        k = int(self.weight.shape[-1])
        flat = x.reshape(-1, k).astype(mx.float32)
        y = (flat * self.weight).sum(-1, keepdims=True)  # [M, 1]
        return y.reshape(*x.shape[:-1], 1).astype(x.dtype)


def wrap(linear: Any) -> Any:
    """``RowwiseGate`` for a quantized 1-output gate without bias; any other module unchanged."""
    if isinstance(linear, RowwiseGate) or not hasattr(linear, "scales"):
        return linear
    if getattr(linear, "bias", None) is not None or int(linear.weight.shape[0]) != 1:
        return linear
    return RowwiseGate(linear)


def install(language_model: Any) -> int:
    """Wrap every MoE block's ``shared_expert_gate`` in ``language_model``; returns the count."""
    count = 0
    for layer in getattr(language_model.model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "shared_expert_gate", None)
        if gate is None:
            continue
        wrapped = wrap(gate)
        if wrapped is not gate:
            mlp.shared_expert_gate = wrapped
            count += 1
    return count
