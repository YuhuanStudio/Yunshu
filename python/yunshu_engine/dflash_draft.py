# Upstream (extended): Blaizzy/mlx-vlm (MIT) DFlashDraftModel._project_context_kv @ v0.7.4
"""Fuse private quantized DFlash context K/V projections.

Upstream fuses float Linear weights, but its isinstance guard skips the
QuantizedLinear projections installed for serving. Pack their affine rows
once and project context for every draft layer in one call. This affects
proposals only: the target weights, verifier and shared draft readout are
untouched. Install only after the private drafter has been quantized.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def install_quantized_context(drafter: Any) -> bool:
    if getattr(drafter, "_yunshu_quantized_context", False):
        return True
    projections = [
        projection
        for layer in drafter.layers
        for projection in (layer.self_attn.k_proj, layer.self_attn.v_proj)
    ]
    if not projections or not all(
        isinstance(p, nn.QuantizedLinear)
        and getattr(p, "mode", "affine") == "affine"
        and p.biases is not None
        and "bias" not in p
        for p in projections
    ):
        return False
    first = projections[0]
    signature = (
        first.bits,
        first.group_size,
        first.weight.shape[1],
        first.scales.dtype,
    )
    if any(
        (p.bits, p.group_size, p.weight.shape[1], p.scales.dtype) != signature
        for p in projections
    ):
        return False
    widths = [int(p.weight.shape[0]) for p in projections]
    boundaries, total = [], 0
    for width in widths[:-1]:
        total += width
        boundaries.append(total)
    weight, scales, biases = (
        mx.concatenate([getattr(p, name) for p in projections], axis=0)
        for name in ("weight", "scales", "biases")
    )
    mx.eval(weight, scales, biases)

    def project_context(hidden: mx.array) -> list[tuple[mx.array, mx.array]]:
        projected = mx.quantized_matmul(
            hidden,
            weight,
            scales,
            biases,
            transpose=True,
            group_size=first.group_size,
            bits=first.bits,
        )
        chunks = mx.split(projected, boundaries, axis=-1)
        return list(zip(chunks[::2], chunks[1::2], strict=True))

    object.__setattr__(drafter, "_project_context_kv", project_context)
    object.__setattr__(drafter, "_yunshu_quantized_context", True)
    return True


def install_selector_readout(drafter: Any) -> bool:
    """Keep DFlash2's trained candidate selector on the greedy proposal path.

    Its inherited DFlashDraftModel.draft_block_greedy reads unary argmax and
    bypasses the subclass's pairwise selector. Target verification is unchanged.
    """
    if getattr(drafter, "_yunshu_selector_readout", False):
        return True
    selector = getattr(drafter, "candidate_selector", None)
    if not callable(getattr(selector, "select", None)):
        return False
    object.__setattr__(drafter, "draft_block_greedy", drafter.draft_block)
    object.__setattr__(drafter, "_yunshu_selector_readout", True)
    return True


__all__ = ["install_quantized_context", "install_selector_readout"]
