from __future__ import annotations
"""Yunshu KV Compression — Three-tier KV cache hierarchy.

Hot (UMA FP16) → Warm (4-bit quantized) → Cold (SSD)

This module provides a simple symmetric per-channel 4-bit quantization
for the warm tier. Metal kernel implementation will replace the numpy
path in Phase 2.
"""


from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class KVTier(Enum):
    HOT = auto()   # FP16 in UMA
    WARM = auto()  # INT4 quantized
    COLD = auto()  # SSD


@dataclass
class TierConfig:
    """Configuration for KV tier sizing."""

    hot_ratio: float = 0.60
    warm_ratio: float = 0.30
    cold_ratio: float = 0.10
    warm_bits: int = 4
    warm_group_size: int = 64
    cold_ssd_path: str = ""


def quantize_kv_4bit(
    kv_array,
    group_size: int = 64,
):
    """Quantize FP16 KV array to 4-bit symmetric per-group.

    Args:
        kv_array: MLX array, shape [..., num_tokens, head_dim] in FP16.
        group_size: Number of elements per quantization group.

    Returns:
        (quantized_packed, scales) where:
        - quantized_packed: UINT8, 2 values packed per byte. Shape [..., num_tokens, head_dim // 2]
        - scales: FP16 per-group absolute max. Shape [..., num_tokens, head_dim // group_size]
    """
    if HAS_MLX:
        arr = np.array(kv_array, dtype=np.float32)
    else:
        arr = np.array(kv_array, dtype=np.float32)

    *batch_dims, num_tokens, head_dim = arr.shape

    if arr.ndim < 2 or head_dim == 0:
        raise ValueError(f"kv_array must be at least 2D with head_dim > 0, got shape {arr.shape}")

    # Pad head_dim to be divisible by group_size * 2 (for packing)
    effective_dim = head_dim
    pad_needed = (group_size - head_dim % group_size) % group_size
    if pad_needed:
        arr = np.pad(arr, ((0,0),) * len(batch_dims) + ((0,0), (0, pad_needed)))
        effective_dim = head_dim + pad_needed

    # Reshape into groups: [..., tokens, num_groups, group_size]
    num_groups = effective_dim // group_size
    grouped = arr.reshape(batch_dims + [num_tokens, num_groups, group_size])

    # Per-group absolute max
    group_max = np.max(np.abs(grouped), axis=-1, keepdims=True)  # [..., tokens, groups, 1]
    group_max = np.where(np.isfinite(group_max), group_max, 0.0)
    scales = (group_max / 7.0).squeeze(-1)  # [..., tokens, groups]
    scales = np.where(scales == 0, 1.0, scales)

    # Quantize: [..., tokens, groups, group_size]
    scales_expanded = scales[..., None]
    quantized = np.clip(np.round(arr.reshape(batch_dims + [num_tokens, num_groups, group_size]) / scales_expanded), -7, 7).astype(np.int8)
    quantized_uint = (quantized + 8).astype(np.uint8)

    # Reshape back: [..., tokens, effective_dim]
    flat = quantized_uint.reshape(batch_dims + [num_tokens, effective_dim])

    # Pack: 2 values per byte
    even_vals = flat[..., 0::2]
    odd_vals = flat[..., 1::2]
    packed = ((even_vals << 4) | odd_vals).astype(np.uint8)

    # Trim back to original head_dim in packed space
    packed_dim = head_dim // 2 + (head_dim % 2)
    packed = packed[..., :packed_dim]

    scales_out = np.array(scales, dtype=np.float16)

    if HAS_MLX:
        return mx.array(packed), mx.array(scales_out)
    return packed, scales_out


def dequantize_kv_4bit(
    packed,
    scales,
    head_dim: int = 0,
    group_size: int = 64,
):
    """Dequantize 4-bit packed KV back to FP16.

    Args:
        packed: UINT8 packed array, shape [..., tokens, packed_dim].
        scales: Per-group scales, shape [..., tokens, num_groups].
        head_dim: Original head dimension.
        group_size: Number of elements per quantization group.

    Returns:
        FP16 MLX array.
    """
    if HAS_MLX:
        packed_np = np.array(packed, dtype=np.uint8)
        scales_np = np.array(scales, dtype=np.float32)
    else:
        packed_np = np.array(packed, dtype=np.uint8)
        scales_np = np.array(scales, dtype=np.float32)

    # Unpack nibbles
    high = (packed_np >> 4) & 0xF
    low = packed_np & 0xF

    # Interleave back to full dimension
    *batch_dims, num_tokens, packed_dim = high.shape
    total_dim = packed_dim * 2
    unpacked = np.empty(batch_dims + [num_tokens, total_dim], dtype=np.uint8)
    unpacked[..., 0::2] = high
    unpacked[..., 1::2] = low

    # Convert back to signed: subtract 8
    signed = unpacked.astype(np.float32) - 8.0

    # Trim to head_dim if specified
    if head_dim > 0 and head_dim < total_dim:
        signed = signed[..., :head_dim]
        total_dim = head_dim

    # Expand per-group scales: [..., tokens, groups] → [..., tokens, groups, 1] → [..., tokens, dim]
    num_groups = scales_np.shape[-1]
    scales_expanded = scales_np.reshape(batch_dims + [num_tokens, num_groups, 1])
    scales_expanded = np.repeat(scales_expanded, group_size, axis=-1)
    scales_expanded = scales_expanded.reshape(batch_dims + [num_tokens, -1])

    # Match the trimmed dimension
    if scales_expanded.shape[-1] > total_dim:
        scales_expanded = scales_expanded[..., :total_dim]

    result = signed * scales_expanded

    if HAS_MLX:
        return mx.array(result.astype(np.float16))
    return result.astype(np.float16)


def compute_compression_ratio(bits: int = 4) -> float:
    return 16.0 / bits
