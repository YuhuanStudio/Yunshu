"""Yunshu KV Quantization — group-wise 4-bit quantization for warm-tier KV cache.

Implements group-wise uniform quantization for KV cache tensors destined for
the warm storage tier. Packing 32-bit floats into 4-bit integers reduces KV
memory by 8x, trading accuracy for compression.

Algorithm:
  1. Divide the last dimension of the KV tensor into groups of group_size.
  2. For each group, compute a scale factor (and optional zero-point).
  3. Quantize each element: q = round(x / scale) clipped to [0, 2^bits - 1].
  4. Pack two 4-bit values into each byte (nibble packing).

Dequantization reverses the process: x_hat = q * scale.

Tensor layout assumed: [layers, heads, seq_len, dim].
Quantization is applied along the last dimension in groups of group_size.

References:
  - KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Wang et al.)
  - GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

Integration:
  - AdaptiveKVQuantizer (kv_optimizations.py) uses KVQuantizer internally for
    per-layer INT4/INT8 quantization. Import via:
      from .kv_optimizations import AdaptiveKVQuantizer
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class KVQuantConfig:
    """Configuration for KV cache quantization.

    Attributes:
        bits: Number of bits per element (default 4).
        group_size: Number of elements per quantization group (default 64).
        symmetric: If True, use symmetric quantization (zero-point = 0).
                   If False, use asymmetric with an explicit zero-point.
    """

    bits: int = 4
    group_size: int = 64
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.bits < 1 or self.bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def max_q(self) -> int:
        """Maximum quantized value (2^bits - 1)."""
        return (1 << self.bits) - 1

    @property
    def compression_ratio(self) -> float:
        """Expected compression ratio vs 32-bit float."""
        return 32.0 / self.bits


# ── KV Quantizer ─────────────────────────────────────────────────────────────


class KVQuantizer:
    """Group-wise KV cache quantizer.

    Quantizes float KV tensors to packed 4-bit (or configurable-bit) bytes
    with per-group scale factors.

    Usage:
        config = KVQuantConfig(bits=4, group_size=64)
        quantizer = KVQuantizer(config)

        # kv_tensor: list of floats or nested lists
        packed, meta = quantizer.quantize(kv_tensor)
        reconstructed = quantizer.dequantize(packed, meta)

        ratio = quantizer.estimate_compression(len(original_bytes))
        metrics = quantizer.validate_accuracy(original, reconstructed)
    """

    def __init__(self, config: KVQuantConfig | None = None) -> None:
        self._config = config or KVQuantConfig()

    @property
    def config(self) -> KVQuantConfig:
        return self._config

    def quantize(
        self, kv_tensor: list,
    ) -> tuple[bytes, dict[str, Any]]:
        """Quantize a float KV tensor to packed bytes.

        Flattens the tensor, quantizes in groups, and packs into bytes.

        Args:
            kv_tensor: Nested list of floats representing the KV tensor.
                       Expected shape: [layers, heads, seq_len, dim].

        Returns:
            Tuple of (packed_data, metadata).
            metadata contains:
              - shape: original tensor shape
              - config: KVQuantConfig fields
              - scales: list of scale factors per group
              - zero_points: list of zero-points per group (if asymmetric)
              - num_elements: total number of elements
        """
        # Flatten and record shape
        flat, shape = _flatten_with_shape(kv_tensor)
        num_elements = len(flat)

        if num_elements == 0:
            return b"", {
                "shape": shape,
                "bits": self._config.bits,
                "group_size": self._config.group_size,
                "symmetric": self._config.symmetric,
                "scales": [],
                "zero_points": [],
                "num_elements": 0,
            }

        group_size = self._config.group_size
        max_q = self._config.max_q
        num_groups = math.ceil(num_elements / group_size)

        scales: list[float] = []
        zero_points: list[float] = []
        quantized_values: list[int] = []

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, num_elements)
            group = flat[start:end]

            g_min = min(group)
            g_max = max(group)

            if self._config.symmetric:
                # Symmetric: map [-abs_max, +abs_max] to [0, max_q]
                # Scale so that abs_max maps to max_q//2 (half the range)
                abs_max = max(abs(g_min), abs(g_max))
                half_range = max_q // 2
                if abs_max == 0:
                    scale = 1.0
                else:
                    scale = abs_max / half_range if half_range > 0 else 1.0
                zp = 0.0
            else:
                # Asymmetric: map [g_min, g_max] to [0, max_q]
                range_val = g_max - g_min
                if range_val == 0:
                    scale = 1.0
                else:
                    scale = range_val / max_q
                zp = g_min

            scales.append(scale)
            zero_points.append(zp)

            for val in group:
                if self._config.symmetric:
                    q = round(val / scale) if scale != 0 else 0
                    # Clamp to [0, max_q] for symmetric unsigned storage
                    q = max(0, min(max_q, q + max_q // 2))
                else:
                    q = round((val - zp) / scale) if scale != 0 else 0
                    q = max(0, min(max_q, q))
                quantized_values.append(q)

        # Pack into bytes
        packed = _pack_nibbles(quantized_values, self._config.bits)

        metadata = {
            "shape": shape,
            "bits": self._config.bits,
            "group_size": self._config.group_size,
            "symmetric": self._config.symmetric,
            "scales": scales,
            "zero_points": zero_points if not self._config.symmetric else [],
            "num_elements": num_elements,
        }

        return packed, metadata

    def dequantize(self, packed_data: bytes, metadata: dict) -> list:
        """Dequantize packed bytes back to approximate float values.

        Reconstructs the original nested list shape from the metadata.

        Args:
            packed_data: Packed quantized bytes from quantize().
            metadata: Metadata dict from quantize().

        Returns:
            Nested list of floats matching the original shape.
        """
        num_elements = metadata.get("num_elements", 0)
        if num_elements == 0:
            return _unflatten_to_shape([], metadata.get("shape", []))

        bits = metadata.get("bits", 4)
        group_size = metadata.get("group_size", 64)
        symmetric = metadata.get("symmetric", True)
        scales = metadata.get("scales", [])
        zero_points = metadata.get("zero_points", [])
        shape = metadata.get("shape", [])
        max_q = (1 << bits) - 1

        # Unpack
        quantized = _unpack_nibbles(packed_data, bits, num_elements)

        # Dequantize per group
        num_groups = len(scales)
        flat: list[float] = [0.0] * num_elements

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, num_elements)
            scale = scales[g]

            if symmetric:
                half = max_q // 2
                for i in range(start, end):
                    flat[i] = (quantized[i] - half) * scale
            else:
                zp = zero_points[g] if g < len(zero_points) else 0.0
                for i in range(start, end):
                    flat[i] = quantized[i] * scale + zp

        return _unflatten_to_shape(flat, shape)

    def estimate_compression(self, original_bytes: int) -> float:
        """Estimate compression ratio for a given original byte size.

        Args:
            original_bytes: Size of the original float data in bytes.

        Returns:
            Compression ratio (e.g., 8.0 for 4-bit vs 32-bit float).
        """
        if original_bytes == 0:
            return 0.0
        return self._config.compression_ratio

    def validate_accuracy(
        self, original: list, reconstructed: list,
    ) -> dict[str, float]:
        """Compute accuracy metrics between original and reconstructed tensors.

        Args:
            original: Original float tensor (nested list).
            reconstructed: Reconstructed float tensor (nested list).

        Returns:
            Dict with:
              - mse: Mean squared error
              - max_error: Maximum absolute error
              - cosine_similarity: Cosine similarity between flattened vectors
        """
        flat_orig, _ = _flatten_with_shape(original)
        flat_recon, _ = _flatten_with_shape(reconstructed)

        n = len(flat_orig)
        if n == 0:
            return {"mse": 0.0, "max_error": 0.0, "cosine_similarity": 1.0}

        # Pad reconstructed to match length if needed
        while len(flat_recon) < n:
            flat_recon.append(0.0)

        # MSE
        sq_errors = [(flat_orig[i] - flat_recon[i]) ** 2 for i in range(n)]
        mse = sum(sq_errors) / n

        # Max error
        max_error = max(abs(flat_orig[i] - flat_recon[i]) for i in range(n))

        # Cosine similarity
        dot = sum(flat_orig[i] * flat_recon[i] for i in range(n))
        norm_o = math.sqrt(sum(v * v for v in flat_orig))
        norm_r = math.sqrt(sum(v * v for v in flat_recon[:n]))

        if norm_o == 0 or norm_r == 0:
            cos_sim = 1.0 if norm_o == 0 and norm_r == 0 else 0.0
        else:
            cos_sim = dot / (norm_o * norm_r)
            # Clamp to [-1, 1]
            cos_sim = max(-1.0, min(1.0, cos_sim))

        return {
            "mse": mse,
            "max_error": max_error,
            "cosine_similarity": cos_sim,
        }


# ── Packing / Unpacking ──────────────────────────────────────────────────────


def _pack_nibbles(values: list[int], bits: int) -> bytes:
    """Pack quantized integer values into bytes.

    For 4-bit: two values per byte (low nibble first, then high).
    For 8-bit: one value per byte.
    For other bit widths: pack sequentially into bytes.

    Args:
        values: List of integer values in [0, 2^bits - 1].
        bits: Bits per value.

    Returns:
        Packed bytes.
    """
    if bits == 8:
        return bytes(values)

    if bits == 4:
        # Pack two 4-bit values per byte
        packed = bytearray()
        for i in range(0, len(values), 2):
            low = values[i] & 0x0F
            high = values[i + 1] & 0x0F if i + 1 < len(values) else 0
            packed.append(low | (high << 4))
        return bytes(packed)

    # General case: pack values bitwise into bytes
    packed = bytearray()
    bit_buffer = 0
    bits_in_buffer = 0

    for val in values:
        bit_buffer |= (val & ((1 << bits) - 1)) << bits_in_buffer
        bits_in_buffer += bits
        while bits_in_buffer >= 8:
            packed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8

    if bits_in_buffer > 0:
        packed.append(bit_buffer & 0xFF)

    return bytes(packed)


def _unpack_nibbles(packed: bytes, bits: int, num_values: int) -> list[int]:
    """Unpack bytes back to integer values.

    Args:
        packed: Packed bytes from _pack_nibbles.
        bits: Bits per value.
        num_values: Number of values to unpack.

    Returns:
        List of integer values.
    """
    if num_values == 0:
        return []

    if bits == 8:
        return list(packed[:num_values])

    if bits == 4:
        values = []
        for byte in packed:
            values.append(byte & 0x0F)
            values.append((byte >> 4) & 0x0F)
            if len(values) >= num_values:
                break
        return values[:num_values]

    # General case: unpack values bitwise
    mask = (1 << bits) - 1
    values = []
    bit_buffer = 0
    bits_in_buffer = 0
    byte_idx = 0

    for _ in range(num_values):
        while bits_in_buffer < bits and byte_idx < len(packed):
            bit_buffer |= packed[byte_idx] << bits_in_buffer
            bits_in_buffer += 8
            byte_idx += 1
        values.append(bit_buffer & mask)
        bit_buffer >>= bits
        bits_in_buffer -= bits

    return values


# ── Shape Utilities ──────────────────────────────────────────────────────────


def _flatten_with_shape(nested: list) -> tuple[list[float], list[int]]:
    """Flatten a nested list and return (flat_list, shape).

    Args:
        nested: Arbitrarily nested list of numbers.

    Returns:
        Tuple of (flat values, shape as list of ints).
    """
    shape: list[int] = []

    def _get_shape(obj: Any) -> list[int]:
        if not isinstance(obj, list) or len(obj) == 0:
            return []
        s = [len(obj)]
        inner = _get_shape(obj[0])
        return s + inner

    def _flatten(obj: Any) -> list[float]:
        if isinstance(obj, (int, float)):
            return [float(obj)]
        result: list[float] = []
        for item in obj:
            result.extend(_flatten(item))
        return result

    shape = _get_shape(nested)
    flat = _flatten(nested)
    return flat, shape


def _unflatten_to_shape(flat: list[float], shape: list[int]) -> list:
    """Reshape a flat list into a nested list matching the given shape.

    Args:
        flat: Flat list of float values.
        shape: Target shape as list of dimension sizes.

    Returns:
        Nested list matching shape.
    """
    if not shape:
        return flat

    if len(shape) == 1:
        return list(flat[:shape[0]])

    chunk_size = 1
    for d in shape[1:]:
        chunk_size *= d

    result = []
    for i in range(shape[0]):
        start = i * chunk_size
        end = start + chunk_size
        result.append(_unflatten_to_shape(flat[start:end], shape[1:]))

    return result
