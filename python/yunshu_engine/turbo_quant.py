from __future__ import annotations

"""TurboQuant KV Cache — per-layer mixed-precision KV quantization.

Instead of uniform quantization across all layers,
TurboQuant applies different quantization levels to different attention
layers based on their sensitivity:

- Early layers (0..start): Full precision (FP16) — high sensitivity
- Middle layers: 8-bit quantization — moderate sensitivity
- Later layers: 4-bit quantization — attention patterns stabilize

This achieves 3-4x KV memory reduction with minimal quality loss,
compared to 8x from uniform 4-bit quantization.

The quantization is applied at the block boundary (every block_size tokens)
during the paged scheduler's cache_completed_blocks() path.
"""

import logging
from dataclasses import dataclass
from typing import Any

from .kv_quantization import KVQuantConfig, KVQuantizer

logger = logging.getLogger(__name__)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant mixed-precision KV cache.

    Attributes:
        enabled: Whether TurboQuant is active.
        total_layers: Total number of attention layers in the model.
        fp16_end_layer: Layers 0..fp16_end_layer use FP16 (no quantization).
        int8_end_layer: Layers fp16_end_layer+1..int8_end_layer use 8-bit.
        int4_group_size: Group size for 4-bit quantization of remaining layers.
    """

    enabled: bool = False
    total_layers: int = 0
    fp16_end_layer: int = 4  # First N layers stay FP16
    int8_end_layer: int = 16  # Next M layers use 8-bit
    int4_group_size: int = 64

    def get_layer_config(self, layer_idx: int) -> KVQuantConfig | None:
        """Get the quantization config for a specific layer.

        Returns None for FP16 layers (no quantization).
        """
        if not self.enabled:
            return None
        if layer_idx <= self.fp16_end_layer:
            return None  # FP16
        if layer_idx <= self.int8_end_layer:
            return KVQuantConfig(bits=8, group_size=32)
        return KVQuantConfig(bits=4, group_size=self.int4_group_size)

    @property
    def expected_compression_ratio(self) -> float:
        """Estimate overall compression ratio."""
        if not self.enabled or self.total_layers == 0:
            return 1.0
        fp16_count = min(self.fp16_end_layer + 1, self.total_layers)
        int8_count = min(
            max(self.int8_end_layer - self.fp16_end_layer, 0),
            self.total_layers - fp16_count,
        )
        int4_count = self.total_layers - fp16_count - int8_count

        # Compression: FP16=1x, INT8=2x, INT4=4x
        total = self.total_layers
        ratio = (fp16_count * 1 + int8_count * 2 + int4_count * 4) / total
        return ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_layers": self.total_layers,
            "fp16_end_layer": self.fp16_end_layer,
            "int8_end_layer": self.int8_end_layer,
            "int4_group_size": self.int4_group_size,
            "expected_compression_ratio": round(self.expected_compression_ratio, 2),
        }


class TurboQuantManager:
    """Manages per-layer mixed-precision KV quantization.

    Creates and caches KVQuantizer instances for each quantization tier,
    then applies the correct quantizer based on layer index.
    """

    def __init__(self, config: TurboQuantConfig) -> None:
        self._config = config
        self._quantizers: dict[int, KVQuantizer] = {}
        self._stats = {
            "layers_fp16": 0,
            "layers_int8": 0,
            "layers_int4": 0,
            "total_quantized_bytes": 0,
            "total_fp16_bytes": 0,
        }

        if config.enabled:
            self._build_quantizers()

    @property
    def config(self) -> TurboQuantConfig:
        return self._config

    def _build_quantizers(self) -> None:
        for layer_idx in range(self._config.total_layers):
            cfg = self._config.get_layer_config(layer_idx)
            if cfg is not None:
                self._quantizers[layer_idx] = KVQuantizer(cfg)
                if cfg.bits == 8:
                    self._stats["layers_int8"] += 1
                elif cfg.bits == 4:
                    self._stats["layers_int4"] += 1
            else:
                self._stats["layers_fp16"] += 1

        logger.info(
            f"TurboQuant initialized: {self._stats['layers_fp16']} FP16, "
            f"{self._stats['layers_int8']} INT8, {self._stats['layers_int4']} INT4 "
            f"(compression: {self._config.expected_compression_ratio:.1f}x)"
        )

    def quantize_layer(
        self, layer_idx: int, kv_tensor: list
    ) -> tuple[bytes | list, dict | None]:
        """Quantize a single layer's KV tensor.

        Returns:
            (quantized_data, metadata) — for FP16 layers, returns (original, None).
        """
        if not self._config.enabled:
            return kv_tensor, None

        quantizer = self._quantizers.get(layer_idx)
        if quantizer is None:
            self._stats["total_fp16_bytes"] += _estimate_size(kv_tensor)
            return kv_tensor, None

        packed, meta = quantizer.quantize(kv_tensor)
        self._stats["total_quantized_bytes"] += len(packed)
        return packed, meta

    def dequantize_layer(
        self, layer_idx: int, data: bytes | list, meta: dict | None
    ) -> list:
        """Dequantize a single layer's KV tensor."""
        if meta is None:
            return data if isinstance(data, list) else []

        quantizer = self._quantizers.get(layer_idx)
        if quantizer is None:
            return data if isinstance(data, list) else []

        return quantizer.dequantize(data, meta)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "config": self._config.to_dict(),
        }

    @staticmethod
    def from_model_settings(settings: Any) -> TurboQuantConfig:
        """Create TurboQuantConfig from ModelSettings.

        Uses kv_cache_quant_bits and kv_cache_quant_start_layer to
        determine the tier boundaries.
        """
        quant_bits = getattr(settings, "kv_cache_quant_bits", None)
        if quant_bits is None:
            return TurboQuantConfig(enabled=False)

        total_layers = getattr(settings, "total_layers", 0)
        start_layer = getattr(settings, "kv_cache_quant_start_layer", 0)
        group_size = getattr(settings, "kv_cache_quant_group_size", 64)

        if total_layers == 0:
            return TurboQuantConfig(enabled=False)

        return TurboQuantConfig(
            enabled=True,
            total_layers=total_layers,
            fp16_end_layer=start_layer - 1 if start_layer > 0 else 0,
            int8_end_layer=min(start_layer + total_layers // 3, total_layers - 1),
            int4_group_size=group_size,
        )


def _estimate_size(tensor: list) -> int:
    """Rough byte size estimate for a nested list of floats."""
    if isinstance(tensor, (bytes, bytearray)):
        return len(tensor)
    count = 0
    stack = [tensor]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            count += 1
    return count * 4  # float32
