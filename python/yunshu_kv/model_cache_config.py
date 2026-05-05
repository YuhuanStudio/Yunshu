"""Yunshu Cache Type Registry — per-layer type-aware KV cache handling.

Based on oMLX's type_handlers.py and type_registry.py pattern.
Maps each model layer to its cache type and provides type-aware operations:
- Block slicing (KVCache: yes, ArraysCache: no, RotatingKVCache: circular)
- State extraction/reconstruction per type
- Boundary snapshot routing (non-sliceable layers go to SSD immediately)

Architecture:
  ModelCacheConfig.build_from_model(model) → per-layer CacheLayerConfig list
  CacheLayerConfig determines: sliceable? boundary-eligible? handler?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .mlx_cache import CacheType, detect_cache_type, is_sliceable

logger = logging.getLogger(__name__)


@dataclass
class CacheLayerConfig:
    """Per-layer cache configuration (oMLX ModelCacheConfig pattern)."""
    layer_index: int
    cache_type: CacheType
    sliceable: bool
    boundary_eligible: bool  # Needs boundary snapshot (non-sliceable layers)

    @property
    def type_name(self) -> str:
        return self.cache_type.name


class ModelCacheConfig:
    """Per-model cache type registry (oMLX ModelCacheConfig pattern).

    Inspects the model's prompt cache list to determine each layer's cache
    type and capabilities. This enables type-aware routing:
    - Sliceable layers → normal paged KV cache
    - Non-sliceable layers → boundary snapshot to SSD

    Usage:
        config = ModelCacheConfig.build_from_model(model)
        for layer_cfg in config.layers:
            if layer_cfg.sliceable:
                # Normal paged cache path
            else:
                # Boundary snapshot path
    """

    def __init__(self, layers: list[CacheLayerConfig]) -> None:
        self.layers = layers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def sliceable_layers(self) -> list[CacheLayerConfig]:
        return [l for l in self.layers if l.sliceable]

    @property
    def boundary_layers(self) -> list[CacheLayerConfig]:
        return [l for l in self.layers if l.boundary_eligible]

    def get_layer(self, index: int) -> CacheLayerConfig:
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return CacheLayerConfig(
            layer_index=index,
            cache_type=CacheType.UNKNOWN,
            sliceable=False,
            boundary_eligible=True,
        )

    @staticmethod
    def build_from_cache_list(cache_list: list) -> ModelCacheConfig:
        """Build config from an existing prompt cache list."""
        layers = []
        for i, cache_obj in enumerate(cache_list):
            ct = detect_cache_type(cache_obj)
            slice = is_sliceable(cache_obj)
            layers.append(CacheLayerConfig(
                layer_index=i,
                cache_type=ct,
                sliceable=slice,
                boundary_eligible=not slice,
            ))
        return ModelCacheConfig(layers)

    @staticmethod
    def build_from_model(model: Any) -> ModelCacheConfig:
        """Build config by creating a temporary cache and inspecting types."""
        try:
            from mlx_lm.utils import make_prompt_cache
            cache_list = make_prompt_cache(model)
            return ModelCacheConfig.build_from_cache_list(cache_list)
        except Exception as e:
            logger.warning(f"Could not inspect model cache types: {e}")
            # Fallback: assume all layers are standard KVCache
            num_layers = 0
            config = getattr(model, "config", None) or getattr(model, "args", None)
            if config is not None:
                num_layers = getattr(config, "num_hidden_layers", 0)
            return ModelCacheConfig([
                CacheLayerConfig(
                    layer_index=i,
                    cache_type=CacheType.KVCACHE,
                    sliceable=True,
                    boundary_eligible=False,
                )
                for i in range(max(num_layers, 1))
            ])

    def summary(self) -> dict:
        type_counts = {}
        for l in self.layers:
            type_counts[l.type_name] = type_counts.get(l.type_name, 0) + 1
        return {
            "num_layers": self.num_layers,
            "num_sliceable": len(self.sliceable_layers),
            "num_boundary": len(self.boundary_layers),
            "type_distribution": type_counts,
        }
