"""Tests for ModelCacheConfig and CacheLayerConfig."""

from unittest.mock import MagicMock

from yunshu_kv.mlx_cache import CacheType
from yunshu_kv.model_cache_config import CacheLayerConfig, ModelCacheConfig


class TestCacheLayerConfig:
    def test_type_name(self):
        cfg = CacheLayerConfig(
            layer_index=0,
            cache_type=CacheType.KVCACHE,
            sliceable=True,
            boundary_eligible=False,
        )
        assert cfg.type_name == "KVCACHE"

    def test_rotating_not_sliceable(self):
        cfg = CacheLayerConfig(
            layer_index=2,
            cache_type=CacheType.ROTATING_KVCACHE,
            sliceable=False,
            boundary_eligible=True,
        )
        assert cfg.sliceable is False
        assert cfg.boundary_eligible is True


class TestModelCacheConfig:
    def test_num_layers(self):
        layers = [
            CacheLayerConfig(0, CacheType.KVCACHE, True, False),
            CacheLayerConfig(1, CacheType.KVCACHE, True, False),
            CacheLayerConfig(2, CacheType.ARRAYS_CACHE, False, True),
        ]
        cfg = ModelCacheConfig(layers)
        assert cfg.num_layers == 3

    def test_sliceable_layers(self):
        layers = [
            CacheLayerConfig(0, CacheType.KVCACHE, True, False),
            CacheLayerConfig(1, CacheType.ARRAYS_CACHE, False, True),
            CacheLayerConfig(2, CacheType.KVCACHE, True, False),
        ]
        cfg = ModelCacheConfig(layers)
        assert len(cfg.sliceable_layers) == 2
        assert cfg.sliceable_layers[0].layer_index == 0
        assert cfg.sliceable_layers[1].layer_index == 2

    def test_boundary_layers(self):
        layers = [
            CacheLayerConfig(0, CacheType.KVCACHE, True, False),
            CacheLayerConfig(1, CacheType.ARRAYS_CACHE, False, True),
        ]
        cfg = ModelCacheConfig(layers)
        assert len(cfg.boundary_layers) == 1
        assert cfg.boundary_layers[0].layer_index == 1

    def test_get_layer_valid(self):
        layers = [CacheLayerConfig(0, CacheType.KVCACHE, True, False)]
        cfg = ModelCacheConfig(layers)
        assert cfg.get_layer(0).cache_type == CacheType.KVCACHE

    def test_get_layer_invalid_returns_unknown(self):
        cfg = ModelCacheConfig([])
        result = cfg.get_layer(99)
        assert result.cache_type == CacheType.UNKNOWN
        assert result.boundary_eligible is True

    def test_summary(self):
        layers = [
            CacheLayerConfig(0, CacheType.KVCACHE, True, False),
            CacheLayerConfig(1, CacheType.KVCACHE, True, False),
            CacheLayerConfig(2, CacheType.ARRAYS_CACHE, False, True),
        ]
        cfg = ModelCacheConfig(layers)
        s = cfg.summary()
        assert s["num_layers"] == 3
        assert s["num_sliceable"] == 2
        assert s["num_boundary"] == 1
        assert s["type_distribution"]["KVCACHE"] == 2
        assert s["type_distribution"]["ARRAYS_CACHE"] == 1

    def test_build_from_cache_list(self):
        kv_mock = MagicMock()
        kv_mock.__class__.__name__ = "KVCache"

        arrays_mock = MagicMock()
        arrays_mock.__class__.__name__ = "ArraysCache"

        cfg = ModelCacheConfig.build_from_cache_list([kv_mock, arrays_mock])
        assert cfg.num_layers == 2

    def test_empty_config(self):
        cfg = ModelCacheConfig([])
        assert cfg.num_layers == 0
        assert cfg.sliceable_layers == []
        assert cfg.boundary_layers == []
