"""Tests for DFlash Block Diffusion Engine."""

from yunshu_engine.dflash import (
    BlockPlan,
    DFlashConfig,
    DFlashEngine,
    L1BlockCache,
)


class TestDFlashConfig:
    def test_defaults(self):
        cfg = DFlashConfig()
        assert not cfg.enabled
        assert cfg.block_size == 256
        assert cfg.coarse_steps == 2
        assert cfg.refine_steps == 4

    def test_total_steps(self):
        cfg = DFlashConfig(coarse_steps=3, refine_steps=5)
        assert cfg.total_steps == 8

    def test_speedup_estimate(self):
        cfg = DFlashConfig(coarse_steps=2, refine_steps=4)
        assert cfg.speedup_estimate > 1.0

    def test_to_dict(self):
        cfg = DFlashConfig(enabled=True)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert "speedup_estimate" in d

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_DFLASH", "1")
        monkeypatch.setenv("YUNSHU_DFLASH_BLOCK_SIZE", "512")
        cfg = DFlashConfig.from_env()
        assert cfg.enabled
        assert cfg.block_size == 512


class TestBlockPlan:
    def test_create_single_block(self):
        plan = BlockPlan.create(256, 256, 256, overlap=0)
        assert plan.num_blocks == 1

    def test_create_multiple_blocks(self):
        plan = BlockPlan.create(512, 512, 256, overlap=0)
        assert plan.num_blocks >= 4  # At least 2x2

    def test_create_with_overlap(self):
        plan = BlockPlan.create(512, 512, 256, overlap=16)
        assert plan.num_blocks >= 4
        # With overlap, might have more blocks to cover edges

    def test_block_coordinates(self):
        plan = BlockPlan.create(256, 256, 256, overlap=0)
        x, y, x_end, y_end = plan.blocks[0]
        assert x == 0
        assert y == 0
        assert x_end == 256
        assert y_end == 256


class TestL1BlockCache:
    def test_put_and_get(self):
        cache = L1BlockCache(max_entries=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self):
        cache = L1BlockCache()
        assert cache.get("missing") is None

    def test_eviction(self):
        cache = L1BlockCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None  # evicted

    def test_stats(self):
        cache = L1BlockCache()
        cache.put("k", "v")
        cache.get("k")  # hit
        cache.get("miss")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        cache = L1BlockCache()
        cache.put("k", "v")
        cache.clear()
        assert cache.get("k") is None


class TestDFlashEngine:
    def test_not_enabled_by_default(self):
        engine = DFlashEngine(DFlashConfig(enabled=False))
        assert not engine.is_enabled

    def test_enabled(self):
        engine = DFlashEngine(DFlashConfig(enabled=True))
        assert engine.is_enabled

    def test_generate_disabled(self):
        engine = DFlashEngine(DFlashConfig(enabled=False))
        result = engine.generate(None, None, "test")
        assert "error" in result

    def test_stats(self):
        engine = DFlashEngine(DFlashConfig(enabled=True))
        stats = engine.get_stats()
        assert "total_generations" in stats
        assert "l1_cache" in stats
        assert "config" in stats

    def test_clear_cache(self):
        engine = DFlashEngine(DFlashConfig(enabled=True))
        engine._l1_cache.put("k", "v")
        engine.clear_cache()
        assert engine._l1_cache.get("k") is None

    def test_is_compatible_none(self):
        assert not DFlashEngine.is_compatible(None)

    def test_is_compatible_flux(self):
        class FakeConfig:
            model_type = "flux"

        class FakeModel:
            config = FakeConfig()

        assert DFlashEngine.is_compatible(FakeModel())

    def test_is_compatible_incompatible(self):
        class FakeConfig:
            model_type = "llama"

        class FakeModel:
            config = FakeConfig()

        assert not DFlashEngine.is_compatible(FakeModel())
