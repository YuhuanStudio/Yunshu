"""Tests for kv manager — block pool and KV cache management."""

import pytest

from yunshu_kv.manager import KVCacheManager, KVCacheConfig, compute_num_blocks


class TestKVCacheConfig:
    def test_create_config(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        assert config.block_size == 64
        assert config.num_layers == 32


class TestComputeNumBlocks:
    def test_compute(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=2,
            num_kv_heads=4,
            head_dim=128,
        )
        total_memory = 1024 * 1024 * 1024
        num_blocks = compute_num_blocks(config, total_memory, 0.25)
        assert num_blocks > 0

    def test_zero_memory(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=2,
            num_kv_heads=4,
            head_dim=128,
        )
        num_blocks = compute_num_blocks(config, 0, 0.25)
        assert num_blocks == 0


class TestKVCacheManager:
    def test_create_manager(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=2,
            num_kv_heads=4,
            head_dim=128,
        )
        mgr = KVCacheManager(config, num_blocks=100)
        assert mgr.block_size == 64

    def test_num_free_blocks(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=2,
            num_kv_heads=4,
            head_dim=128,
        )
        mgr = KVCacheManager(config, num_blocks=100)
        assert mgr.num_free_blocks > 95  # 1 block may be reserved

    def test_usage_initially_zero(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=2,
            num_kv_heads=4,
            head_dim=128,
        )
        mgr = KVCacheManager(config, num_blocks=100)
        assert mgr.usage == 0.0

    def test_hit_rate_initially_zero(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=2,
            num_kv_heads=4,
            head_dim=128,
        )
        mgr = KVCacheManager(config, num_blocks=100)
        assert mgr.hit_rate == 0.0
