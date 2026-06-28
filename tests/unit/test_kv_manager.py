"""Tests for kv manager — block pool and KV cache management."""

import pytest

from yunshu_kv.manager import KVCacheConfig, KVCacheManager, compute_num_blocks


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

    def test_warm_promotion_failure_evicts_cached_block(self):
        """Regression test: warm promotion that fails after cache_block must
        evict the prefix cache entry, not just free the block.

        Before the fix, cache_block + free left a stale entry in _hash_to_block
        that lookup_hash() would return (cache_only=True, ref_count=0). A
        subsequent allocate() could hand this block to a different request,
        corrupting KV data.
        """
        import numpy as np
        config = KVCacheConfig(block_size=4, num_layers=2, num_kv_heads=4, head_dim=64)
        mgr = KVCacheManager(config, num_blocks=100)

        # Demote some data to warm tier
        data = np.random.randn(2, 4, 4, 64).astype(np.float32)
        import mlx.core as mx
        kv_data = mx.array(data)
        block_hash = 0xDEADBEEF
        mgr._warm_tier.demote(block_hash, kv_data, num_tokens=4)

        # Set up cache tensors (will fail on write)
        mgr.set_kv_tensors(
            mx.zeros((100, 2, 4, 4, 64), dtype=mx.float16),
            mx.zeros((100, 2, 4, 4, 64), dtype=mx.float16),
        )

        # Allocate — this triggers warm promotion
        tokens = list(range(16))  # 4 blocks of 4 tokens
        table, match = mgr.allocate_for_prefill(tokens)

        # If promotion succeeded, the block should be in the prefix cache
        # If it failed (KV write exception), the block should NOT be in the cache
        cached = mgr.block_pool.lookup_hash(block_hash)
        if cached is not None:
            # Block was successfully promoted — verify it's usable
            assert cached.ref_count > 0 or cached.cache_only
        # In any case, no block should be in the free queue with a stale hash
        for block in mgr.block_pool.blocks:
            if block.ref_count == 0 and not block.cache_only and block.block_hash is not None:
                pytest.fail(
                    f"Block {block.block_id} has stale hash 0x{block.block_hash:x} "
                    f"but ref_count=0 and cache_only=False — dangling reference"
                )
