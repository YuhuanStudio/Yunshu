"""Tests for KV Cache management."""

import pytest
import tempfile
import os

from yunshu_kv.tiered import SSDCacheStore


class TestSSDCacheStore:
    """Test SSD-backed KV cache storage."""

    def test_create_and_store(self, tmp_path):
        import mlx.core as mx

        store = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        data = mx.ones((2, 64), dtype=mx.float16)
        assert store.store(block_hash=12345, kv_data=data, num_tokens=64)

    def test_store_and_contains(self, tmp_path):
        import mlx.core as mx

        store = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        data = mx.ones((2, 64), dtype=mx.float16)
        store.store(block_hash=12345, kv_data=data, num_tokens=64)
        assert store.contains(12345)
        assert not store.contains(99999)

    def test_store_and_load(self, tmp_path):
        import mlx.core as mx

        store = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        data = mx.ones((2, 64), dtype=mx.float16)
        store.store(block_hash=12345, kv_data=data, num_tokens=64)

        loaded = store.load(12345)
        assert loaded is not None
        assert loaded.shape == (2, 64)

    def test_load_missing(self, tmp_path):
        import mlx.core as mx

        store = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        assert store.load(99999) is None

    def test_store_dedup(self, tmp_path):
        import mlx.core as mx

        store = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        data = mx.ones((2, 64), dtype=mx.float16)
        store.store(block_hash=12345, kv_data=data, num_tokens=64)
        store.store(block_hash=12345, kv_data=data, num_tokens=64)
        assert store.get_stats()["num_entries"] == 1

    def test_stats(self, tmp_path):
        import mlx.core as mx

        store = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        data = mx.ones((2, 64), dtype=mx.float16)
        store.store(block_hash=1, kv_data=data, num_tokens=64)
        store.store(block_hash=2, kv_data=data, num_tokens=64)

        stats = store.get_stats()
        assert stats["num_entries"] == 2
        assert stats["size_bytes"] > 0

    def test_persistence(self, tmp_path):
        import mlx.core as mx

        store1 = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        data = mx.ones((2, 64), dtype=mx.float16)
        store1.store(block_hash=12345, kv_data=data, num_tokens=64)
        store1._save_index()

        store2 = SSDCacheStore(cache_dir=str(tmp_path), max_size_bytes=1024 * 1024)
        assert store2.contains(12345)


class TestKVBlock:
    """Test KV cache block operations."""

    def test_block_creation(self):
        from yunshu_kv.block import KVBlock

        block = KVBlock(block_id=0)
        assert block.block_id == 0
        assert block.ref_count == 0

    def test_block_pool(self):
        from yunshu_kv.block import BlockPool

        pool = BlockPool(num_blocks=100, block_size=16)
        assert pool.get_free_block_count() == 99  # -1 for null block

    def test_block_pool_allocate(self):
        from yunshu_kv.block import BlockPool

        pool = BlockPool(num_blocks=100, block_size=16)
        blocks = pool.allocate(5)
        assert len(blocks) == 5
        assert pool.get_free_block_count() == 94

    def test_block_pool_free(self):
        from yunshu_kv.block import BlockPool

        pool = BlockPool(num_blocks=100, block_size=16)
        blocks = pool.allocate(5)
        pool.free(blocks)
        assert pool.get_free_block_count() == 99

    def test_block_pool_prefix_caching(self):
        from yunshu_kv.block import BlockPool

        pool = BlockPool(num_blocks=100, block_size=16)
        blocks = pool.allocate(1)
        pool.cache_block(blocks[0], 12345)
        found = pool.lookup_hash(12345)
        assert found is blocks[0]


class TestBlockHash:
    """Test KV block hashing."""

    def test_chain_hash(self):
        from yunshu_kv.hash import compute_block_hash

        h1 = compute_block_hash(None, [1, 2, 3, 4])
        h2 = compute_block_hash(None, [1, 2, 3, 4])
        assert h1 == h2

    def test_chain_hash_different(self):
        from yunshu_kv.hash import compute_block_hash

        h1 = compute_block_hash(None, [1, 2, 3, 4])
        h2 = compute_block_hash(None, [5, 6, 7, 8])
        assert h1 != h2

    def test_parent_hash_changes(self):
        from yunshu_kv.hash import compute_block_hash

        h1 = compute_block_hash(0, [1, 2, 3])
        h2 = compute_block_hash(999, [1, 2, 3])
        assert h1 != h2

    def test_prompt_hashes(self):
        from yunshu_kv.hash import compute_prompt_hashes

        tokens = list(range(64))
        hashes = compute_prompt_hashes(tokens, block_size=16)
        assert len(hashes) == 4

    def test_prompt_hashes_partial(self):
        from yunshu_kv.hash import compute_prompt_hashes

        tokens = list(range(20))
        hashes = compute_prompt_hashes(tokens, block_size=16)
        assert len(hashes) == 1  # Only 1 complete block
