"""Tests for KV cache block pool, block table, and manager."""

import pytest

from yunshu_kv.block import BlockPool, FreeBlockQueue, KVBlock
from yunshu_kv.block_table import BlockTable
from yunshu_kv.hash import compute_block_hash, compute_prompt_hashes
from yunshu_kv.manager import KVCacheConfig, KVCacheManager, compute_num_blocks


# ── FreeBlockQueue ──


class TestFreeBlockQueue:
    def test_basic_push_pop(self):
        blocks = [KVBlock(block_id=i) for i in range(5)]
        q = FreeBlockQueue(blocks)
        assert q.num_free_blocks == 5

        b = q.popleft()
        assert b.block_id == 0
        assert q.num_free_blocks == 4

    def test_pop_n(self):
        blocks = [KVBlock(block_id=i) for i in range(5)]
        q = FreeBlockQueue(blocks)
        result = q.popleft_n(3)
        assert len(result) == 3
        assert [b.block_id for b in result] == [0, 1, 2]
        assert q.num_free_blocks == 2

    def test_append_and_lru_order(self):
        blocks = [KVBlock(block_id=i) for i in range(3)]
        q = FreeBlockQueue(blocks)
        # Pop oldest (0), then append it back
        b0 = q.popleft()
        assert b0.block_id == 0
        q.append(b0)
        # Now oldest should be 1
        b1 = q.popleft()
        assert b1.block_id == 1

    def test_remove_specific(self):
        blocks = [KVBlock(block_id=i) for i in range(5)]
        q = FreeBlockQueue(blocks)
        q.remove(blocks[2])
        assert q.num_free_blocks == 4
        # Next pop should be block 0 (oldest)
        assert q.popleft().block_id == 0


# ── BlockPool ──


class TestBlockPool:
    def test_allocate_and_free(self):
        pool = BlockPool(num_blocks=10, block_size=64)
        assert pool.get_free_block_count() == 9  # -1 for null block

        blocks = pool.allocate(3)
        assert len(blocks) == 3
        assert pool.get_free_block_count() == 6

        pool.free(blocks)
        assert pool.get_free_block_count() == 9

    def test_ref_counting(self):
        pool = BlockPool(num_blocks=10, block_size=64)
        block = pool.allocate(1)[0]
        assert block.ref_count == 1

        # Touch to simulate sharing
        pool.touch(block)
        assert block.ref_count == 2

        # First free only decrements
        pool.free([block])
        assert block.ref_count == 1

        # Second free returns to pool
        pool.free([block])
        assert block.ref_count == 0
        assert pool.get_free_block_count() == 9

    def test_prefix_caching(self):
        pool = BlockPool(num_blocks=10, block_size=64, enable_caching=True)

        block = pool.allocate(1)[0]
        pool.cache_block(block, 12345)
        assert block.block_hash == 12345

        # Lookup
        found = pool.lookup_hash(12345)
        assert found is block

        # Different hash → miss
        assert pool.lookup_hash(99999) is None

    def test_allocate_evicts_cached(self):
        pool = BlockPool(num_blocks=5, block_size=64, enable_caching=True)

        # Allocate all available blocks
        blocks = pool.allocate(4)
        for i, b in enumerate(blocks):
            pool.cache_block(b, hash(f"hash_{i}"))

        # Free them all (they go back to free list but remain cached)
        pool.free(blocks)
        assert pool.get_free_block_count() == 4

        # Allocate again — should evict cached blocks
        new_blocks = pool.allocate(4)
        assert len(new_blocks) == 4
        # Old hash lookups should miss
        assert pool.lookup_hash(hash("hash_0")) is None

    def test_usage(self):
        pool = BlockPool(num_blocks=10, block_size=64)
        assert pool.get_usage() == 0.0

        pool.allocate(3)
        assert abs(pool.get_usage() - 3 / 9) < 0.01


# ── Block Hashing ──


class TestBlockHashing:
    def test_chain_hash_deterministic(self):
        tokens = [1, 2, 3, 4]
        h1 = compute_block_hash(None, tokens)
        h2 = compute_block_hash(None, tokens)
        assert h1 == h2

    def test_chain_hash_different_parent(self):
        tokens = [1, 2, 3, 4]
        h1 = compute_block_hash(None, tokens)
        h2 = compute_block_hash(12345, tokens)
        assert h1 != h2

    def test_chain_hash_different_tokens(self):
        h1 = compute_block_hash(None, [1, 2, 3, 4])
        h2 = compute_block_hash(None, [5, 6, 7, 8])
        assert h1 != h2

    def test_prompt_hashes(self):
        tokens = list(range(256))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 4  # 256 / 64 = 4 complete blocks

        # Incomplete trailing tokens are not hashed
        tokens2 = list(range(200))
        hashes2 = compute_prompt_hashes(tokens2, block_size=64)
        assert len(hashes2) == 3  # 192 / 64 = 3, remainder 8 tokens not hashed


# ── BlockTable ──


class TestBlockTable:
    def test_append_and_get(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        table = BlockTable(block_size=16)

        blocks = pool.allocate(3)
        table.append_blocks(blocks)
        assert table.num_blocks == 3
        assert table.get_block(0) is blocks[0]

    def test_slot_for_token(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        table = BlockTable(block_size=16)

        blocks = pool.allocate(2)
        table.append_blocks(blocks)

        block_id, offset = table.slot_for_token(20)
        assert block_id == blocks[1].block_id
        assert offset == 4  # 20 % 16

    def test_fork(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        table = BlockTable(block_size=16)

        blocks = pool.allocate(2)
        table.append_blocks(blocks)

        forked = table.fork()
        assert forked.num_blocks == 2
        # Same physical blocks
        assert forked.get_block(0) is table.get_block(0)

    def test_clear(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        table = BlockTable(block_size=16)

        blocks = pool.allocate(2)
        table.append_blocks(blocks)

        returned = table.clear()
        assert len(returned) == 2
        assert table.num_blocks == 0


# ── KVCacheManager ──


class TestKVCacheManager:
    def test_basic_prefill_and_free(self):
        config = KVCacheConfig(
            block_size=16,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            enable_caching=True,
        )
        mgr = KVCacheManager(config, num_blocks=100)

        # Prefill with 64 tokens → 4 blocks
        tokens = list(range(64))
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert table.num_blocks == 4
        assert match.num_matched_tokens == 0  # first request, no cache hit
        assert len(match.unmatched_token_ids) == 64

        mgr.free_request(table)
        assert mgr.num_free_blocks == 99

    def test_prefix_cache_hit(self):
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=100)

        # First request: 64 tokens
        tokens1 = list(range(64))
        table1, match1 = mgr.allocate_for_prefill(tokens1, model_hash=42)
        assert match1.num_matched_tokens == 0

        # Cache the completed blocks
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)

        # Second request: same prefix + extra 16 tokens
        tokens2 = list(range(64)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)

        assert match2.num_matched_tokens == 64  # full prefix hit
        assert len(match2.unmatched_token_ids) == 16  # only new tokens
        assert table2.num_blocks == 5  # 4 cached + 1 new

        mgr.free_request(table1)
        mgr.free_request(table2)

    def test_partial_prefix_hit(self):
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=100)

        # First request: 48 tokens → 3 full blocks
        tokens1 = list(range(48))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)

        # Second request: 80 tokens, first 48 match
        tokens2 = list(range(48)) + list(range(100, 132))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)

        assert match2.num_matched_tokens == 48
        assert len(match2.unmatched_token_ids) == 32

        mgr.free_request(table1)
        mgr.free_request(table2)

    def test_decode_block_allocation(self):
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=100)

        tokens = list(range(16))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)
        assert table.num_blocks == 1

        # Decode fills up the current block → need a new one
        block = mgr.allocate_block_for_decode(table)
        assert table.num_blocks == 2

        mgr.free_request(table)

    def test_compute_num_blocks(self):
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        # 192 GB UMA, 40 GB weights
        n = compute_num_blocks(config, 192 * 2**30, 40 * 2**30)
        assert n > 0
        # bytes per block = 64 * 32 * 8 * 128 * 2 * 2 = 8,388,608 = 8MB
        # available = (192 - 40) * 0.85 = 129.2 GB
        # blocks = 129.2 GB / 8 MB ≈ 16,691
        assert 15000 < n < 18000
