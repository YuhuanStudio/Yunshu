"""Unit tests for KV block pool."""
import pytest
from yunshu_kv.block import KVBlock, BlockPool, FreeBlockQueue


class TestKVBlock:
    def test_defaults(self):
        b = KVBlock(block_id=0)
        assert b.block_id == 0
        assert b.ref_count == 0
        assert b.block_hash is None
        assert b.is_null is False

    def test_reset_hash(self):
        b = KVBlock(block_id=0, block_hash=12345)
        assert b.block_hash == 12345
        b.reset_hash()
        assert b.block_hash is None


class TestFreeBlockQueue:
    def _make_blocks(self, n):
        return [KVBlock(block_id=i) for i in range(n)]

    def test_popleft_returns_oldest(self):
        blocks = self._make_blocks(3)
        q = FreeBlockQueue(blocks)
        b = q.popleft()
        assert b.block_id == 0
        assert q.num_free_blocks == 2

    def test_popleft_n(self):
        blocks = self._make_blocks(5)
        q = FreeBlockQueue(blocks)
        popped = q.popleft_n(3)
        assert len(popped) == 3
        assert [b.block_id for b in popped] == [0, 1, 2]
        assert q.num_free_blocks == 2

    def test_empty_queue_raises(self):
        blocks = self._make_blocks(1)
        q = FreeBlockQueue(blocks)
        q.popleft()
        with pytest.raises(AssertionError):
            q.popleft()

    def test_append_and_popleft(self):
        blocks = self._make_blocks(2)
        q = FreeBlockQueue(blocks)
        b = q.popleft()
        q.append(b)
        assert q.num_free_blocks == 2


class TestBlockPool:
    def test_allocate(self):
        pool = BlockPool(num_blocks=10, block_size=4)
        # One block reserved as null, so 9 free
        blocks = pool.allocate(3)
        assert len(blocks) == 3
        assert pool.get_free_block_count() == 6

    def test_allocate_all_free(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        # 1 null + 4 free
        blocks = pool.allocate(4)
        assert len(blocks) == 4
        assert pool.get_free_block_count() == 0

    def test_allocate_exhausted_raises(self):
        pool = BlockPool(num_blocks=3, block_size=4)
        # 1 null + 2 free
        pool.allocate(2)
        with pytest.raises(ValueError):
            pool.allocate(1)

    def test_allocate_sets_ref_count(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        blocks = pool.allocate(2)
        assert blocks[0].ref_count == 1
        assert blocks[1].ref_count == 1

    def test_free_decrements_ref(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        blocks = pool.allocate(2)
        pool.free(blocks)
        # Ref count goes to 0, blocks return to free list
        assert blocks[0].ref_count == 0
        assert pool.get_free_block_count() == 4

    def test_free_and_reuse(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        b1 = pool.allocate(2)
        pool.free(b1)
        b2 = pool.allocate(2)
        assert len(b2) == 2

    def test_touch_increments_ref(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        blocks = pool.allocate(1)
        pool.touch(blocks[0])
        assert blocks[0].ref_count == 2

    def test_cache_block_sets_hash(self):
        pool = BlockPool(num_blocks=5, block_size=4, enable_caching=True)
        blocks = pool.allocate(1)
        pool.cache_block(blocks[0], 0xDEAD)
        assert blocks[0].block_hash == 0xDEAD

    def test_lookup_hash(self):
        pool = BlockPool(num_blocks=5, block_size=4, enable_caching=True)
        blocks = pool.allocate(1)
        pool.cache_block(blocks[0], 0xBEEF)
        found = pool.lookup_hash(0xBEEF)
        assert found is blocks[0]

    def test_lookup_hash_miss(self):
        pool = BlockPool(num_blocks=5, block_size=4, enable_caching=True)
        assert pool.lookup_hash(0x1234) is None

    def test_get_usage_empty(self):
        pool = BlockPool(num_blocks=10, block_size=4)
        # Nothing allocated (null block excluded from total)
        assert pool.get_usage() == 0.0

    def test_get_usage_partial(self):
        pool = BlockPool(num_blocks=10, block_size=4)
        # 1 null + 9 free, allocate 3 → 6 free out of 9 total
        pool.allocate(3)
        assert abs(pool.get_usage() - 3/9) < 0.01

    def test_get_free_block_count_initial(self):
        pool = BlockPool(num_blocks=10, block_size=4)
        assert pool.get_free_block_count() == 9

    def test_auto_evict_cached_blocks(self):
        pool = BlockPool(num_blocks=4, block_size=4, enable_caching=True)
        # 1 null + 3 free
        blocks = pool.allocate(3)
        pool.cache_block(blocks[0], 0xA1)
        pool.cache_block(blocks[1], 0xA2)
        pool.free(blocks)

        # All 3 freed, allocate 3 again (evicts cached from free list)
        new_blocks = pool.allocate(3)
        assert len(new_blocks) == 3

    def test_caching_disabled(self):
        pool = BlockPool(num_blocks=5, block_size=4, enable_caching=False)
        blocks = pool.allocate(1)
        pool.cache_block(blocks[0], 0xDEAD)
        assert blocks[0].block_hash is None

    def test_null_block_reserved(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        assert pool.null_block.is_null is True
        assert pool.null_block.block_id == 0

    def test_reset_prefix_cache(self):
        pool = BlockPool(num_blocks=5, block_size=4, enable_caching=True)
        blocks = pool.allocate(1)
        pool.cache_block(blocks[0], 0xBEEF)
        pool.reset_prefix_cache()
        assert pool.lookup_hash(0xBEEF) is None


class TestBlockPoolCOW:
    """Tests for copy-on-write (COW) in BlockPool."""

    def test_cow_noop_for_exclusive_block(self):
        """COW on a block with ref_count == 1 returns the same block."""
        pool = BlockPool(num_blocks=10, block_size=4)
        blocks = pool.allocate(1)
        assert blocks[0].ref_count == 1

        result = pool.cow_block(blocks[0])
        assert result is blocks[0]
        assert pool.cow_stats["cow_clones"] == 0

    def test_cow_clones_shared_block(self):
        """COW on a block with ref_count > 1 allocates a new block."""
        pool = BlockPool(num_blocks=10, block_size=4)
        blocks = pool.allocate(1)

        # Simulate sharing: touch the block (ref_count → 2)
        pool.touch(blocks[0])
        assert blocks[0].ref_count == 2

        result = pool.cow_block(blocks[0])
        assert result is not blocks[0]
        assert result.ref_count == 1
        assert blocks[0].ref_count == 1
        assert pool.cow_stats["cow_clones"] == 1
        assert pool.get_free_block_count() == 7  # 9 - 1 (original alloc) - 1 (cow)

    def test_cow_shared_block_freed_when_zero(self):
        """COW decrements original ref_count; last sharer gets it exclusive."""
        pool = BlockPool(num_blocks=10, block_size=4)
        blocks = pool.allocate(1)

        # Three shares (ref_count → 3): requests A, B, and C all share this block
        pool.touch(blocks[0])
        pool.touch(blocks[0])
        assert blocks[0].ref_count == 3

        # First COW (request A detaches): ref_count → 2
        result1 = pool.cow_block(blocks[0])
        assert result1 is not blocks[0]
        assert result1.ref_count == 1
        assert blocks[0].ref_count == 2

        # Second COW (request B detaches): ref_count → 1
        result2 = pool.cow_block(blocks[0])
        assert result2 is not blocks[0]
        assert blocks[0].ref_count == 1

        # Third COW (request C is the last sharer): no-op, returns original
        result3 = pool.cow_block(blocks[0])
        assert result3 is blocks[0]  # Exclusive already, no clone needed
        assert blocks[0].ref_count == 1

        # Free the last sharer → ref_count → 0, block goes back to free pool
        pool.free([blocks[0]])
        assert blocks[0].ref_count == 0
        assert pool.cow_stats["cow_clones"] == 2

    def test_cow_raises_when_exhausted(self):
        """COW raises ValueError when no free blocks available."""
        pool = BlockPool(num_blocks=3, block_size=4)
        # 1 null + 2 free
        blocks = pool.allocate(2)
        pool.touch(blocks[0])
        assert pool.get_free_block_count() == 0

        with pytest.raises(ValueError, match="COW failed"):
            pool.cow_block(blocks[0])

    def test_cow_in_table_replaces_entry(self):
        """cow_block_in_table updates the BlockTable entry."""
        from yunshu_kv.block_table import BlockTable

        pool = BlockPool(num_blocks=10, block_size=4)
        blocks = pool.allocate(2)
        pool.touch(blocks[0])  # ref_count → 2

        table = BlockTable(block_size=4)
        table.append_block(blocks[0])
        table.append_block(blocks[1])

        # COW the first block in the table
        new_block = pool.cow_block_in_table(table, 0)
        assert new_block is not blocks[0]
        assert table.get_block(0) is new_block
        assert table.get_block(1) is blocks[1]  # Unchanged
        assert new_block.ref_count == 1

    def test_cow_in_table_noop_for_exclusive(self):
        """cow_block_in_table is a no-op for exclusive blocks."""
        from yunshu_kv.block_table import BlockTable

        pool = BlockPool(num_blocks=10, block_size=4)
        blocks = pool.allocate(1)

        table = BlockTable(block_size=4)
        table.append_block(blocks[0])

        result = pool.cow_block_in_table(table, 0)
        assert result is blocks[0]
        assert pool.cow_stats["cow_clones"] == 0
