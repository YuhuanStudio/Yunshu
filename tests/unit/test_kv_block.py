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
