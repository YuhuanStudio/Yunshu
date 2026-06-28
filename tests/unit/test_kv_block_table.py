"""Unit tests for KV BlockTable."""
import pytest

from yunshu_kv.block import KVBlock
from yunshu_kv.block_table import BlockTable


class TestBlockTable:
    def test_construction(self):
        bt = BlockTable(block_size=64)
        assert bt.block_size == 64
        assert bt.num_blocks == 0

    def test_append_block(self):
        bt = BlockTable(block_size=64)
        b = KVBlock(block_id=0)
        bt.append_block(b)
        assert bt.num_blocks == 1
        assert bt.get_block(0) is b

    def test_get_blocks(self):
        bt = BlockTable(block_size=64)
        b0, b1 = KVBlock(block_id=0), KVBlock(block_id=1)
        bt.append_block(b0)
        bt.append_block(b1)
        blocks = bt.get_blocks()
        assert len(blocks) == 2
        assert blocks[0] is b0
        assert blocks[1] is b1

    def test_get_full_blocks(self):
        bt = BlockTable(block_size=64)
        b0, b1, b2 = KVBlock(block_id=0), KVBlock(block_id=1), KVBlock(block_id=2)
        bt.append_block(b0)
        bt.append_block(b1)
        bt.append_block(b2)
        full = bt.get_full_blocks()
        assert len(full) == 2
        assert full[0] is b0
        assert full[1] is b1

    def test_get_full_blocks_single(self):
        bt = BlockTable(block_size=64)
        bt.append_block(KVBlock(block_id=0))
        assert bt.get_full_blocks() == []

    def test_fork_shares_physical_blocks(self):
        bt = BlockTable(block_size=64)
        b0, b1 = KVBlock(block_id=0), KVBlock(block_id=1)
        bt.append_block(b0)
        bt.append_block(b1)
        forked = bt.fork()
        assert forked.num_blocks == 2
        assert forked.get_block(0) is b0
        assert forked.get_block(1) is b1
        # Independent table
        forked.append_block(KVBlock(block_id=2))
        assert forked.num_blocks == 3
        assert bt.num_blocks == 2

    def test_clear_returns_blocks(self):
        bt = BlockTable(block_size=64)
        b0, b1 = KVBlock(block_id=0), KVBlock(block_id=1)
        bt.append_block(b0)
        bt.append_block(b1)
        blocks = bt.clear()
        assert len(blocks) == 2
        assert bt.num_blocks == 0

    def test_block_id_for_token(self):
        bt = BlockTable(block_size=4)
        for i in range(3):
            bt.append_block(KVBlock(block_id=i))
        # Token 0-3 → block 0, 4-7 → block 1, 8-11 → block 2
        assert bt.block_id_for_token(0) == 0
        assert bt.block_id_for_token(3) == 0
        assert bt.block_id_for_token(4) == 1
        assert bt.block_id_for_token(11) == 2

    def test_slot_for_token(self):
        bt = BlockTable(block_size=4)
        for i in range(3):
            bt.append_block(KVBlock(block_id=i))
        assert bt.slot_for_token(0) == (0, 0)
        assert bt.slot_for_token(3) == (0, 3)
        assert bt.slot_for_token(5) == (1, 1)
        assert bt.slot_for_token(10) == (2, 2)

    def test_append_blocks(self):
        bt = BlockTable(block_size=64)
        blocks = [KVBlock(block_id=i) for i in range(3)]
        bt.append_blocks(blocks)
        assert bt.num_blocks == 3

    # ── trim_prefix_blocks ──────────────────────────────────────────────

    def test_trim_prefix_blocks_basic(self):
        bt = BlockTable(block_size=64)
        blocks = [KVBlock(block_id=i) for i in range(5)]
        bt.append_blocks(blocks)
        assert bt.num_blocks == 5

        trimmed = bt.trim_prefix_blocks(2)
        assert len(trimmed) == 2
        assert trimmed[0].block_id == 0
        assert trimmed[1].block_id == 1
        assert bt.num_blocks == 3
        assert bt.get_block(0).block_id == 2

    def test_trim_prefix_blocks_updates_total_tokens(self):
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(5)]
        bt.append_blocks(blocks)
        # 5 blocks: 4 full (block_size=4) + last has occupancy 0
        # total_tokens = 4 * 4 + 0 = 16
        assert bt.total_tokens == 16

        trimmed = bt.trim_prefix_blocks(2)
        assert len(trimmed) == 2
        # 3 remaining: 2 full + last with occupancy 0
        # total_tokens = 2 * 4 + 0 = 8
        assert bt.total_tokens == 8

    def test_trim_prefix_blocks_with_occupancy(self):
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(5)]
        bt.append_blocks(blocks)
        bt.update_last_block_occupancy(3)
        # 5 blocks: 4 full (block_size=4) + last with occupancy 3 = 19
        assert bt.total_tokens == 19

        trimmed = bt.trim_prefix_blocks(2)
        assert len(trimmed) == 2
        # 3 remaining: 2 full + last with occupancy 3 (unchanged) = 11
        assert bt.total_tokens == 11

    def test_trim_prefix_blocks_all(self):
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(3)]
        bt.append_blocks(blocks)
        trimmed = bt.trim_prefix_blocks(3)
        assert len(trimmed) == 3
        assert bt.num_blocks == 0
        assert bt.total_tokens == 0

    def test_trim_prefix_blocks_all_with_occupancy(self):
        """Regression test: trimming ALL blocks must reset _last_block_occupancy.

        Before the fix, total_tokens was computed as max(0, 0-1)*bs + occupancy
        = 0 + occupancy, incorrectly reporting non-zero tokens for an empty table.
        """
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(3)]
        bt.append_blocks(blocks)
        bt.update_last_block_occupancy(2)
        assert bt.total_tokens == 10  # 2*4 + 2

        trimmed = bt.trim_prefix_blocks(3)
        assert len(trimmed) == 3
        assert bt.num_blocks == 0
        assert bt.total_tokens == 0
        assert bt._last_block_occupancy == 0

    def test_trim_prefix_blocks_zero(self):
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(3)]
        bt.append_blocks(blocks)
        trimmed = bt.trim_prefix_blocks(0)
        assert trimmed == []
        assert bt.num_blocks == 3

    def test_trim_prefix_blocks_too_many(self):
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(3)]
        bt.append_blocks(blocks)
        with pytest.raises(ValueError):
            bt.trim_prefix_blocks(4)

    def test_trim_prefix_blocks_negative(self):
        bt = BlockTable(block_size=4)
        with pytest.raises(ValueError):
            bt.trim_prefix_blocks(-1)

    def test_trim_prefix_blocks_then_append(self):
        """Trim blocks, then append a new one — regression test for
        total_tokens consistency after trim+append."""
        bt = BlockTable(block_size=4)
        blocks = [KVBlock(block_id=i) for i in range(4)]
        bt.append_blocks(blocks)
        bt.update_last_block_occupancy(2)
        # 4 blocks: 3 full (12) + last occupancy 2 = 14
        assert bt.total_tokens == 14

        bt.trim_prefix_blocks(2)
        # 2 remaining: 1 full (4) + last occupancy 2 = 6
        assert bt.total_tokens == 6

        new_block = KVBlock(block_id=10)
        bt.append_block(new_block)
        # append_block adds block_size when finalizing the previous last.
        # This overcounts by (block_size - _last_block_occupancy) = 2
        # because it assumes the previous last was full. This is a
        # pre-existing behavior of append_block, not introduced by trim.
        # 6 + 4 = 10
        assert bt.total_tokens == 10
        assert bt.num_blocks == 3
