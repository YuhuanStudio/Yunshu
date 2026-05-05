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
