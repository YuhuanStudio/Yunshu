"""Tests for RadixAttention tree-based prefix sharing."""

import pytest

from yunshu_kv.block import KVBlock
from yunshu_kv.radix_attention import RadixNode, RadixTree


class TestRadixNode:
    def test_basic_properties(self):
        node = RadixNode(token_ids=[1, 2, 3])
        assert node.num_tokens == 3
        assert node.is_leaf
        assert node.is_root is False

    def test_root_node(self):
        root = RadixNode()
        assert root.is_root
        assert root.num_tokens == 0


class TestRadixTree:
    def test_empty_tree_match(self):
        tree = RadixTree()
        node, remaining = tree.match([1, 2, 3])
        assert node is tree.root
        assert remaining == [1, 2, 3]

    def test_insert_and_match_full(self):
        tree = RadixTree()
        blocks = [KVBlock(block_id=i) for i in range(2)]
        tree.insert([1, 2, 3, 4], blocks[:2], [100, 200])

        node, remaining = tree.match([1, 2, 3, 4])
        assert node.token_ids == [1, 2, 3, 4]
        assert remaining == []
        assert node.blocks == blocks[:2]

    def test_partial_match(self):
        tree = RadixTree()
        blocks = [KVBlock(block_id=i) for i in range(2)]
        tree.insert([1, 2, 3, 4], blocks, [100, 200])

        # Match only first 2 tokens — with node splitting, this should
        # split the [1,2,3,4] node into [1,2] + [3,4] and return [1,2]
        node, remaining = tree.match([1, 2])
        assert node.token_ids == [1, 2]  # Split node with matched prefix
        assert remaining == []

    def test_extend_after_match(self):
        tree = RadixTree()
        # Insert first path
        b1 = [KVBlock(block_id=0)]
        tree.insert([1, 2, 3], b1, [100])

        # Insert second path with different prefix (no conflict at first token)
        b2 = [KVBlock(block_id=1)]
        tree.insert([4, 5, 6], b2, [200])

        # Both should match independently
        n1, r1 = tree.match([1, 2, 3])
        assert r1 == []
        assert n1.token_ids == [1, 2, 3]

        n2, r2 = tree.match([4, 5, 6])
        assert r2 == []
        assert n2.token_ids == [4, 5, 6]

    def test_ref_counting(self):
        tree = RadixTree()
        blocks = [KVBlock(block_id=0)]
        node = tree.insert([1, 2, 3], blocks, [100])

        tree.inc_ref(node)
        assert node.ref_count == 1
        assert tree.root.ref_count == 1  # parent also incremented

        tree.dec_ref(node)
        assert node.ref_count == 0
        assert tree.root.ref_count == 0

    def test_eviction(self):
        tree = RadixTree()

        # Insert two paths
        b1 = [KVBlock(block_id=0)]
        b2 = [KVBlock(block_id=1)]
        n1 = tree.insert([1, 2, 3], b1, [100])
        n2 = tree.insert([4, 5, 6], b2, [200])

        # Evict one (oldest first — n1 was inserted earlier)
        freed = tree.evict(1)
        assert len(freed) == 1
        assert freed[0].block_id == 0

        # n2 should still be in the tree
        _, remaining = tree.match([4, 5, 6])
        assert remaining == []

    def test_eviction_respects_ref_count(self):
        tree = RadixTree()
        blocks = [KVBlock(block_id=0)]
        node = tree.insert([1, 2, 3], blocks, [100])
        tree.inc_ref(node)

        # Should not evict (ref_count > 0)
        freed = tree.evict(1)
        assert len(freed) == 0

        tree.dec_ref(node)
        freed = tree.evict(1)
        assert len(freed) == 1

    def test_stats(self):
        tree = RadixTree()
        tree.insert([1, 2, 3], [KVBlock(block_id=0)], [100])
        tree.insert([4, 5, 6], [KVBlock(block_id=1)], [200])

        stats = tree.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_blocks"] == 2
        assert stats["total_tokens"] == 6

    def test_multi_turn_conversation(self):
        """Simulate multi-turn conversation prefix sharing."""
        tree = RadixTree()

        # Turn 1: system + user prompt
        turn1 = list(range(100))  # 100 tokens
        b1 = [KVBlock(block_id=i) for i in range(7)]  # 100/16 ≈ 7 blocks (block_size=16)
        n1 = tree.insert(turn1, b1[:6], [100 + i for i in range(6)])

        # Turn 2: same prefix + response + new user prompt
        turn2 = turn1 + list(range(200, 230))  # 100 old + 30 new
        matched, remaining = tree.match(turn2)
        assert matched.token_ids == turn1
        assert remaining == list(range(200, 230))

        # Insert the new part
        b2 = [KVBlock(block_id=10 + i) for i in range(2)]
        tree.insert(remaining, b2, [300, 301], start_node=matched)

        # Turn 3: same prefix should still match
        turn3 = turn1 + list(range(200, 230)) + list(range(300, 310))
        matched3, remaining3 = tree.match(turn3)
        # Should match all the way to the end of turn2's new tokens
        assert len(remaining3) == 10  # only the newest 10 tokens
