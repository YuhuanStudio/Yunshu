"""Tests for RadixTree eviction strategies."""
import time
import pytest


class TestRadixTreeEvictionStrategies:
    def _make_tree_with_leaves(self, strategy="lru"):
        from yunshu_kv.radix_attention import RadixTree, RadixNode
        tree = RadixTree(eviction_strategy=strategy)
        # Insert some nodes
        node = tree.insert([1, 2, 3], [], [])
        node.creation_time = 100.0
        node.last_access_time = 200.0
        node.access_count = 1

        node2 = tree.insert([1, 2, 4], [], [])
        node2.creation_time = 300.0
        node2.last_access_time = 400.0
        node2.access_count = 5

        node3 = tree.insert([5, 6], [], [])
        node3.creation_time = 50.0
        node3.last_access_time = 60.0
        node3.access_count = 2

        return tree, [node, node2, node3]

    def test_lru_eviction_order(self):
        tree, nodes = self._make_tree_with_leaves("lru")
        # 4 nodes total: [1,2] + [3] + [4] + [5,6]
        assert tree.total_nodes == 4
        freed = tree.evict(1)
        assert len(freed) == 0  # No blocks in these nodes
        assert tree.total_nodes == 3  # One evicted, [1,2] merged with surviving child

    def test_lfu_eviction_order(self):
        tree, nodes = self._make_tree_with_leaves("lfu")
        assert tree.total_nodes == 4
        # LFU: node with access_count=1 should be evicted first
        freed = tree.evict(1)
        # After eviction, merge compacts the tree
        assert tree.total_nodes == 2  # [1,2,4] (merged) + [5,6]

    def test_fifo_eviction_order(self):
        tree, nodes = self._make_tree_with_leaves("fifo")
        assert tree.total_nodes == 4
        # FIFO: node with earliest creation_time (node[2], creation_time=50) evicted first
        freed = tree.evict(1)
        assert tree.total_nodes == 3

    def test_default_is_lru(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        assert tree._eviction_strategy == "lru"

    def test_eviction_respects_ref_count(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree(eviction_strategy="lru")
        node = tree.insert([1, 2, 3], [], [])
        node.creation_time = 100.0
        node.last_access_time = 100.0
        node.ref_count = 1  # Active reference
        freed = tree.evict(1)
        assert tree.total_nodes == 1  # Not evicted because ref_count > 0


class TestRadixNodeAccessTracking:
    def test_access_count_increments(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        node = tree.insert([1, 2, 3], [], [])
        assert node.access_count == 0
        tree.inc_ref(node)
        assert node.access_count == 1
        tree.inc_ref(node)
        assert node.access_count == 2

    def test_creation_time_set(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        node = tree.insert([1, 2, 3], [], [])
        assert node.creation_time > 0
        assert node.last_access_time > 0
