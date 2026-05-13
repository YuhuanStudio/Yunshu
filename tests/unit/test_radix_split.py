"""Tests for RadixTree node splitting and partial prefix matching."""
import pytest


class TestRadixTreeMatch:
    """Test the match() method for prefix lookup."""

    def test_match_empty_tree(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        node, remaining = tree.match([1, 2, 3])
        assert node is tree.root
        assert remaining == [1, 2, 3]

    def test_match_exact(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3], [], [])
        node, remaining = tree.match([1, 2, 3])
        assert remaining == []
        assert node.token_ids == [1, 2, 3]

    def test_match_prefix_of_inserted(self):
        """Matching a prefix of an inserted sequence splits the node.

        Insert [1,2,3,4,5], match [1,2,3]:
        - Child [1,2,3,4,5] partially matches (3/5 tokens)
        - Split creates: [1,2,3] + [4,5]
        - Returns [1,2,3] with remaining=[]
        """
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3, 4, 5], [], [])
        assert tree.total_nodes == 1

        node, remaining = tree.match([1, 2, 3])
        assert remaining == []
        assert node.token_ids == [1, 2, 3]
        # The split created a new intermediate node
        assert tree.total_nodes == 2  # [1,2,3] parent + [4,5] child

    def test_match_no_match(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3], [], [])
        node, remaining = tree.match([4, 5, 6])
        assert node is tree.root
        assert remaining == [4, 5, 6]


class TestRadixTreeSplit:
    """Test node splitting during partial prefix matching."""

    def test_split_on_partial_match(self):
        """When a child partially matches, it should be split.

        Insert [1,2,3], then match [1,2,4]:
        - [1,2,3] should split into [1,2] + [3]
        - Match returns node [1,2] with remaining [4]
        """
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3], [], [])
        node, remaining = tree.match([1, 2, 4])
        assert remaining == [4]
        # The split node has the shared prefix [1,2]
        assert node.token_ids == [1, 2]
        assert tree.total_nodes == 2  # [1,2] parent + [3] child

    def test_split_preserves_children(self):
        """After split, the original child's children are moved to new node."""
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        # Insert [1,2,3] and [1,2,4] — creates two children off [1,2]
        tree.insert([1, 2, 3], [], [])
        tree.insert([1, 2, 4], [], [])

        # Now match [1,2,5] — should split [1,2] off
        node, remaining = tree.match([1, 2, 5])
        assert remaining == [5]
        # The tree should have: root → [1,2] → [3], [4]
        assert tree.total_nodes == 3

    def test_insert_after_split(self):
        """After a split creates a shared prefix node, insert the new suffix."""
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3], [], [])

        # Match [1,2,4] triggers split
        node, remaining = tree.match([1, 2, 4])
        assert remaining == [4]

        # Insert the remaining suffix
        new_node = tree.insert(remaining, [], [], start_node=node)
        assert new_node.token_ids == [4]
        assert tree.total_nodes == 3  # [1,2] + [3] + [4]

        # Now matching [1,2,3] and [1,2,4] both work
        n1, r1 = tree.match([1, 2, 3])
        assert r1 == []
        n2, r2 = tree.match([1, 2, 4])
        assert r2 == []

    def test_split_with_blocks(self):
        """Split correctly divides blocks between new_node and original child."""
        from yunshu_kv.radix_attention import RadixTree, RadixNode
        from yunshu_kv.block import KVBlock

        tree = RadixTree()
        blocks = [KVBlock(block_id=i) for i in range(3)]
        hashes = [100, 200, 300]

        tree.insert([1, 2, 3], blocks, hashes)

        # Match [1,2,4] — splits at position 2
        node, remaining = tree.match([1, 2, 4])
        assert len(node.blocks) == 2  # First 2 blocks go to new_node
        assert node.blocks[0].block_id == 0
        assert node.blocks[1].block_id == 1
        assert node.block_hashes == [100, 200]

        # The original child keeps the remaining block
        assert len(list(node.children.values())[0].blocks) == 1
        assert list(node.children.values())[0].blocks[0].block_id == 2


class TestRadixTreeCompaction:
    """Test post-eviction compaction (merge parent with single child)."""

    def test_merge_after_eviction(self):
        """After evicting a leaf, parent with single child merges."""
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()

        # Insert two paths: [1,2,3] and [1,2,4]
        tree.insert([1, 2, 3], [], [])
        tree.insert([1, 2, 4], [], [])
        # Second insert triggers split: [1,2] + [3] + [4] = 3 nodes
        assert tree.total_nodes == 3

        # Evict one leaf
        freed = tree.evict(1)
        # After evicting [4], parent [1,2] has single child [3] → merge
        assert tree.total_nodes == 1  # [1,2,3] after merge

    def test_no_merge_with_multiple_children(self):
        """Parent with 2+ children should NOT merge after single eviction."""
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()

        tree.insert([1, 2, 3], [], [])
        tree.insert([1, 2, 4], [], [])
        tree.insert([1, 2, 5], [], [])
        # [1,2] + [3] + [4] + [5] = 4 nodes (shared prefix node + 3 leaves)
        assert tree.total_nodes == 4

        tree.evict(1)
        # Only one evicted, parent [1,2] still has 2 children → no merge
        assert tree.total_nodes == 3


class TestRadixTreeRefcounting:
    """Test reference counting for safe sharing."""

    def test_inc_ref_propagates_to_root(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        child = tree.insert([1, 2, 3], [], [])
        intermediate = list(tree.root.children.values())[0]

        tree.inc_ref(child)
        assert child.ref_count == 1
        assert intermediate.ref_count == 1
        assert tree.root.ref_count == 1

    def test_dec_ref_propagates_to_root(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        child = tree.insert([1, 2, 3], [], [])
        intermediate = list(tree.root.children.values())[0]

        tree.inc_ref(child)
        tree.dec_ref(child)
        assert child.ref_count == 0
        assert intermediate.ref_count == 0

    def test_eviction_skips_active_refs(self):
        """Nodes with ref_count > 0 should not be evicted."""
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        node = tree.insert([1, 2, 3], [], [])
        node.last_access_time = 0  # Oldest

        tree.inc_ref(node)
        freed = tree.evict(1)
        assert tree.total_nodes == 1  # Not evicted

        tree.dec_ref(node)
        freed = tree.evict(1)
        assert tree.total_nodes == 0  # Now evicted


class TestRadixTreeStats:
    """Test tree statistics reporting."""

    def test_stats_empty(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        stats = tree.get_stats()
        assert stats["total_nodes"] == 0
        assert stats["total_tokens"] == 0

    def test_stats_after_insert(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3], [], [])
        stats = tree.get_stats()
        assert stats["total_nodes"] == 1
        assert stats["total_tokens"] == 3

    def test_eviction_stats(self):
        from yunshu_kv.radix_attention import RadixTree
        tree = RadixTree()
        tree.insert([1, 2, 3], [], [])
        tree.evict(1)
        stats = tree.get_stats()
        assert stats["eviction_stats"]["lru"] == 1
