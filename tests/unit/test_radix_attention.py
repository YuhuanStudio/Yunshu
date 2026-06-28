"""Unit tests for radix attention tree with node splitting."""

from yunshu_kv.block import KVBlock
from yunshu_kv.radix_attention import RadixNode, RadixTree


class TestRadixNode:
    def test_defaults(self):
        n = RadixNode()
        assert n.token_ids == []
        assert n.blocks == []
        assert n.ref_count == 0
        assert n.is_leaf is True
        assert n.is_root is True

    def test_num_tokens(self):
        n = RadixNode(token_ids=[1, 2, 3])
        assert n.num_tokens == 3

    def test_num_blocks(self):
        n = RadixNode(blocks=[KVBlock(block_id=0), KVBlock(block_id=1)])
        assert n.num_blocks == 2

    def test_is_leaf(self):
        n = RadixNode()
        child = RadixNode(token_ids=[1], parent=n)
        n.children[1] = child
        assert n.is_leaf is False
        assert child.is_leaf is True

    def test_is_root(self):
        root = RadixNode()
        child = RadixNode(token_ids=[1], parent=root)
        assert root.is_root is True
        assert child.is_root is False

    def test_total_tokens(self):
        root = RadixNode()
        mid = RadixNode(token_ids=[1, 2], parent=root)
        leaf = RadixNode(token_ids=[3, 4, 5], parent=mid)
        root.children[1] = mid
        mid.children[3] = leaf
        assert leaf.total_tokens() == 5
        assert mid.total_tokens() == 2

    def test_path_blocks(self):
        b0, b1, b2 = KVBlock(block_id=0), KVBlock(block_id=1), KVBlock(block_id=2)
        root = RadixNode()
        mid = RadixNode(token_ids=[1], blocks=[b0, b1], parent=root)
        leaf = RadixNode(token_ids=[2], blocks=[b2], parent=mid)
        root.children[1] = mid
        mid.children[2] = leaf
        blocks = leaf.path_blocks()
        # root→leaf token order: mid's blocks [b0,b1] in order, then leaf's [b2].
        # (This test previously asserted [1,0,2] — it had ENCODED the flat-reverse bug
        # that scrambled intra-node block order; the correct order is [0,1,2].)
        assert [b.block_id for b in blocks] == [0, 1, 2]


class TestRadixTree:
    def test_empty_tree(self):
        t = RadixTree()
        assert t.total_nodes == 0

    def test_match_empty_tree(self):
        t = RadixTree()
        node, remaining = t.match([1, 2, 3])
        assert node is t.root
        assert remaining == [1, 2, 3]

    def test_insert_and_match(self):
        t = RadixTree()
        b0 = KVBlock(block_id=0)
        leaf = t.insert([1, 2, 3], [b0], [0xAAA])
        assert t.total_nodes == 1

        node, remaining = t.match([1, 2, 3, 4, 5])
        assert node is leaf
        assert remaining == [4, 5]

    def test_match_exact(self):
        t = RadixTree()
        leaf = t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        node, remaining = t.match([1, 2, 3])
        assert node is leaf
        assert remaining == []

    def test_match_no_match(self):
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        node, remaining = t.match([9, 9, 9])
        assert node is t.root
        assert remaining == [9, 9, 9]

    def test_multiple_branches(self):
        t = RadixTree()
        leaf_a = t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xA1])
        leaf_b = t.insert([4, 5, 6], [KVBlock(block_id=1)], [0xB1])

        node_a, rem_a = t.match([1, 2, 3, 7])
        assert node_a is leaf_a
        assert rem_a == [7]

        node_b, rem_b = t.match([4, 5, 6, 8])
        assert node_b is leaf_b
        assert rem_b == [8]

    def test_ref_counting(self):
        t = RadixTree()
        leaf = t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        t.inc_ref(leaf)
        assert leaf.ref_count == 1
        assert t.root.ref_count == 1
        t.inc_ref(leaf)
        assert leaf.ref_count == 2
        t.dec_ref(leaf)
        assert leaf.ref_count == 1
        t.dec_ref(leaf)
        assert leaf.ref_count == 0
        assert t.root.ref_count == 0

    def test_evict(self):
        t = RadixTree()
        b0 = KVBlock(block_id=0)
        b1 = KVBlock(block_id=1)
        t.insert([1, 2], [b0], [0xA1])
        t.insert([3, 4], [b1], [0xB1])
        assert t.total_nodes == 2

        freed = t.evict(1)
        assert len(freed) == 1
        assert t.total_nodes == 1

    def test_evict_respects_ref_count(self):
        t = RadixTree()
        leaf = t.insert([1, 2], [KVBlock(block_id=0)], [0xAAA])
        t.inc_ref(leaf)
        freed = t.evict(1)
        assert len(freed) == 0

    def test_get_stats(self):
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        stats = t.get_stats()
        assert stats["total_nodes"] == 1
        assert stats["total_blocks"] == 1
        assert stats["total_tokens"] == 3

    def test_insert_from_start_node(self):
        t = RadixTree()
        mid = t.insert([1, 2], [KVBlock(block_id=0)], [0xA1])
        leaf = t.insert([3, 4], [KVBlock(block_id=1)], [0xB1], start_node=mid)
        assert t.total_nodes == 2
        node, remaining = t.match([1, 2, 3, 4, 5])
        assert node is leaf
        assert remaining == [5]


class TestRadixNodeSplitting:
    """Test node splitting in the radix tree for prefix sharing."""

    def test_split_on_partial_match(self):
        """When query partially matches a child, the child should be split.

        Insert [1,2,3], then query [1,2,9]. The node should split into
        [1,2] (shared) + [3] (original suffix) with [9] as new branch.
        """
        t = RadixTree()
        b0 = KVBlock(block_id=0)
        t.insert([1, 2, 3], [b0], [0xAAA])

        # Query with partial match
        node, remaining = t.match([1, 2, 9])
        # After splitting, the matched node should have tokens [1,2]
        assert node.token_ids == [1, 2]
        assert remaining == [9]
        # The original child should be shortened to [3]
        assert 3 in node.children
        suffix_child = node.children[3]
        assert suffix_child.token_ids == [3]

    def test_split_increments_node_count(self):
        """Splitting creates a new intermediate node."""
        t = RadixTree()
        t.insert([1, 2, 3, 4], [KVBlock(block_id=0)], [0xAAA])
        assert t.total_nodes == 1

        # Trigger split
        node, _ = t.match([1, 2, 9])
        assert t.total_nodes == 2  # +1 for the new intermediate node

    def test_split_preserves_original_child_data(self):
        """After split, the original child retains its suffix data."""
        t = RadixTree()
        b0 = KVBlock(block_id=0)
        t.insert([1, 2, 3, 4], [b0], [0xAAA])

        node, remaining = t.match([1, 2, 9])
        # The split node has [1,2]
        assert node.token_ids == [1, 2]
        # Original child now has [3,4]
        assert 3 in node.children
        child = node.children[3]
        assert child.token_ids == [3, 4]

    def test_split_with_multiple_children(self):
        """Split a node that has other siblings."""
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xA1])
        leaf_b = t.insert([5, 6, 7], [KVBlock(block_id=1)], [0xB1])

        # Trigger split on leaf_a
        node, remaining = t.match([1, 2, 9])
        assert node.token_ids == [1, 2]
        assert remaining == [9]

        # leaf_b should still be intact
        node_b, rem_b = t.match([5, 6, 7, 8])
        assert node_b is leaf_b
        assert rem_b == [8]

    def test_split_then_insert_divergent_path(self):
        """After splitting, insert a new branch from the split point."""
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xA1])

        # Split
        node, remaining = t.match([1, 2, 9])
        assert remaining == [9]

        # Insert divergent branch from the split point
        new_leaf = t.insert([9, 10], [KVBlock(block_id=1)], [0xB1], start_node=node)

        # Verify both paths work
        node_a, rem_a = t.match([1, 2, 3, 4])
        assert rem_a == [4]

        node_b, rem_b = t.match([1, 2, 9, 10, 11])
        assert node_b is new_leaf
        assert rem_b == [11]

    def test_split_at_first_token(self):
        """Split at position 1 (only first token matches)."""
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xA1])

        node, remaining = t.match([1, 9, 9])
        assert node.token_ids == [1]
        assert remaining == [9, 9]

    def test_match_partial_prefix_now_succeeds(self):
        """Previously partial match returned early; now it splits."""
        t = RadixTree()
        t.insert([1, 2, 3, 4], [KVBlock(block_id=0)], [0xAAA])

        # Query [1, 2] should now split and return matched [1,2] node
        node, remaining = t.match([1, 2])
        assert node.token_ids == [1, 2]
        assert remaining == []

    def test_split_ref_count_inheritance(self):
        """Split node inherits ref_count from original child."""
        t = RadixTree()
        leaf = t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        t.inc_ref(leaf)
        assert leaf.ref_count == 1

        # Trigger split
        node, _ = t.match([1, 2, 9])
        # New intermediate node should have inherited ref_count
        assert node.ref_count == 1

    def test_evict_after_split_with_merge(self):
        """After evicting a leaf from a split node, try merge compaction."""
        t = RadixTree()
        # Insert two branches that share prefix [1,2]
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xA1])
        t.insert([1, 2, 4], [KVBlock(block_id=1)], [0xB1])

        # Insert triggers split of [1,2,3] into [1,2] + [3]
        # Actually, the second insert also goes through match+insert,
        # but let's trigger explicit split via match
        node, _ = t.match([1, 2, 9])  # This splits [1,2,3]
        assert t.total_nodes == 3  # root, [1,2], [3], and [1,2,4]

        # Evict the suffix child
        freed = t.evict(10)  # Evict all evictable
        # After eviction, merges may compact single-child nodes
        assert isinstance(freed, list)

    def test_try_merge_single_child(self):
        """_try_merge should merge a node with single child and ref_count=0."""
        t = RadixTree()
        # Manually create a structure: root -> [1,2] -> [3]
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])

        # Split to create: root -> [1,2] -> [3]
        node, _ = t.match([1, 2, 9])
        assert node.token_ids == [1, 2]

        # Now the suffix [3] is the only child of node [1,2]
        # If we try to merge node [1,2] (ref_count=0, single child)
        t._try_merge(node)
        # Should have merged: node should now be [1,2,3] with no children
        assert node.token_ids == [1, 2, 3]
        assert len(node.children) == 0

    def test_try_merge_skips_root(self):
        """_try_merge should not merge root nodes."""
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        # Should not crash or modify root
        t._try_merge(t.root)
        assert t.root.token_ids == []

    def test_try_merge_skips_ref_counted(self):
        """_try_merge should not merge nodes with active references."""
        t = RadixTree()
        t.insert([1, 2, 3], [KVBlock(block_id=0)], [0xAAA])
        node, _ = t.match([1, 2, 9])
        node.ref_count = 1
        t._try_merge(node)
        # Should not merge
        assert node.token_ids == [1, 2]
