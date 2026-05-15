"""Yunshu RadixAttention — Tree-based prefix sharing for KV cache.

Improves on oMLX's flat block-level prefix matching with a radix tree
that supports:
- O(k) prefix lookup where k = matched depth
- Automatic subtree pruning for memory management
- Per-node reference counting for safe sharing
- Efficient multi-turn conversation cache reuse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .block import KVBlock


@dataclass
class RadixNode:
    """A node in the radix tree.

    Each node represents a sequence of token IDs and holds references
    to the KV blocks that cache those tokens.
    """

    # Token IDs at this node (may be empty for root)
    token_ids: list[int] = field(default_factory=list)
    # KV blocks holding cached attention data for these tokens
    blocks: list[KVBlock] = field(default_factory=list)
    # Block hashes for this node's blocks
    block_hashes: list[int] = field(default_factory=list)
    # Children indexed by first token of child's prefix
    children: dict[int, RadixNode] = field(default_factory=dict)
    # Parent reference
    parent: Optional[RadixNode] = None
    # Reference count: how many active requests use this node
    ref_count: int = 0
    # Last access time for LRU eviction
    last_access_time: float = 0.0
    # Access count for LFU eviction
    access_count: int = 0
    # Creation time for FIFO eviction
    creation_time: float = 0.0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None and self.num_tokens == 0

    def total_tokens(self) -> int:
        """Count total tokens from root to this node."""
        count = self.num_tokens
        node = self.parent
        while node is not None:
            count += node.num_tokens
            node = node.parent
        return count

    def path_blocks(self) -> list[KVBlock]:
        """Collect all blocks from root to this node."""
        result = []
        node: RadixNode | None = self
        while node is not None:
            result.extend(node.blocks)
            node = node.parent
        result.reverse()
        return result


class RadixTree:
    """Radix tree for KV cache prefix sharing.

    Based on SGLang's RadixAttention concept but adapted for Yunshu's
    UMA-resident block pool.

    Operations:
    - match(prefix_tokens) → (matched_node, remaining_tokens)
    - insert(prefix_tokens, blocks) → new_node
    - evict(n_bytes) → freed blocks
    """

    def __init__(self, eviction_strategy: str = "lru", block_size: int = 64) -> None:
        self.root = RadixNode()
        self._total_nodes = 0
        self._total_ref_count = 0
        self._eviction_strategy = eviction_strategy  # lru, lfu, fifo
        self._block_size = block_size
        self._eviction_stats = {"lru": 0, "lfu": 0, "fifo": 0, "total_freed_blocks": 0}

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    def match(self, token_ids: list[int]) -> tuple[RadixNode, list[int]]:
        """Find the longest matching prefix in the tree.

        Supports node splitting: when a child node partially matches
        (e.g., child=[A,B,C] but query=[A,B,D]), the child is split
        into [A,B] parent + [C] child, enabling proper prefix sharing.

        Args:
            token_ids: The full token sequence to match against.

        Returns:
            (matched_node, remaining_tokens) where remaining_tokens are
            the unmatched suffix that needs fresh prefill.
        """
        node = self.root
        pos = 0

        while pos < len(token_ids):
            first_token = token_ids[pos]
            child = node.children.get(first_token)
            if child is None:
                break

            # Match child's token_ids against input
            child_tokens = child.token_ids
            match_len = 0
            for i in range(min(len(child_tokens), len(token_ids) - pos)):
                if child_tokens[i] != token_ids[pos + i]:
                    break
                match_len += 1

            if match_len == len(child_tokens):
                # Full match of this child node
                node = child
                pos += match_len
            elif match_len > 0:
                # Partial match — split the child node
                node = self._split_node(node, child, match_len)
                pos += match_len
                break
            else:
                break

        remaining = token_ids[pos:]
        return node, remaining

    def _split_node(
        self,
        parent: RadixNode,
        child: RadixNode,
        split_pos: int,
    ) -> RadixNode:
        """Split a child node at position split_pos.

        Before: parent → child=[A,B,C]
        After:  parent → new_node=[A,B] → child=[C]

        The new_node inherits the first `split_pos` tokens and blocks
        from the original child. The original child is shortened to
        the remaining tokens. All of the original child's other
        children are moved to the new intermediate node.

        Args:
            parent: The parent node of the child being split.
            child: The child node to split.
            split_pos: Position at which to split (number of matched tokens).

        Returns:
            The new intermediate node containing the shared prefix.
        """
        # Convert token-based split_pos to block-based index.
        # Each block covers self._block_size tokens, so we must slice the
        # blocks and block_hashes lists at the block boundary, not the
        # token boundary.
        split_block_idx = split_pos // self._block_size

        # Create the intermediate node with the shared prefix
        new_node = RadixNode(
            token_ids=child.token_ids[:split_pos],
            blocks=child.blocks[:split_block_idx] if len(child.blocks) > split_block_idx else list(child.blocks),
            block_hashes=child.block_hashes[:split_block_idx] if len(child.block_hashes) > split_block_idx else list(child.block_hashes),
            parent=parent,
        )

        # Shorten the original child to the suffix
        child.token_ids = child.token_ids[split_pos:]
        if len(child.blocks) > split_block_idx:
            child.blocks = child.blocks[split_block_idx:]
        else:
            child.blocks = []
        if len(child.block_hashes) > split_block_idx:
            child.block_hashes = child.block_hashes[split_block_idx:]
        else:
            child.block_hashes = []

        # Rewire parent: replace child with new_node
        first_tok = new_node.token_ids[0] if new_node.token_ids else None
        if first_tok is not None:
            parent.children[first_tok] = new_node

        # Move child to be a child of new_node
        child_first_tok = child.token_ids[0] if child.token_ids else None
        if child_first_tok is not None:
            new_node.children[child_first_tok] = child
        child.parent = new_node

        # Inherit ref_count: the new_node gets the child's ref_count
        # so that eviction doesn't remove it while active requests use it
        new_node.ref_count = child.ref_count
        new_node.last_access_time = child.last_access_time

        self._total_nodes += 1
        return new_node

    def insert(
        self,
        token_ids: list[int],
        blocks: list[KVBlock],
        block_hashes: list[int],
        start_node: RadixNode | None = None,
    ) -> RadixNode:
        """Insert a new prefix path into the tree.

        If an existing child shares a prefix with the new tokens, the child
        is split at the divergence point to enable proper prefix sharing
        (SGLang RadixCache pattern).

        Args:
            token_ids: Tokens for this path.
            blocks: KV blocks corresponding to these tokens.
            block_hashes: Hashes for the blocks.
            start_node: Node to start insertion from (for extending after match).

        Returns:
            The leaf node created.
        """
        node = start_node or self.root

        if not token_ids:
            return node

        # Check if a child with the same first token already exists
        first_token = token_ids[0]
        existing = node.children.get(first_token)

        if existing is not None:
            # Find the shared prefix length between existing child and new tokens
            match_len = 0
            for i in range(min(len(existing.token_ids), len(token_ids))):
                if existing.token_ids[i] != token_ids[i]:
                    break
                match_len += 1

            if match_len == 0:
                # No shared prefix (shouldn't happen since first_token matched)
                pass
            elif match_len < len(existing.token_ids):
                # Partial overlap: split existing child at match point
                split_node = self._split_node(node, existing, match_len)
                # split_node now holds the shared prefix
                # Recurse: insert remaining new tokens as child of split_node
                remaining_new = token_ids[match_len:]
                # Convert token offset to block offset for slicing
                match_block_idx = match_len // self._block_size
                remaining_blocks = blocks[match_block_idx:] if len(blocks) > match_block_idx else []
                remaining_hashes = block_hashes[match_block_idx:] if len(block_hashes) > match_block_idx else []
                return self.insert(remaining_new, remaining_blocks, remaining_hashes, start_node=split_node)
            else:
                # New tokens are a prefix of or equal to existing child
                if len(token_ids) == len(existing.token_ids):
                    # Exact match — update existing node
                    if blocks:
                        existing.blocks = list(blocks)
                    if block_hashes:
                        existing.block_hashes = list(block_hashes)
                    return existing
                else:
                    # New is shorter — split existing at len(token_ids)
                    split_node = self._split_node(node, existing, len(token_ids))
                    return split_node

        # No existing child: create a new leaf node
        now = _now()
        new_node = RadixNode(
            token_ids=list(token_ids),
            blocks=list(blocks),
            block_hashes=list(block_hashes),
            parent=node,
            creation_time=now,
            last_access_time=now,
        )
        node.children[first_token] = new_node
        self._total_nodes += 1
        return new_node

    def inc_ref(self, node: RadixNode) -> None:
        """Increment reference count from node to root."""
        now = _now()
        current = node
        while current is not None:
            current.ref_count += 1
            current.last_access_time = now
            current.access_count += 1
            current = current.parent
        self._total_ref_count += 1

    def dec_ref(self, node: RadixNode) -> None:
        """Decrement reference count from node to root."""
        current = node
        while current is not None:
            current.ref_count -= 1
            current = current.parent
        self._total_ref_count -= 1

    def evict(self, n_nodes: int) -> list[KVBlock]:
        """Evict leaf nodes with ref_count == 0 using configured strategy.

        Strategies:
        - lru (default): least recently used — evict oldest last_access_time
        - lfu: least frequently used — evict lowest access_count
        - fifo: first in first out — evict oldest creation_time

        After eviction, merges parent nodes that end up with a single child
        and ref_count == 0 (post-split compaction).

        Returns:
            List of freed KV blocks.
        """
        freed_blocks: list[KVBlock] = []
        evicted = 0

        # Collect all evictable leaves (ref_count == 0, not root)
        leaves = self._collect_evictable_leaves()
        # Sort by chosen strategy
        if self._eviction_strategy == "lfu":
            leaves.sort(key=lambda n: n.access_count)
        elif self._eviction_strategy == "fifo":
            leaves.sort(key=lambda n: n.creation_time)
        else:  # lru (default)
            leaves.sort(key=lambda n: n.last_access_time)

        for leaf in leaves:
            if evicted >= n_nodes:
                break
            if leaf.ref_count > 0:
                continue

            freed_blocks.extend(leaf.blocks)
            # Remove from parent's children
            if leaf.parent is not None:
                parent = leaf.parent
                first_tok = leaf.token_ids[0] if leaf.token_ids else None
                if first_tok is not None:
                    parent.children.pop(first_tok, None)

                # Compact: merge parent with single remaining child
                self._try_merge(parent)
            self._total_nodes -= 1
            evicted += 1

        if evicted > 0:
            self._eviction_stats[self._eviction_strategy] += evicted
            self._eviction_stats["total_freed_blocks"] += len(freed_blocks)

        return freed_blocks

    def _try_merge(self, node: RadixNode) -> None:
        """Merge a node with its single child if the node has ref_count == 0.

        After a child is evicted, the parent may end up with exactly one
        remaining child and no active references. In that case, we merge
        the parent and child into a single node to avoid unnecessary
        tree depth (post-split compaction).
        """
        if node.is_root or node.ref_count > 0:
            return
        if len(node.children) != 1:
            return

        # Merge with the single child
        child = next(iter(node.children.values()))
        if child.ref_count > 0:
            return

        # Append child's tokens, blocks, and hashes to the node
        node.token_ids.extend(child.token_ids)
        node.blocks.extend(child.blocks)
        node.block_hashes.extend(child.block_hashes)

        # Adopt child's children
        node.children = child.children
        for grandchild in node.children.values():
            grandchild.parent = node

        # Update ref_count to child's value
        node.ref_count = child.ref_count
        self._total_nodes -= 1

    def _collect_evictable_leaves(self) -> list[RadixNode]:
        """Collect all leaf nodes (no children, not root)."""
        leaves = []

        def _walk(node: RadixNode) -> None:
            if node.is_leaf and not node.is_root:
                leaves.append(node)
            for child in node.children.values():
                _walk(child)

        _walk(self.root)
        return leaves

    def get_stats(self) -> dict:
        """Return tree statistics with detailed metrics."""
        total_blocks = 0
        total_tokens = 0
        active_refs = 0
        max_depth = 0
        leaf_count = 0

        def _walk(node: RadixNode, depth: int = 0) -> None:
            nonlocal total_blocks, total_tokens, active_refs, max_depth, leaf_count
            total_blocks += node.num_blocks
            total_tokens += node.num_tokens
            if node.ref_count > 0:
                active_refs += 1
            if not node.children:
                leaf_count += 1
            max_depth = max(max_depth, depth)
            for child in node.children.values():
                _walk(child, depth + 1)

        _walk(self.root, 0)
        return {
            "total_nodes": self._total_nodes,
            "total_blocks": total_blocks,
            "total_tokens": total_tokens,
            "total_ref_count": self._total_ref_count,
            "active_ref_nodes": active_refs,
            "leaf_count": leaf_count,
            "max_depth": max_depth,
            "eviction_strategy": self._eviction_strategy,
            "eviction_stats": dict(self._eviction_stats),
        }

    def get_bigram_view(self, node: RadixNode, n: int = 5) -> list[int]:
        """Return the last N tokens leading to and including this node.

        Used by EAGLE/spec decode to get context from the radix tree for
        draft token generation. Walks from the node toward the root,
        collecting the most recent `n` tokens.

        Args:
            node: The matched node in the tree.
            n: Maximum number of trailing tokens to return.

        Returns:
            List of up to `n` token IDs (in original order, root→node).
        """
        tokens: list[int] = []
        current: RadixNode | None = node
        while current is not None and len(tokens) < n:
            tokens.extend(reversed(current.token_ids))
            current = current.parent
        tokens.reverse()
        # Trim to last n
        return tokens[-n:] if len(tokens) > n else tokens

    def get_continuation_tokens(self, token_ids: list[int], max_results: int = 5) -> list[int]:
        """Return continuation tokens from the tree after a prefix match.

        Given a prefix, finds the matched node and returns the first token
        of each child as possible continuations. This is the "bigram view"
        for speculative decoding: after matching a prefix, the tree tells
        us what tokens have historically followed.

        Args:
            token_ids: Prefix tokens to match.
            max_results: Maximum continuation tokens to return.

        Returns:
            List of possible continuation token IDs (children's first tokens).
        """
        node, remaining = self.match(token_ids)
        if remaining:
            # Not a full match — no continuation info
            return []

        continuations = []
        for first_tok, child in node.children.items():
            if child.ref_count > 0 or child.access_count > 0:
                continuations.append(first_tok)
            if len(continuations) >= max_results:
                break
        return continuations


def _now() -> float:
    import time

    return time.monotonic()
