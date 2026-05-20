from __future__ import annotations
"""Yunshu RadixAttention — Tree-based prefix sharing for KV cache.

Improves on oMLX's flat block-level prefix matching with a radix tree
that supports:
- O(k) prefix lookup where k = matched depth
- Automatic subtree pruning for memory management
- Per-node reference counting for safe sharing
- Efficient multi-turn conversation cache reuse
"""


import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .block import KVBlock

logger = logging.getLogger(__name__)


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
        # Match statistics for Prometheus monitoring (SGLang pattern)
        self._match_total: int = 0
        self._match_hits: int = 0
        # Thread safety: all tree mutations are protected by a single lock.
        # Using threading.Lock (not RLock) for minimal overhead on hot paths.
        self._lock = threading.Lock()

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
        with self._lock:
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
                    node = self._split_node_unlocked(node, child, match_len)
                    pos += match_len
                    break
                else:
                    break

            remaining = token_ids[pos:]
            # Track match statistics (SGLang pattern for Prometheus export).
            # A "hit" means at least one token was matched from the tree,
            # avoiding a full prefill for the matched portion.
            self._match_total += 1
            if pos > 0:
                self._match_hits += 1
            return node, remaining

    def _split_node(
        self,
        parent: RadixNode,
        child: RadixNode,
        split_pos: int,
    ) -> RadixNode:
        """Split a child node at position split_pos (thread-safe)."""
        with self._lock:
            return self._split_node_unlocked(parent, child, split_pos)

    def _split_node_unlocked(
        self,
        parent: RadixNode,
        child: RadixNode,
        split_pos: int,
    ) -> RadixNode:
        """Internal: split a child node (caller holds lock).

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
        if split_pos <= 0:
            return child

        # Convert token-based split_pos to block-based index.
        # Use ceiling division: when split_pos is not block-aligned, the
        # boundary block (which straddles the split point) goes to the
        # new intermediate node (parent), NOT the child.  This ensures
        # each node owns blocks that fully cover its token_ids — the
        # intermediate node's token_ids[:split_pos] may extend into the
        # boundary block, and that block must be owned by the intermediate
        # node so path_blocks() returns complete coverage.
        # The child gets blocks starting from split_block_idx, which are
        # all fully within the child's token range.
        split_block_idx = (split_pos + self._block_size - 1) // self._block_size

        # Create the intermediate node with the shared prefix
        new_node = RadixNode(
            token_ids=child.token_ids[:split_pos],
            blocks=child.blocks[:split_block_idx],
            block_hashes=child.block_hashes[:split_block_idx],
            parent=parent,
        )

        # Shorten the original child to the suffix
        child.token_ids = child.token_ids[split_pos:]
        child.blocks = child.blocks[split_block_idx:]
        child.block_hashes = child.block_hashes[split_block_idx:]

        # Rewire parent: replace child with new_node
        first_tok = new_node.token_ids[0] if new_node.token_ids else None
        if first_tok is not None:
            parent.children[first_tok] = new_node

        # Move child to be a child of new_node
        child_first_tok = child.token_ids[0] if child.token_ids else None
        if child_first_tok is not None:
            new_node.children[child_first_tok] = child
        child.parent = new_node

        # Boundary block safety: when split_pos is not block-aligned, the
        # child's logical token range starts mid-way through the boundary
        # block.  The boundary block was assigned to new_node via ceiling
        # division, but child still needs it for correct path_blocks()
        # coverage.  Insert a reference to the boundary block at the head
        # of the child's block list so that evicting new_node alone does
        # not free a block the child depends on.
        is_aligned = (split_pos % self._block_size == 0)
        if not is_aligned and new_node.blocks:
            # The last block in new_node is the boundary block that straddles
            # the split point.  Give the child a reference to it.
            boundary_block = new_node.blocks[-1]
            child.blocks.insert(0, boundary_block)

        # Ref-count bookkeeping after split.
        # Before the split, child had ref_count R meaning R requests'
        # paths passed through it. After the split:
        #   - new_node is now on all those same paths (R requests)
        #   - child is still on those same R paths (the suffix portion)
        # So both new_node and child should retain ref_count R.
        # Setting child.ref_count = sum(children) was WRONG because it
        # dropped the direct references from requests that matched the
        # original (now-split) child.
        new_node.ref_count = child.ref_count
        new_node.last_access_time = child.last_access_time
        new_node.creation_time = child.creation_time
        new_node.access_count = child.access_count

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
        with self._lock:
            return self._insert_unlocked(token_ids, blocks, block_hashes, start_node)

    def _insert_unlocked(
        self,
        token_ids: list[int],
        blocks: list[KVBlock],
        block_hashes: list[int],
        start_node: RadixNode | None = None,
    ) -> RadixNode:
        """Internal: insert a new prefix path (caller holds lock)."""
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
                # No shared prefix — cannot happen because first_token
                # was used to look up the child, so at minimum the first
                # token matches.  Defensive: return existing to avoid
                # overwriting the child subtree at line node.children[first_token].
                return existing
            elif match_len < len(existing.token_ids):
                # Partial overlap: split existing child at match point
                split_node = self._split_node_unlocked(node, existing, match_len)
                # split_node now holds the shared prefix
                # Recurse: insert remaining new tokens as child of split_node
                remaining_new = token_ids[match_len:]
                # Use ceiling division matching _split_node_unlocked so that
                # the boundary block is NOT double-counted between the
                # intermediate node and the remaining new tokens.
                # _split_node_unlocked uses ceiling: split_block_idx = ceil(match_len / block_size)
                # We must use the same ceiling here so remaining_blocks starts
                # AFTER the boundary block that was assigned to split_node.
                match_block_idx = (match_len + self._block_size - 1) // self._block_size
                remaining_blocks = blocks[match_block_idx:]
                remaining_hashes = block_hashes[match_block_idx:]
                return self._insert_unlocked(remaining_new, remaining_blocks, remaining_hashes, start_node=split_node)
            else:
                # match_len == len(existing.token_ids): the existing child is
                # fully matched.  Two sub-cases:
                if len(token_ids) == len(existing.token_ids):
                    # Exact match — update existing node
                    if blocks:
                        # KNOWN LEAK: old blocks in existing.blocks are
                        # silently discarded without returning to
                        # BlockPool.free().  The caller should ideally free
                        # the old blocks, but the current API does not
                        # support returning them.  Log a debug warning so
                        # the leak is trackable in production logs.
                        old_blocks = existing.blocks
                        if old_blocks:
                            old_ids = [
                                getattr(b, "block_id", id(b))
                                for b in old_blocks
                            ]
                            logger.debug(
                                "RadixTree exact-match block replacement: "
                                "%d old blocks discarded without free (IDs: %s). "
                                "This is a known leak point — caller should free.",
                                len(old_blocks),
                                old_ids[:8],
                            )
                        existing.blocks = list(blocks)
                    if block_hashes:
                        existing.block_hashes = list(block_hashes)
                    return existing
                else:
                    # len(token_ids) > len(existing.token_ids): the new tokens
                    # extend BEYOND the existing child.  Recurse with the
                    # remaining tokens inserted as a child of existing.
                    remaining_new = token_ids[match_len:]
                    match_block_idx = len(existing.blocks)
                    remaining_blocks = blocks[match_block_idx:]
                    remaining_hashes = block_hashes[match_block_idx:]
                    return self._insert_unlocked(
                        remaining_new, remaining_blocks, remaining_hashes,
                        start_node=existing,
                    )

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
        with self._lock:
            now = _now()
            current = node
            while current is not None:
                current.ref_count += 1
                current.last_access_time = now
                current.access_count += 1
                current = current.parent
            self._total_ref_count += 1

    def dec_ref(self, node: RadixNode) -> None:
        """Decrement reference count from node to root.

        Guards against underflow: if a node's ref_count is already 0,
        it is not decremented further. This prevents corruption from
        double-decrement bugs.
        """
        with self._lock:
            current = node
            while current is not None:
                if current.ref_count > 0:
                    current.ref_count -= 1
                current = current.parent
            if self._total_ref_count > 0:
                self._total_ref_count -= 1

    def evict(self, n_nodes: int) -> list[KVBlock]:
        """Evict leaf nodes with ref_count == 0 using configured strategy.

        Strategies:
        - lru (default): least recently used — evict oldest last_access_time
        - lfu: least frequently used — evict lowest access_count
        - fifo: first in first out — evict oldest creation_time

        After eviction, merges parent nodes that end up with a single child
        and ref_count == 0 (post-split compaction).

        Optimisation: uses heapq for O(n log n) selection instead of full sort
        and filters ref_count > 0 before sorting to avoid wasted comparisons.

        Returns:
            List of freed KV blocks.
        """
        with self._lock:
            import heapq

            freed_blocks: list[KVBlock] = []
            evicted = 0

            # Collect only evictable leaves (ref_count == 0, leaf, not root)
            leaves = self._collect_evictable_leaves()

            # Early exit when nothing to evict
            if not leaves:
                return freed_blocks

            # Build a min-heap keyed by the eviction strategy metric so we
            # only pop as many entries as we actually need, avoiding a full sort
            # of potentially thousands of leaves.
            if self._eviction_strategy == "lfu":
                heap = [(n.access_count, id(n), n) for n in leaves]
            elif self._eviction_strategy == "fifo":
                heap = [(n.creation_time, id(n), n) for n in leaves]
            else:  # lru (default)
                heap = [(n.last_access_time, id(n), n) for n in leaves]

            heapq.heapify(heap)

            while heap and evicted < n_nodes:
                _key, _tid, leaf = heapq.heappop(heap)
                # Re-check ref_count — it may have changed since collection
                # (e.g., another inc_ref happened between collection and eviction).
                if leaf.ref_count > 0:
                    continue

                # Double-check it's still a leaf (merge may have changed children).
                if not leaf.is_leaf or leaf.is_root:
                    continue

                freed_blocks.extend(leaf.blocks)
                # Remove from parent's children
                if leaf.parent is not None:
                    parent = leaf.parent
                    first_tok = leaf.token_ids[0] if leaf.token_ids else None
                    if first_tok is not None:
                        parent.children.pop(first_tok, None)
                    else:
                        # Leaf has empty token_ids — find and remove by identity
                        for key, child in list(parent.children.items()):
                            if child is leaf:
                                parent.children.pop(key)
                                break

                    # Compact: merge parent with single remaining child
                    self._try_merge_unlocked(parent)

                # Clear stale references from the evicted leaf so future
                # path_blocks() / match() calls cannot return freed blocks.
                leaf.blocks = []
                leaf.block_hashes = []
                leaf.token_ids = []
                self._total_nodes -= 1
                evicted += 1

            if evicted > 0:
                self._eviction_stats[self._eviction_strategy] += evicted
                self._eviction_stats["total_freed_blocks"] += len(freed_blocks)

            return freed_blocks

    def _try_merge(self, node: RadixNode) -> None:
        """Merge a node with its single child if the node has ref_count == 0 (thread-safe)."""
        with self._lock:
            self._try_merge_unlocked(node)

    def _try_merge_unlocked(self, node: RadixNode) -> None:
        """Internal: merge a node with its single child (caller holds lock).

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

        # Save child's ref_count before clearing (needed for node inherit).
        child_ref_count = child.ref_count

        # Append child's tokens, blocks, and hashes to the node
        node.token_ids.extend(child.token_ids)
        node.blocks.extend(child.blocks)
        node.block_hashes.extend(child.block_hashes)

        # Adopt child's children before clearing child.
        node.children = child.children
        for grandchild in node.children.values():
            grandchild.parent = node

        # Clear child's data to prevent double-free if the child node
        # object is later popped from an eviction heap (the heap may
        # still hold a stale reference to this child).
        # Set child.parent = None to prevent stale dec_ref traversals
        # from reaching the merged parent node via the orphaned child.
        child.token_ids = []
        child.blocks = []
        child.block_hashes = []
        child.parent = None
        child.children = {}
        child.ref_count = 0

        # Inherit child's ref_count and access metadata.  This is safe
        # even when child.ref_count > 0: any request that previously
        # traversed ...-> node -> child -> ... now traverses ...-> node -> ...
        # because node absorbed child's children.  The ref_count is
        # preserved on node so eviction accounting stays correct.
        # Since node.ref_count == 0 (merge precondition), addition is
        # equivalent to overwrite but clearer about intent.
        node.ref_count = node.ref_count + child_ref_count
        node.last_access_time = max(
            node.last_access_time, child.last_access_time
        )
        node.access_count += child.access_count
        self._total_nodes -= 1

        # Recursively merge parent if it now also has a single child + ref_count == 0
        if node.parent is not None:
            self._try_merge_unlocked(node.parent)

    def _collect_evictable_leaves(self) -> list[RadixNode]:
        """Collect leaf nodes eligible for eviction (no children, not root, ref_count == 0)."""
        leaves = []

        def _walk(node: RadixNode) -> None:
            if node.is_leaf and not node.is_root and node.ref_count == 0:
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
        match_rate = (self._match_hits / self._match_total) if self._match_total > 0 else 0.0
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
            "match_total": self._match_total,
            "match_hits": self._match_hits,
            "match_rate": match_rate,
        }



def _now() -> float:
    return time.monotonic()
