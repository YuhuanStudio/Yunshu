"""Yunshu KV Cache Block Pool — UMA-resident PagedAttention.

Inspired by vLLM's BlockPool but adapted for Apple Silicon UMA:
- Blocks live in unified memory (no GPU/CPU copy needed)
- Block metadata is pure Python (no torch tensors)
- Hash uses xxhash instead of SHA256 for speed
- Reference counting with O(1) LRU eviction via doubly-linked list
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVBlock:
    """A single KV cache block.

    Attributes:
        block_id: Unique physical block identifier.
        ref_count: Number of requests sharing this block.
        block_hash: Hash of content for prefix caching (None if not yet full/cached).
        prev: Previous block in LRU free list.
        next: Next block in LRU free list.
    """

    block_id: int
    ref_count: int = 0
    block_hash: Optional[int] = None
    is_null: bool = False
    prev: Optional[KVBlock] = None
    next: Optional[KVBlock] = None

    def reset_hash(self) -> None:
        self.block_hash = None


class FreeBlockQueue:
    """Doubly-linked list of free blocks for O(1) LRU eviction.

    Eviction order: head (oldest/cold) → tail (newest/hot).
    """

    def __init__(self, blocks: list[KVBlock]) -> None:
        self.num_free_blocks: int = len(blocks)
        # Sentinel nodes simplify insertion/removal
        self._head = KVBlock(block_id=-1)  # oldest free
        self._tail = KVBlock(block_id=-1)  # newest free
        self._head.next = self._tail
        self._tail.prev = self._head

        for block in blocks:
            self._push_back(block)

    def _push_back(self, block: KVBlock) -> None:
        """Insert block at tail (most recently used)."""
        prev = self._tail.prev
        assert prev is not None
        block.prev = prev
        block.next = self._tail
        prev.next = block
        self._tail.prev = block

    def popleft(self) -> KVBlock:
        """Remove and return the oldest free block (head)."""
        assert self.num_free_blocks > 0
        block = self._head.next
        assert block is not None and block is not self._tail
        self._remove(block)
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n: int) -> list[KVBlock]:
        """Remove and return the n oldest free blocks."""
        assert n <= self.num_free_blocks
        blocks = []
        for _ in range(n):
            blocks.append(self.popleft())
        return blocks

    def append(self, block: KVBlock) -> None:
        """Add a freed block at tail (most recently freed)."""
        self._push_back(block)
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVBlock]) -> None:
        for block in blocks:
            self.append(block)

    def remove(self, block: KVBlock) -> None:
        """Remove a specific block from the free list."""
        self._remove(block)
        self.num_free_blocks -= 1

    def _remove(self, block: KVBlock) -> None:
        prev, nxt = block.prev, block.next
        assert prev is not None and nxt is not None
        prev.next = nxt
        nxt.prev = prev
        block.prev = None
        block.next = None


class BlockPool:
    """Manages KV cache blocks in UMA with prefix caching.

    Architecture:
    - Pre-allocates a fixed number of blocks at init (num_blocks).
    - Each block holds `block_size` tokens of KV data.
    - Free blocks are tracked in a doubly-linked list for O(1) LRU eviction.
    - Cached (full) blocks are indexed by hash for prefix deduplication.
    - Reference counting enables safe sharing across requests.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_caching: bool = True,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_caching = enable_caching

        # All blocks
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue(self.blocks)

        # Null block (block_id=0 reserved, never freed)
        self.null_block = self.free_queue.popleft()
        self.null_block.is_null = True

        # Hash → block mapping for prefix caching
        self._hash_to_block: dict[int, KVBlock] = {}

    def get_free_block_count(self) -> int:
        return self.free_queue.num_free_blocks

    def get_usage(self) -> float:
        total = self.num_blocks - 1  # exclude null block
        if total == 0:
            return 0.0
        return 1.0 - (self.get_free_block_count() / total)

    def allocate(self, n: int) -> list[KVBlock]:
        """Allocate n fresh blocks from the free pool."""
        if n > self.get_free_block_count():
            raise ValueError(
                f"Cannot allocate {n} blocks, only {self.get_free_block_count()} free"
            )
        blocks = self.free_queue.popleft_n(n)
        for block in blocks:
            if self.enable_caching and block.block_hash is not None:
                self._evict_cached_block(block)
            block.ref_count = 1
        return blocks

    def touch(self, block: KVBlock) -> None:
        """Increase ref count of a shared block (prefix cache hit)."""
        if block.ref_count == 0 and not block.is_null:
            self.free_queue.remove(block)
        block.ref_count += 1

    def free(self, blocks: list[KVBlock]) -> None:
        """Decrease ref count; blocks reaching 0 go back to free list."""
        freed = []
        for block in blocks:
            block.ref_count -= 1
            if block.ref_count == 0 and not block.is_null:
                freed.append(block)
        self.free_queue.append_n(freed)

    def cache_block(self, block: KVBlock, block_hash: int) -> None:
        """Register a full block in the prefix cache."""
        if not self.enable_caching:
            return
        block.block_hash = block_hash
        self._hash_to_block[block_hash] = block

    def lookup_hash(self, block_hash: int) -> Optional[KVBlock]:
        """Find a cached block by hash."""
        block = self._hash_to_block.get(block_hash)
        if block is not None:
            return block
        return None

    def _evict_cached_block(self, block: KVBlock) -> bool:
        """Remove a block from the prefix cache."""
        if block.block_hash is None:
            return False
        cached = self._hash_to_block.pop(block.block_hash, None)
        if cached is not None:
            cached.reset_hash()
            return True
        return False

    def get_cached_blocks(self) -> list[KVBlock]:
        """Return all blocks currently in the prefix cache."""
        return list(self._hash_to_block.values())

    def reset_prefix_cache(self) -> None:
        """Clear all prefix cache entries."""
        self._hash_to_block.clear()
        for block in self.blocks:
            block.reset_hash()
