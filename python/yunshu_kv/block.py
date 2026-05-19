from __future__ import annotations

"""Yunshu KV Cache Block Pool — UMA-resident PagedAttention.

Inspired by vLLM's BlockPool but adapted for Apple Silicon UMA:
- Blocks live in unified memory (no GPU/CPU copy needed)
- Block metadata is pure Python (no torch tensors)
- Hash uses xxhash instead of SHA256 for speed
- Reference counting with O(1) LRU eviction via doubly-linked list
"""


import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .cache_events import CacheEventBus

logger = logging.getLogger(__name__)


@dataclass
class KVBlock:
    """A single KV cache block.

    Attributes:
        block_id: Unique physical block identifier.
        ref_count: Number of requests sharing this block.
        block_hash: Hash of content for prefix caching (None if not yet full/cached).
        cache_only: True when block is only held by the prefix cache (no active request).
        prev: Previous block in LRU free list.
        next: Next block in LRU free list.
        last_access_time: Monotonic timestamp of last prefix-cache access (for LRU eviction).
    """

    block_id: int
    ref_count: int = 0
    block_hash: Optional[int] = None
    cache_only: bool = False
    is_null: bool = False
    prev: Optional[KVBlock] = None
    next: Optional[KVBlock] = None
    last_access_time: float = 0.0

    def reset_hash(self) -> None:
        self.block_hash = None
        self.cache_only = False


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
        if prev is None:
            raise RuntimeError("FreeBlockQueue corrupted: tail.prev is None")
        block.prev = prev
        block.next = self._tail
        prev.next = block
        self._tail.prev = block

    def popleft(self) -> KVBlock:
        """Remove and return the oldest free block (head)."""
        if self.num_free_blocks <= 0:
            raise IndexError("FreeBlockQueue is empty")
        block = self._head.next
        if block is None or block is self._tail:
            raise IndexError("FreeBlockQueue corrupted: head.next is sentinel")
        self._remove(block)
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n: int) -> list[KVBlock]:
        """Remove and return the n oldest free blocks."""
        if n > self.num_free_blocks:
            raise IndexError(f"Cannot pop {n} blocks, only {self.num_free_blocks} free")
        blocks = []
        for _ in range(n):
            blocks.append(self.popleft())
        return blocks

    def append(self, block: KVBlock) -> None:
        """Add a freed block at tail (most recently freed)."""
        # Detach from old position if already in the list.
        # A block in the free list has both prev and next set (they are
        # never None for list members — only unlinked blocks have None).
        if block.prev is not None or block.next is not None:
            # Guard: sentinel nodes must never enter the free list or count.
            if block is self._head or block is self._tail:
                return
            self._remove(block)
            self.num_free_blocks -= 1
        self._push_back(block)
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVBlock]) -> None:
        for block in blocks:
            self.append(block)

    def remove(self, block: KVBlock) -> None:
        """Remove a specific block from the free list."""
        if block.prev is None and block.next is None:
            return  # Not in the free list
        # Don't remove sentinel nodes
        if block is self._head or block is self._tail:
            return
        self._remove(block)
        self.num_free_blocks -= 1

    def _remove(self, block: KVBlock) -> None:
        prev, nxt = block.prev, block.next
        if prev is None or nxt is None:
            raise RuntimeError("FreeBlockQueue corrupted: block has None prev/next")
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
    - Copy-on-write (COW): shared blocks are transparently cloned on write
      to prevent corruption between requests sharing a prefix.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_caching: bool = True,
        event_bus: CacheEventBus | None = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_caching = enable_caching
        self._event_bus = event_bus

        # All blocks
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue(self.blocks)

        # Null block (block_id=0 reserved, never freed)
        self.null_block = self.free_queue.popleft()
        self.null_block.is_null = True

        # Hash → block mapping for prefix caching
        self._hash_to_block: dict[int, KVBlock] = {}

        # COW statistics
        self._cow_clones: int = 0

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
            block.cache_only = False
        return blocks

    def touch(self, block: KVBlock) -> None:
        """Increase ref count of a shared block (prefix cache hit)."""
        import time
        if block.is_null:
            return
        if block.ref_count == 0:
            self.free_queue.remove(block)
        block.ref_count += 1
        block.cache_only = False
        block.last_access_time = time.monotonic()

    def free(self, blocks: list[KVBlock]) -> None:
        """Decrease ref count; blocks reaching 0 go back to free list.

        When a block with a block_hash reaches ref_count 0, it is marked
        cache_only — it's in the free queue but still registered in the
        prefix cache. Eviction can safely free these blocks.

        Duplicate blocks in the input list are silently skipped to prevent
        ref_count from being decremented multiple times for the same block
        (a caller bug that would prematurely free blocks still in use).
        """
        freed = []
        seen_ids: set[int] = set()
        for block in blocks:
            if block.block_id in seen_ids:
                continue
            seen_ids.add(block.block_id)
            if block.ref_count <= 0:
                continue
            block.ref_count -= 1
            if block.ref_count == 0 and not block.is_null:
                if block.block_hash is not None:
                    block.cache_only = True
                freed.append(block)
        self.free_queue.append_n(freed)

    def cache_block(self, block: KVBlock, block_hash: int) -> None:
        """Register a full block in the prefix cache."""
        if not self.enable_caching:
            return
        import time
        # If a different block already owns this hash slot, clear its
        # cache metadata so it doesn't linger in cache_only state.
        old_block = self._hash_to_block.get(block_hash)
        if old_block is not None and old_block is not block:
            old_block.reset_hash()
        block.block_hash = block_hash
        block.last_access_time = time.monotonic()
        self._hash_to_block[block_hash] = block
        if self._event_bus is not None:
            from .cache_events import CacheEvent
            self._event_bus.publish(CacheEvent(
                "block_cached",
                block_hash=block_hash,
                block_ids=[block.block_id],
            ))

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
        # Only pop from the hash map if this block is still the current
        # entry for its hash.  If a different block has taken over the
        # hash (via cache_block), evicting would incorrectly remove the
        # newer block's cache entry.
        current = self._hash_to_block.get(block.block_hash)
        if current is block:
            self._hash_to_block.pop(block.block_hash, None)
        # Snapshot hash before clearing — needed for the event.
        evicted_hash = block.block_hash
        # Always clear the requesting block's hash, even if the hash
        # map pointed to a different block.  Leaving a stale block_hash
        # on an allocated block causes _evict_cached_block to do
        # redundant work on the next allocation and can mislead callers
        # that check block_hash for cache membership.
        block.reset_hash()
        if self._event_bus is not None and current is block:
            from .cache_events import CacheEvent
            self._event_bus.publish(CacheEvent(
                "block_evicted",
                block_hash=evicted_hash,
                block_ids=[block.block_id],
            ))
        return current is block

    def get_cached_blocks(self) -> list[KVBlock]:
        """Return all blocks currently in the prefix cache."""
        return list(self._hash_to_block.values())

    def reset_prefix_cache(self) -> None:
        """Clear all prefix cache entries."""
        self._hash_to_block.clear()
        for block in self.blocks:
            block.reset_hash()

    # ── Copy-on-Write (COW) ─────────────────────────────────────────

    def cow_block(self, block: KVBlock) -> KVBlock:
        """Copy-on-write: clone a shared block for exclusive access.

        When a block's ref_count > 1 (shared by multiple requests via
        prefix cache), mutating its KV data would corrupt other sharers.
        COW allocates a fresh block, copies metadata, and decrements the
        original's ref_count — leaving the caller with an exclusive block.

        For blocks with ref_count == 1, this is a no-op (returns the same
        block since the caller already has exclusive access).

        Args:
            block: The block to potentially clone.

        Returns:
            An exclusive (ref_count == 1) block. May be the same block
            if it wasn't shared.

        Raises:
            ValueError: If no free blocks are available for cloning.
        """
        if block.ref_count <= 1:
            return block

        if self.free_queue.num_free_blocks == 0:
            raise ValueError(
                "COW failed: no free blocks available for cloning"
            )

        # Allocate a fresh block
        new_block = self.free_queue.popleft()
        # Clear any stale hash from a previous lifecycle to prevent
        # _hash_to_block from returning this block for outdated lookups.
        if new_block.block_hash is not None:
            self._evict_cached_block(new_block)
        new_block.ref_count = 1

        # Decrement original's ref_count (we're detaching from it)
        block.ref_count -= 1
        if block.ref_count == 0 and not block.is_null:
            # Block going to free queue — must clear its hash to prevent
            # stale _hash_to_block lookups from returning a freed block
            self._evict_cached_block(block)
            self.free_queue.append(block)

        self._cow_clones += 1
        return new_block

    def cow_block_in_table(
        self,
        table: Any,
        logical_idx: int,
        key_cache: Any = None,
        value_cache: Any = None,
    ) -> tuple[KVBlock, Any, Any]:
        """COW a block within a BlockTable, copying KV data if available.

        After COW, the table entry at `logical_idx` points to the new
        exclusive block. If KV cache tensors are provided, the KV data
        from the original block is copied to the new block's slot.

        Args:
            table: BlockTable containing the block to COW.
            logical_idx: Logical index within the table.
            key_cache: Key cache tensor array (shape [num_blocks, ...]).
            value_cache: Value cache tensor array (shape [num_blocks, ...]).

        Returns:
            The new exclusive block and (possibly updated) cache tensors.
        """
        old_block = table.get_block(logical_idx)
        new_block = self.cow_block(old_block)

        if new_block is not old_block:
            # Copy KV data from old block to new block's slot
            if key_cache is not None and value_cache is not None:
                try:
                    import mlx.core as mx
                    mx.eval(key_cache[new_block.block_id])
                    if isinstance(key_cache, mx.array):
                        key_cache = key_cache.at[new_block.block_id].set(key_cache[old_block.block_id])
                        value_cache = value_cache.at[new_block.block_id].set(value_cache[old_block.block_id])
                    else:
                        key_cache[new_block.block_id] = key_cache[old_block.block_id]
                        value_cache[new_block.block_id] = value_cache[old_block.block_id]
                except Exception:
                    logger.warning("KV data copy in cow_block_in_table failed — returning old block to avoid corruption", exc_info=True)
                    # Put the new block back — it was allocated from the free
                    # queue but we can't use it.  Without this, the block leaks
                    # permanently (ref_count=1 but nobody holds a reference).
                    new_block.ref_count = 0
                    self.free_queue.append(new_block)
                    return old_block, key_cache, value_cache
            # Update the table entry
            table._blocks[logical_idx] = new_block

        return new_block, key_cache, value_cache

    @property
    def cow_stats(self) -> dict:
        """Return COW statistics."""
        return {
            "cow_clones": self._cow_clones,
        }
