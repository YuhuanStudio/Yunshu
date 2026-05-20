from __future__ import annotations
"""Yunshu KV Cache Manager — UMA-resident paged KV cache with prefix sharing.

Orchestrates BlockPool (allocation/eviction) + BlockTable (per-request mapping)
+ MLX KV cache tensors (actual key/value data in unified memory).

Three-tier KV hierarchy (Phase 1 implements hot tier only):
  Hot (UMA FP16) → Warm (TurboQuant 4-bit) → Cold (SSD)
"""


import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from .block import BlockPool, KVBlock
from .block_table import BlockTable
from .cache_events import CacheEvent, CacheEventBus
from .hash import compute_block_hash, compute_prompt_hashes
from .warm_tier import KVWarmTier, KVTierConfig


@dataclass
class KVCacheConfig:
    """Configuration for the KV cache manager."""

    block_size: int = 64  # tokens per block
    # Memory budget: computed from model architecture
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    dtype_bytes: int = 2  # FP16
    # Derive total blocks from UMA budget
    max_kv_memory_bytes: int = 0
    enable_caching: bool = True


def compute_num_blocks(
    config: KVCacheConfig,
    total_uma_bytes: int,
    model_weight_bytes: int,
    activation_reserve_ratio: float = 0.15,
) -> int:
    """Compute how many KV blocks fit in available UMA.

    Args:
        config: KV cache config with model architecture info.
        total_uma_bytes: Total unified memory (e.g., 192 * 2**30 for M3 Ultra).
        model_weight_bytes: Model weights in memory.
        activation_reserve_ratio: Fraction of UMA reserved for activations.

    Returns:
        Number of KV blocks that fit.
    """
    available = total_uma_bytes - model_weight_bytes
    available = int(available * (1.0 - activation_reserve_ratio))

    bytes_per_block = (
        config.block_size
        * config.num_layers
        * config.num_kv_heads
        * config.head_dim
        * config.dtype_bytes
        * 2  # keys + values
    )
    if bytes_per_block == 0:
        return 0
    return max(0, available // bytes_per_block)


@dataclass
class PrefixMatch:
    """Result of prefix cache lookup."""

    matched_blocks: list[KVBlock]  # Cached blocks to reuse
    num_matched_tokens: int
    unmatched_token_ids: list[int]  # Tokens still needing prefill


class KVCacheManager:
    """Manages paged KV cache in Apple Silicon unified memory.

    This is the central coordinator for KV cache operations:
    - Allocating/freeing blocks for requests
    - Prefix caching (hash-based deduplication)
    - RadixTree-based prefix sharing (C8: SGLang pattern)
    - Computing memory budget from model architecture
    - Integration with MLX KV cache tensors (Phase 2)
    """

    def __init__(
        self,
        config: KVCacheConfig,
        num_blocks: int,
        event_bus: CacheEventBus | None = None,
    ) -> None:
        self.config = config
        self._event_bus = event_bus if event_bus is not None else CacheEventBus()
        self.block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=config.block_size,
            enable_caching=config.enable_caching,
            event_bus=self._event_bus,
        )
        # Hit rate tracking
        self._total_lookups: int = 0
        self._total_hits: int = 0
        # KV cache tensors (set externally via set_kv_tensors)
        self._key_cache = None
        self._value_cache = None
        # Warm tier: 4-bit quantized block cache (default instance)
        self._warm_tier: KVWarmTier | None = KVWarmTier(KVTierConfig())
        # RadixTree for prefix sharing (C8: SGLang RadixAttention pattern)
        from .radix_attention import RadixTree
        self._radix_tree = RadixTree(block_size=config.block_size)
        self._request_nodes: dict[str, Any] = {}  # request_id → RadixNode
        # Thread safety: protects all manager-level mutations involving
        # block_pool, radix_tree, and internal state.
        # Using threading.Lock (not RLock) for minimal overhead on hot paths.
        self._lock = threading.Lock()

    @property
    def hit_rate(self) -> float:
        """Prefix cache hit rate (0.0 to 1.0)."""
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def set_kv_tensors(self, key_cache, value_cache) -> None:
        """Set the MLX KV cache tensors for serialization.

        Args:
            key_cache: MLX array of shape [num_blocks, num_layers, ...].
            value_cache: MLX array of shape [num_blocks, num_layers, ...].
        """
        self._key_cache = key_cache
        self._value_cache = value_cache

    def set_warm_tier(self, warm_tier) -> None:
        """Set the warm tier manager for block demotion on eviction."""
        self._warm_tier = warm_tier

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.get_free_block_count()

    def allocate_for_prefill(
        self,
        token_ids: list[int],
        model_hash: int = 0,
        request_id: str | None = None,
    ) -> tuple[BlockTable, PrefixMatch]:
        """Allocate blocks for a new request, checking prefix cache first.

        Uses RadixTree for O(k) prefix matching (C8), falls back to
        hash-chain lookup for individual blocks.

        Args:
            token_ids: Token IDs for the request prompt.
            model_hash: Hash of the model for cache isolation.
            request_id: Optional request ID for radix tree ref counting.

        Returns:
            (BlockTable for the request, PrefixMatch describing cache hit)
        """
        with self._lock:
            return self._allocate_for_prefill_unlocked(token_ids, model_hash, request_id)

    def _allocate_for_prefill_unlocked(
        self,
        token_ids: list[int],
        model_hash: int = 0,
        request_id: str | None = None,
    ) -> tuple[BlockTable, PrefixMatch]:
        """Internal: allocate blocks for a new request (caller holds lock)."""
        # 0. Try RadixTree prefix match (C8: O(k) tree traversal)
        if self.config.enable_caching and len(token_ids) >= self.config.block_size:
            matched_node = self._radix_tree.match(token_ids)[0]
            matched_blocks = matched_node.path_blocks()
            num_matched_tokens = matched_node.total_tokens()
            # Validate matched blocks: evicted blocks may still be
            # referenced by radix tree nodes.  A block is usable only
            # if it is actively held (ref_count > 0) or is a valid
            # prefix cache entry (block_hash set, NOT cache_only —
            # cache_only blocks are in the free queue and can be
            # evicted/reallocated at any moment, so using them would
            # give the request a block whose KV data could be silently
            # overwritten).
            valid_blocks = [
                b for b in matched_blocks
                if b.ref_count > 0
                or (b.block_hash is not None and not b.cache_only)
            ]
            if len(valid_blocks) < len(matched_blocks):
                # Some blocks were evicted — fall through to hash-chain
                # lookup which correctly handles the block pool state.
                logger.debug(
                    "RadixTree match returned %d blocks but only %d are still valid; "
                    "falling back to hash-chain lookup",
                    len(matched_blocks), len(valid_blocks),
                )
            elif num_matched_tokens >= self.config.block_size:
                # RadixTree hit: reuse matched blocks
                self._total_hits += num_matched_tokens // self.config.block_size
                self._total_lookups += len(token_ids) // self.config.block_size
                for block in matched_blocks:
                    self.block_pool.touch(block)
                # Bug 2: inc_ref the matched node so eviction won't remove
                # nodes used by active requests.
                self._radix_tree.inc_ref(matched_node)
                if request_id is not None:
                    old_node = self._request_nodes.get(request_id)
                    if old_node is not None:
                        self._radix_tree.dec_ref(old_node)
                    self._request_nodes[request_id] = matched_node
                return self._build_table_from_match(
                    matched_blocks, num_matched_tokens, token_ids,
                )

        # 1. Compute block hashes for the prompt
        block_hashes = compute_prompt_hashes(
            token_ids, self.config.block_size, extra_keys=(model_hash,)
        )

        # 2. Look up cached blocks
        matched_blocks: list[KVBlock] = []
        matched_hashes: list[int] = []
        # Warm-tier promoted blocks have ref_count=1 already (from allocate());
        # we must NOT touch them again, or ref_count becomes 2 and the block
        # leaks when the request is freed.
        _warm_promoted_blocks: set[int] = set()

        for h in block_hashes:
            self._total_lookups += 1
            cached = self.block_pool.lookup_hash(h)
            if cached is not None:
                matched_blocks.append(cached)
                matched_hashes.append(h)
                self._total_hits += 1
            elif self._warm_tier is not None and self._warm_tier.contains(h):
                # Warm tier hit: promote back to hot tier.
                # Allocate BEFORE promoting to avoid data loss if allocation fails.
                try:
                    new_block = self.block_pool.allocate(1)[0]
                except ValueError:
                    logger.warning(
                        "Warm tier promotion skipped: no free blocks for hash 0x%x",
                        h,
                    )
                    break
                promoted = self._warm_tier.promote(h)
                if promoted is not None:
                    self._total_hits += 1
                    # Register in prefix cache so future lookups can find it.
                    # Directly setting block_hash without cache_block() would
                    # make the block invisible to lookup_hash().
                    self.block_pool.cache_block(new_block, h)
                    # Write promoted KV data into the cache tensors.
                    # The packed format is [2, num_heads, block_size, head_dim]
                    # where dim 0 has key at [0] and value at [1].
                    try:
                        if self._key_cache is not None:
                            import mlx.core as mx
                            if promoted.ndim == 4 and promoted.shape[0] == 2:
                                if isinstance(self._key_cache, mx.array):
                                    self._key_cache = self._key_cache.at[new_block.block_id].set(promoted[0])
                                    if self._value_cache is not None:
                                        self._value_cache = self._value_cache.at[new_block.block_id].set(promoted[1])
                                else:
                                    self._key_cache[new_block.block_id] = promoted[0]
                                    if self._value_cache is not None:
                                        self._value_cache[new_block.block_id] = promoted[1]
                            else:
                                if isinstance(self._key_cache, mx.array):
                                    self._key_cache = self._key_cache.at[new_block.block_id].set(promoted)
                                else:
                                    self._key_cache[new_block.block_id] = promoted
                    except Exception:
                        logger.debug(
                            "warm tier KV data write failed for block %d",
                            new_block.block_id, exc_info=True,
                        )
                    # ref_count is already 1 from allocate(); do NOT touch again
                    _warm_promoted_blocks.add(id(new_block))
                    matched_blocks.append(new_block)
                    matched_hashes.append(h)
                    continue
                # Promotion failed (warm tier entry evicted between contains()
                # and promote()) — free the allocated block to prevent a leak.
                self.block_pool.free([new_block])
                break
            else:
                break  # chain hash: once we miss, all subsequent miss

        num_matched_tokens = len(matched_blocks) * self.config.block_size

        # 3. Touch (increase ref count) for cache-hit blocks only.
        #    Warm-tier promoted blocks already have ref_count=1 from allocate().
        for block in matched_blocks:
            if id(block) not in _warm_promoted_blocks:
                self.block_pool.touch(block)

        # 4. Allocate new blocks for the unmatched portion
        remaining_tokens = token_ids[num_matched_tokens:]
        num_new_blocks = (len(remaining_tokens) + self.config.block_size - 1) // self.config.block_size

        new_blocks = []
        if num_new_blocks > 0:
            new_blocks = self.block_pool.allocate(num_new_blocks)

        # 5. Build the BlockTable
        table = BlockTable(self.config.block_size)
        # Add cached blocks first
        for block in matched_blocks:
            table.append_block(block)
        # Then new blocks
        for block in new_blocks:
            table.append_block(block)

        # 6. Cache new blocks that become full (during/after prefill)
        # For now, we cache them after the fact via cache_completed_blocks()

        prefix_match = PrefixMatch(
            matched_blocks=matched_blocks,
            num_matched_tokens=num_matched_tokens,
            unmatched_token_ids=remaining_tokens,
        )

        return table, prefix_match

    def _build_table_from_match(
        self,
        matched_blocks: list[KVBlock],
        num_matched_tokens: int,
        token_ids: list[int],
    ) -> tuple[BlockTable, PrefixMatch]:
        """Build a BlockTable from RadixTree-matched blocks.

        Handles the "gap" when a radix tree split occurs at a non-block-aligned
        position: floor division in _split_node assigns only fully-contained
        blocks to the prefix, so matched_blocks may cover fewer tokens than
        num_matched_tokens. The gap tokens need fresh blocks and re-prefill.
        """
        bs = self.config.block_size
        covered_tokens = len(matched_blocks) * bs
        gap = max(0, num_matched_tokens - covered_tokens)

        remaining_tokens = token_ids[num_matched_tokens:]
        num_new_blocks = (gap + len(remaining_tokens) + bs - 1) // bs

        new_blocks = []
        if num_new_blocks > 0:
            new_blocks = self.block_pool.allocate(num_new_blocks)

        table = BlockTable(bs)
        for block in matched_blocks:
            table.append_block(block)
        for block in new_blocks:
            table.append_block(block)

        # num_matched_tokens stays as-is so the caller knows how many tokens
        # matched in the tree, but the PrefixMatch must indicate that
        # prefill should start from covered_tokens (not num_matched_tokens)
        # to re-fill the gap.
        prefix_match = PrefixMatch(
            matched_blocks=matched_blocks,
            num_matched_tokens=covered_tokens,  # Only count fully-covered tokens
            unmatched_token_ids=token_ids[covered_tokens:],
        )
        return table, prefix_match

    def allocate_block_for_decode(self, table: BlockTable) -> KVBlock:
        """Allocate one more block when decode fills the current block.

        Before allocating, performs COW on the last block if it's shared
        (ref_count > 1 from prefix cache). This ensures that KV data
        written during decode doesn't corrupt other requests sharing the
        same prefix block.
        """
        with self._lock:
            # COW the last block if shared — the decode may write into it
            # during the transition from partial → full → new block
            blocks = table.get_blocks()
            if blocks:
                last = blocks[-1]
                if last.ref_count > 1:
                    _, self._key_cache, self._value_cache = self.block_pool.cow_block_in_table(
                        table, len(blocks) - 1,
                        self._key_cache, self._value_cache,
                    )

            block = self.block_pool.allocate(1)[0]
            table.append_block(block)
            return block

    def cache_completed_blocks(
        self,
        table: BlockTable,
        token_ids: list[int],
        model_hash: int = 0,
    ) -> int:
        """Register newly completed blocks in the prefix cache.

        Call this after prefill or when a decode fills a block.

        Returns:
            Number of newly cached blocks.
        """
        if not self.config.enable_caching:
            return 0

        with self._lock:
            return self._cache_completed_blocks_unlocked(table, token_ids, model_hash)

    def _cache_completed_blocks_unlocked(
        self,
        table: BlockTable,
        token_ids: list[int],
        model_hash: int = 0,
    ) -> int:
        """Internal: register newly completed blocks (caller holds lock)."""
        blocks = table.get_blocks()
        num_full = len(token_ids) // self.config.block_size
        cached = 0

        # Walk blocks from index 0 to compute the correct chain hash.
        # If a block in the chain was evicted (block_hash cleared to None),
        # we must re-derive the parent hash from the token sequence rather
        # than reading a stale None from the block.
        parent_hash: int | None = None
        for i in range(num_full):
            if i >= len(blocks):
                break
            block = blocks[i]
            start = i * self.config.block_size
            end = start + self.config.block_size
            block_tokens = token_ids[start:end]

            if block.block_hash is not None:
                # Already cached — use its hash as the parent for the next block.
                parent_hash = block.block_hash
                continue

            h = compute_block_hash(parent_hash, block_tokens, (model_hash,))
            self.block_pool.cache_block(block, h)
            parent_hash = h
            cached += 1

        return cached

    def cache_to_radix_tree(
        self,
        token_ids: list[int],
        blocks: list[KVBlock],
        block_hashes: list[int],
    ) -> None:
        """Insert completed blocks into the RadixTree (C8: SGLang pattern).

        The tree enables O(k) prefix matching for future requests.

        Args:
            token_ids: Full token sequence (prompt + output).
            blocks: KVBlocks with non-None block_hash (must be 1:1 with block_hashes).
            block_hashes: Hashes for the blocks.
        """
        if not self.config.enable_caching:
            return
        if len(blocks) != len(block_hashes):
            logger.warning(
                "cache_to_radix_tree: blocks(%d) != block_hashes(%d), skipping",
                len(blocks), len(block_hashes),
            )
            return

        # Find the longest existing prefix match
        matched_node, remaining_tokens = self._radix_tree.match(token_ids)
        matched_len = len(token_ids) - len(remaining_tokens)

        if remaining_tokens:
            # Insert new nodes for the unmatched portion
            bs = self.config.block_size
            # Floor division: the boundary block (the block that straddles
            # the match boundary) must be INCLUDED in new_blocks.  Using
            # ceiling division previously skipped this block, losing its KV
            # data for the unmatched portion of the token sequence.
            # The radix tree's split logic handles the partial overlap
            # correctly — it splits the boundary block's node at the
            # exact token boundary.
            new_start_block = (matched_len + bs - 1) // bs
            if new_start_block > len(blocks):
                return  # Defensive: matched more than we have blocks for
            new_blocks = blocks[new_start_block:]
            new_hashes = block_hashes[new_start_block:]

            self._radix_tree.insert(
                token_ids=remaining_tokens,
                blocks=new_blocks,
                block_hashes=new_hashes,
                start_node=matched_node if not matched_node.is_root else None,
            )
            # Free old blocks from exact-match replacements in the radix tree
            if matched_node is not None and hasattr(matched_node, '_replaced_blocks'):
                replaced = matched_node._replaced_blocks
                if replaced:
                    self.block_pool.free(replaced)
                    matched_node._replaced_blocks = []

    def free_request(self, table: BlockTable, request_id: str | None = None) -> None:
        """Free all blocks held by a request.

        Args:
            table: The request's block table to free.
            request_id: Optional request ID to release radix tree refs (Bug 2).
        """
        with self._lock:
            # Bug 2: dec_ref radix tree nodes held by this request.
            if request_id is not None and request_id in self._request_nodes:
                node = self._request_nodes.pop(request_id)
                self._radix_tree.dec_ref(node)

            blocks = table.clear()
            self.block_pool.free(blocks)

            # Publish request_freed event for distributed cache coherency.
            if request_id is not None:
                block_ids = [b.block_id for b in blocks]
                self._event_bus.publish(CacheEvent(
                    "request_freed",
                    block_ids=block_ids,
                    node_id=request_id,
                ))

    def evict_for_memory(self, needed_blocks: int) -> bool:
        """Try to evict cached blocks to free up space.

        Strategy:
        1. Blocks with ref_count==1 and block_hash: only held by prefix
           cache (no active request). These can be freed to yield new
           free blocks.
        2. Blocks with ref_count==0 and block_hash: already in the free
           queue -- just clear stale hash entries (no new free blocks).
        3. Blocks with ref_count>1: actively shared -- cannot evict.

        Eviction order: LRU by last_access_time (oldest first).

        If a warm tier is configured, evicted blocks are demoted to 4-bit
        quantized storage instead of being lost entirely.

        Returns:
            True if enough blocks were freed.
        """
        with self._lock:
            return self._evict_for_memory_unlocked(needed_blocks)

    def _evict_for_memory_unlocked(self, needed_blocks: int) -> bool:
        """Internal: try to evict cached blocks to free up space (caller holds lock)."""
        initial_free = self.block_pool.get_free_block_count()

        cached = self.block_pool.get_cached_blocks()
        # Sort by last_access_time (LRU: oldest first) for deterministic
        # eviction order instead of arbitrary dict iteration.
        cached.sort(key=lambda b: b.last_access_time)

        for block in cached:
            if block.block_hash is None:
                continue
            if self.block_pool.get_free_block_count() >= needed_blocks:
                break

            if block.ref_count > 1:
                # Actively shared by multiple requests -- cannot evict
                continue

            # Demote to warm tier (if we have KV data to compress)
            demoted_ok = False
            if self._warm_tier is not None and self._key_cache is not None:
                try:
                    block_idx = block.block_id
                    key_slice = self._key_cache[block_idx]
                    val_slice = self._value_cache[block_idx] if self._value_cache is not None else None
                    # Pack key + value into a single array for warm storage.
                    # On promotion, the caller must split them back.
                    import mlx.core as mx
                    if val_slice is not None:
                        kv_packed = mx.stack([key_slice, val_slice], axis=0)
                    else:
                        kv_packed = key_slice
                    demoted_ok = self._warm_tier.demote(block.block_hash, kv_packed, num_tokens=self.config.block_size)
                except Exception:
                    logger.debug("warm tier demote failed in evict_for_memory", exc_info=True)

            if not demoted_ok and self._warm_tier is not None:
                # Demotion failed or was skipped — remove any stale warm-tier
                # entry for this hash so future promotions don't return
                # outdated KV data.
                self._warm_tier.remove(block.block_hash)

            # Re-check ref_count BEFORE evicting — BlockPool.touch() may have
            # reactivated this block between snapshot and now. Evicting a
            # block with ref_count > 1 would strip the prefix cache entry
            # from an actively-shared block.
            if block.ref_count > 1:
                continue  # actively shared by multiple requests

            # Atomically evict from prefix cache and free the block under a
            # single block_pool lock acquisition. The old two-step approach
            # (_evict_cached_block + free) used separate lock scopes, which
            # allowed concurrent allocate() to pop the block from the free
            # queue between the two steps — effectively stealing and then
            # "un-allocating" the block from its new owner.
            self.block_pool.evict_and_free(block)

        freed_block_count = self.block_pool.get_free_block_count() - initial_free

        # Prune stale radix tree nodes (Bug 1: evict() was never called,
        # causing unbounded memory growth in the tree structure).
        if freed_block_count > 0:
            # Evict roughly the number of tree nodes whose blocks were freed.
            # Use ceiling: each freed block may correspond to one tree node.
            freed_tree_blocks = self._radix_tree.evict(max(1, freed_block_count))
            if freed_tree_blocks:
                self.block_pool.free(freed_tree_blocks)

        return self.block_pool.get_free_block_count() >= needed_blocks

    def memory_pressure_evict(self, pressure_threshold: float = 0.90) -> int:
        """Proactively evict cached blocks when memory utilization is high.

        Called by the scheduler periodically. When usage exceeds the
        threshold, evicts the oldest (LRU) cached blocks to bring
        usage below the threshold.

        Eviction order: LRU by last_access_time (oldest first).

        Args:
            pressure_threshold: Eviction triggers when usage exceeds this (0.0-1.0).

        Returns:
            Number of blocks evicted.
        """
        with self._lock:
            return self._memory_pressure_evict_unlocked(pressure_threshold)

    def _memory_pressure_evict_unlocked(self, pressure_threshold: float = 0.90) -> int:
        """Internal: proactively evict cached blocks (caller holds lock)."""
        current_usage = self.usage
        if current_usage < pressure_threshold:
            return 0

        total_blocks = len(self.block_pool.blocks) - 1  # exclude null block
        target_usage = pressure_threshold * 0.85  # Evict to 85% of threshold
        target_used = int(total_blocks * target_usage)
        current_used = total_blocks - self.num_free_blocks
        blocks_to_free = current_used - target_used

        if blocks_to_free <= 0:
            return 0

        initial_free = self.block_pool.get_free_block_count()
        cached = self.block_pool.get_cached_blocks()
        # Sort by last_access_time (LRU: oldest first) for deterministic
        # eviction order instead of arbitrary dict iteration.
        cached.sort(key=lambda b: b.last_access_time)

        for block in cached:
            if self.block_pool.get_free_block_count() - initial_free >= blocks_to_free:
                break
            if block.block_hash is None:
                continue

            if block.ref_count > 1:
                continue

            # Demote to warm tier if available
            demoted_ok = False
            if self._warm_tier is not None and self._key_cache is not None:
                try:
                    block_idx = block.block_id
                    key_slice = self._key_cache[block_idx]
                    val_slice = self._value_cache[block_idx] if self._value_cache is not None else None
                    import mlx.core as mx
                    if val_slice is not None:
                        kv_packed = mx.stack([key_slice, val_slice], axis=0)
                    else:
                        kv_packed = key_slice
                    demoted_ok = self._warm_tier.demote(block.block_hash, kv_packed, num_tokens=self.config.block_size)
                except Exception:
                    logger.debug("warm tier demote failed in memory_pressure_evict", exc_info=True)

            if not demoted_ok and self._warm_tier is not None:
                # Demotion failed or was skipped — remove any stale warm-tier
                # entry for this hash so future promotions don't return
                # outdated KV data.
                self._warm_tier.remove(block.block_hash)

            # Re-check ref_count BEFORE evicting — concurrent touch() may
            # have reactivated this block. Evicting a ref_count > 1 block
            # strips the prefix cache entry from actively-shared blocks.
            if block.ref_count > 1:
                continue

            # Atomically evict + free under a single block_pool lock.
            # See _evict_for_memory_unlocked for the TOCTOU rationale.
            self.block_pool.evict_and_free(block)

        evicted = self.block_pool.get_free_block_count() - initial_free

        # Prune stale radix tree nodes (Bug 1: evict() was never called,
        # causing unbounded memory growth in the tree structure).
        if evicted > 0:
            freed_tree_blocks = self._radix_tree.evict(max(1, evicted))
            if freed_tree_blocks:
                self.block_pool.free(freed_tree_blocks)

        if evicted > 0:
            logger.debug(
                f"Memory pressure eviction: freed {evicted} blocks "
                f"(usage {current_usage:.1%} > {pressure_threshold:.0%})"
            )
        return evicted

    # ── Tier Statistics ────────────────────────────────────────────

    def get_tier_stats(self) -> dict:
        """Return combined hot + warm tier statistics.

        Returns:
            Dictionary with ``hot`` and ``warm`` keys. The ``warm`` key
            is ``None`` when no warm tier is configured.
        """
        hot_stats = {
            "usage_pct": round(self.usage * 100, 1),
            "free_blocks": self.num_free_blocks,
            "block_size": self.block_size,
            "total_lookups": self._total_lookups,
            "total_hits": self._total_hits,
            "hit_rate": round(self.hit_rate, 4),
            **self.block_pool.cow_stats,
        }
        warm_stats = self._warm_tier.get_stats() if self._warm_tier is not None else None
        return {
            "hot": hot_stats,
            "warm": warm_stats,
            "radix_tree": self._radix_tree.get_stats(),
        }

    # ── Serialization ─────────────────────────────────────────────

    def save_prefix(self, prefix_hash: int, path: str) -> None:
        """Save a cached prefix and its KV data to disk.

        Looks up the block with the given hash in the prefix cache and
        serializes it. Requires that the KV cache tensors have been set
        via ``set_kv_tensors()``.

        Args:
            prefix_hash: Block hash identifying the prefix.
            path: File path to write.

        Raises:
            KeyError: If no cached block matches the hash.
        """
        from .serialization import KVCacheSerializer

        block = self.block_pool.lookup_hash(prefix_hash)
        if block is None:
            raise KeyError(f"No cached block with hash {prefix_hash}")

        serializer = KVCacheSerializer()
        key_data = self._key_cache[block.block_id]
        value_data = self._value_cache[block.block_id]
        block_bytes = serializer.serialize_block(block, key_data, value_data)
        with open(path, "wb") as f:
            f.write(block_bytes)

    def load_prefix(self, path: str) -> KVBlock:
        """Load a previously saved prefix block from disk.

        Allocates a fresh block from the pool and copies the deserialized
        KV data into it. The original block_id from disk is NOT reused --
        it may already be in use by another request.

        Args:
            path: File path to read.

        Returns:
            The loaded KVBlock (allocated from pool, registered in prefix cache).

        Raises:
            ValueError: If no free blocks are available.
        """
        from .serialization import KVCacheSerializer

        with open(path, "rb") as f:
            data = f.read()

        serializer = KVCacheSerializer()
        old_block, key_data, value_data = serializer.deserialize_block(data)

        with self._lock:
            # Allocate a fresh block from the pool instead of reusing old block_id
            new_block = self.block_pool.allocate(1)[0]

            # Write data into cache tensors at the NEW block's slot
            if self._key_cache is not None and self._value_cache is not None:
                try:
                    import mlx.core as mx
                    if isinstance(self._key_cache, mx.array):
                        self._key_cache = self._key_cache.at[new_block.block_id].set(key_data)
                        self._value_cache = self._value_cache.at[new_block.block_id].set(value_data)
                    else:
                        self._key_cache[new_block.block_id] = key_data
                        self._value_cache[new_block.block_id] = value_data
                except ImportError:
                    self._key_cache[new_block.block_id] = key_data
                    self._value_cache[new_block.block_id] = value_data

            # Register in prefix cache with the hash from disk
            if old_block.block_hash is not None:
                self.block_pool.cache_block(new_block, old_block.block_hash)

        return new_block

    def save_all_cached(self, path: str) -> None:
        """Save the entire prefix cache to disk.

        Serializes all blocks that have a non-None ``block_hash`` together
        with their KV data.

        Args:
            path: File path to write.
        """
        from .serialization import KVCacheSerializer

        serializer = KVCacheSerializer()
        parts: list[bytes] = []

        cached_blocks = [
            b for b in self.block_pool.blocks if b.block_hash is not None
        ]
        num_cached = len(cached_blocks)

        # Header: number of cached blocks
        import struct
        parts.append(struct.pack(">I", num_cached))

        for block in cached_blocks:
            key_data = self._key_cache[block.block_id]
            value_data = self._value_cache[block.block_id]
            block_bytes = serializer.serialize_block(block, key_data, value_data)
            # Frame: length + data
            parts.append(struct.pack(">I", len(block_bytes)))
            parts.append(block_bytes)

        with open(path, "wb") as f:
            f.write(b"".join(parts))

    def load_cached(self, path: str) -> int:
        """Load a previously saved prefix cache from disk.

        Deserializes all blocks, allocates fresh blocks from the pool for
        each, copies KV data into the new slots, and registers them in
        the prefix cache. The original block_ids from disk are NOT reused.

        Args:
            path: File path to read.

        Returns:
            Number of blocks loaded.
        """
        import struct
        from .serialization import KVCacheSerializer

        with open(path, "rb") as f:
            data = f.read()

        serializer = KVCacheSerializer()
        offset = 0

        (num_cached,) = struct.unpack_from(">I", data, offset)
        offset += 4

        loaded = 0
        with self._lock:
            for _ in range(num_cached):
                (frame_len,) = struct.unpack_from(">I", data, offset)
                offset += 4
                frame = data[offset : offset + frame_len]
                offset += frame_len

                old_block, key_data, value_data = serializer.deserialize_block(frame)

                # Allocate a fresh block instead of reusing old block_id
                try:
                    new_block = self.block_pool.allocate(1)[0]
                except ValueError:
                    logger.warning(
                        "load_cached: ran out of free blocks after loading %d/%d",
                        loaded, num_cached,
                    )
                    break

                # Write data into cache tensors at the new block's slot
                if self._key_cache is not None and self._value_cache is not None:
                    try:
                        import mlx.core as mx
                        if isinstance(self._key_cache, mx.array):
                            self._key_cache = self._key_cache.at[new_block.block_id].set(key_data)
                            self._value_cache = self._value_cache.at[new_block.block_id].set(value_data)
                        else:
                            self._key_cache[new_block.block_id] = key_data
                            self._value_cache[new_block.block_id] = value_data
                    except ImportError:
                        self._key_cache[new_block.block_id] = key_data
                        self._value_cache[new_block.block_id] = value_data

                if old_block.block_hash is not None:
                    self.block_pool.cache_block(new_block, old_block.block_hash)

                loaded += 1

        return loaded
