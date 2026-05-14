"""Yunshu KV Cache Manager — UMA-resident paged KV cache with prefix sharing.

Orchestrates BlockPool (allocation/eviction) + BlockTable (per-request mapping)
+ MLX KV cache tensors (actual key/value data in unified memory).

Three-tier KV hierarchy (Phase 1 implements hot tier only):
  Hot (UMA FP16) → Warm (TurboQuant 4-bit) → Cold (SSD)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

from .block import BlockPool, KVBlock
from .block_table import BlockTable
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

    def __init__(self, config: KVCacheConfig, num_blocks: int) -> None:
        self.config = config
        self.block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=config.block_size,
            enable_caching=config.enable_caching,
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
        self._radix_tree = RadixTree()
        self._request_nodes: dict[str, Any] = {}  # request_id → RadixNode

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
    ) -> tuple[BlockTable, PrefixMatch]:
        """Allocate blocks for a new request, checking prefix cache first.

        Uses RadixTree for O(k) prefix matching (C8), falls back to
        hash-chain lookup for individual blocks.

        Returns:
            (BlockTable for the request, PrefixMatch describing cache hit)
        """
        # 0. Try RadixTree prefix match (C8: O(k) tree traversal)
        if self.config.enable_caching and len(token_ids) >= self.config.block_size:
            matched_node, remaining = self._radix_tree.match(token_ids)
            matched_blocks = matched_node.path_blocks()
            num_matched_tokens = matched_node.total_tokens()
            if num_matched_tokens >= self.config.block_size:
                # RadixTree hit: reuse matched blocks
                self._total_hits += num_matched_tokens // self.config.block_size
                self._total_lookups += len(token_ids) // self.config.block_size
                for block in matched_blocks:
                    self.block_pool.touch(block)
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

        for h in block_hashes:
            self._total_lookups += 1
            cached = self.block_pool.lookup_hash(h)
            if cached is not None:
                matched_blocks.append(cached)
                matched_hashes.append(h)
                self._total_hits += 1
            elif self._warm_tier is not None and self._warm_tier.contains(h):
                # Warm tier hit: promote back to hot tier
                promoted = self._warm_tier.promote(h)
                if promoted is not None:
                    self._total_hits += 1
                    # Re-allocate a hot block and copy the promoted data
                    new_block = self.block_pool.allocate(1)[0]
                    new_block.block_hash = h
                    new_block.ref_count = 1
                    matched_blocks.append(new_block)
                    matched_hashes.append(h)
                    continue
                break
            else:
                break  # chain hash: once we miss, all subsequent miss

        num_matched_tokens = len(matched_blocks) * self.config.block_size

        # 3. Touch (increase ref count) for all matched blocks
        for block in matched_blocks:
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
        """Build a BlockTable from RadixTree-matched blocks."""
        remaining_tokens = token_ids[num_matched_tokens:]
        num_new_blocks = (len(remaining_tokens) + self.config.block_size - 1) // self.config.block_size

        new_blocks = []
        if num_new_blocks > 0:
            new_blocks = self.block_pool.allocate(num_new_blocks)

        table = BlockTable(self.config.block_size)
        for block in matched_blocks:
            table.append_block(block)
        for block in new_blocks:
            table.append_block(block)

        prefix_match = PrefixMatch(
            matched_blocks=matched_blocks,
            num_matched_tokens=num_matched_tokens,
            unmatched_token_ids=remaining_tokens,
        )
        return table, prefix_match

    def register_request_node(self, request_id: str, node: Any) -> None:
        """Register a RadixNode for a request (for ref counting)."""
        self._request_nodes[request_id] = node
        self._radix_tree.inc_ref(node)

    def release_request_node(self, request_id: str) -> None:
        """Release a request's RadixNode reference."""
        node = self._request_nodes.pop(request_id, None)
        if node is not None:
            self._radix_tree.dec_ref(node)

    def allocate_block_for_decode(self, table: BlockTable) -> KVBlock:
        """Allocate one more block when decode fills the current block.

        Before allocating, performs COW on the last block if it's shared
        (ref_count > 1 from prefix cache). This ensures that KV data
        written during decode doesn't corrupt other requests sharing the
        same prefix block.
        """
        # COW the last block if shared — the decode may write into it
        # during the transition from partial → full → new block
        blocks = table.get_blocks()
        if blocks:
            last = blocks[-1]
            if last.ref_count > 1:
                self.block_pool.cow_block_in_table(
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

        blocks = table.get_blocks()
        num_full = len(token_ids) // self.config.block_size
        cached = 0

        # Find the first uncached full block
        for i in range(num_full):
            if i >= len(blocks):
                break
            block = blocks[i]
            if block.block_hash is not None:
                continue  # already cached

            start = i * self.config.block_size
            end = start + self.config.block_size
            block_tokens = token_ids[start:end]

            parent_hash = blocks[i - 1].block_hash if i > 0 else None
            h = compute_block_hash(parent_hash, block_tokens, (model_hash,))
            self.block_pool.cache_block(block, h)
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
        """
        if not self.config.enable_caching:
            return

        # Find the longest existing prefix match
        matched_node, remaining_tokens = self._radix_tree.match(token_ids)
        matched_len = len(token_ids) - len(remaining_tokens)
        matched_blocks = matched_node.path_blocks()

        if remaining_tokens:
            # Insert new nodes for the unmatched portion
            bs = self.config.block_size
            new_start_block = matched_len // bs
            new_blocks = blocks[new_start_block:]
            new_hashes = block_hashes[new_start_block:]

            self._radix_tree.insert(
                token_ids=remaining_tokens,
                blocks=new_blocks,
                block_hashes=new_hashes,
                start_node=matched_node if not matched_node.is_root else None,
            )

    def free_request(self, table: BlockTable) -> None:
        """Free all blocks held by a request."""
        blocks = table.clear()
        self.block_pool.free(blocks)

    def evict_for_memory(self, needed_blocks: int) -> bool:
        """Try to evict cached blocks to free up space.

        If a warm tier is configured, evicted blocks are demoted to 4-bit
        quantized storage instead of being lost entirely.

        Returns:
            True if enough blocks were freed.
        """
        # Try to demote cached blocks to warm tier first
        if self._warm_tier is not None:
            cached = self.block_pool.get_cached_blocks()
            for block in list(cached):
                if block.block_hash is None:
                    continue
                if self.block_pool.get_free_block_count() >= needed_blocks:
                    break
                if block.ref_count > 0:
                    continue  # Block is in active use
                # Demote to warm tier (if we have KV data to compress)
                if self._key_cache is not None:
                    try:
                        block_idx = block.block_id
                        kv_slice = self._key_cache[block_idx]
                        self._warm_tier.demote(block.block_hash, kv_slice)
                    except Exception:
                        logger.debug("warm tier demote failed in evict_for_memory", exc_info=True)
                # Remove from hot prefix cache
                self.block_pool._evict_cached_block(block)
                # If the block is still in the free list (ref_count already 0),
                # it's already accounted for; only free if ref_count > 0.
                if block.ref_count > 0:
                    self.block_pool.free([block])

        return self.block_pool.get_free_block_count() >= needed_blocks

    def memory_pressure_evict(self, pressure_threshold: float = 0.90) -> int:
        """Proactively evict cached blocks when memory utilization is high.

        Called by the scheduler periodically. When usage exceeds the
        threshold, evicts the oldest (LRU) cached blocks to bring
        usage below the threshold.

        Args:
            pressure_threshold: Eviction triggers when usage exceeds this (0.0-1.0).

        Returns:
            Number of blocks evicted.
        """
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

        evicted = 0
        cached = self.block_pool.get_cached_blocks()
        # Sort by recency: evict oldest first
        for block in cached:
            if evicted >= blocks_to_free:
                break
            if block.ref_count > 0:
                continue  # In active use
            if block.block_hash is None:
                continue

            # Demote to warm tier if available
            if self._warm_tier is not None and self._key_cache is not None:
                try:
                    kv_slice = self._key_cache[block.block_id]
                    self._warm_tier.demote(block.block_hash, kv_slice)
                except Exception:
                    logger.debug("warm tier demote failed in memory_pressure_evict", exc_info=True)

            self.block_pool._evict_cached_block(block)
            if block.ref_count > 0:
                self.block_pool.free([block])
            evicted += 1

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

        The block's KV data is written into the current KV cache tensors
        at the block's original ``block_id``. A new block is allocated in
        the pool and registered in the prefix cache.

        Args:
            path: File path to read.

        Returns:
            The loaded KVBlock (now in the prefix cache).
        """
        from .serialization import KVCacheSerializer

        with open(path, "rb") as f:
            data = f.read()

        serializer = KVCacheSerializer()
        block, key_data, value_data = serializer.deserialize_block(data)

        # Write data into cache tensors
        self._key_cache[block.block_id] = key_data
        self._value_cache[block.block_id] = value_data

        # Register in prefix cache
        if block.block_hash is not None:
            self.block_pool.cache_block(block, block.block_hash)

        return block

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

        Deserializes all blocks, writes their KV data into cache tensors,
        and registers them in the prefix cache.

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
        for _ in range(num_cached):
            (frame_len,) = struct.unpack_from(">I", data, offset)
            offset += 4
            frame = data[offset : offset + frame_len]
            offset += frame_len

            block, key_data, value_data = serializer.deserialize_block(frame)

            self._key_cache[block.block_id] = key_data
            self._value_cache[block.block_id] = value_data

            if block.block_hash is not None:
                self.block_pool.cache_block(block, block.block_hash)

            loaded += 1

        return loaded
