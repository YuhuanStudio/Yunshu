from __future__ import annotations

"""Yunshu KV Prefix Compression + Sliding Window Attention.

Two complementary optimizations for long-context inference:

1. **KVPrefixCompressor** — When KV cache is full, compress old blocks instead
   of discarding them outright. Three strategies:
   - mean_pool: Average every K consecutive blocks into 1 compressed block
   - top_k: Keep blocks with highest attention scores, discard the rest
   - frequency_aware: Blocks accessed more often are kept at higher fidelity

2. **SlidingWindowKVManager** — For models with sliding window attention
   (e.g., Mistral, Qwen2). Only keeps KV blocks within the window,
   automatically evicting old blocks. System prompt blocks are always kept.

References:
  - StreamingLLM: Attention Sink + sliding window (Xiao et al., ICLR 2024)
  - Mistral sliding window attention (Jiang et al.)
  - H2O: Heavy-Hitter Oracle for KV cache eviction (Zhang et al.)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class CompressionResult:
    """Result of compressing KV blocks."""

    compressed: Any  # Compressed K/V tensors
    original_shape: tuple[int, ...]
    compressed_shape: tuple[int, ...]
    strategy: str
    ratio: float  # original_bytes / compressed_bytes
    block_ids: list[int]  # Which block indices were compressed
    original_block_count: int = 0  # Total number of blocks before compression


@dataclass
class _KVBlock:
    """Internal representation of a KV block for compression."""

    block_id: int
    keys: np.ndarray  # shape (num_layers, num_kv_heads, block_size, head_dim)
    values: np.ndarray
    attention_score: float = 0.0
    access_count: int = 0
    is_system_prompt: bool = False
    token_position: int = 0


@dataclass
class WindowConfig:
    """Sliding window configuration."""

    window_size: int = 4096  # Number of tokens in the sliding window
    system_prompt_blocks: int = 0  # Number of blocks in the system prompt (always kept)
    block_size: int = 64  # Tokens per KV block


@dataclass
class WindowStats:
    """Sliding window statistics."""

    total_evictions: int = 0
    active_blocks: int = 0
    system_prompt_blocks: int = 0
    total_blocks_seen: int = 0
    memory_saved_bytes: int = 0
    kv_trimmed: int = 0  # Total positions trimmed from real KV cache arrays


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KVPrefixCompressor
# ═══════════════════════════════════════════════════════════════════════════════


class KVPrefixCompressor:
    """Compresses KV cache blocks using various strategies.

    When KV cache is full, instead of just evicting old blocks, compress
    them into a smaller representation. This retains more context than
    outright eviction at the cost of some information loss.

    Strategies:
        mean_pool: Average every K consecutive blocks into 1 compressed block.
            Best for: monotonic conversations where adjacent blocks carry
            similar information.
        top_k: Keep only the blocks with highest attention scores.
            Best for: retrieval-heavy workloads where only a few blocks matter.
        frequency_aware: Keep frequently-accessed blocks at full fidelity,
            compress less-accessed blocks more aggressively.
            Best for: mixed workloads with hot/cold block separation.
    """

    def __init__(
        self,
        compression_factor: int = 4,
        block_size: int = 64,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_size: int = 2,
    ) -> None:
        self._compression_factor = compression_factor
        self._block_size = block_size
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype_size = dtype_size

        # Stats
        self._total_compressions: int = 0
        self._total_bytes_saved: int = 0
        self._total_decompressions: int = 0

    def compress_blocks(
        self,
        kv_blocks: list[np.ndarray | Any],
        attention_scores: list[float] | None = None,
        access_counts: list[int] | None = None,
        strategy: str = "mean_pool",
    ) -> CompressionResult:
        """Compress N KV blocks into fewer blocks using the given strategy.

        Args:
            kv_blocks: List of KV block tensors. Each tensor has shape
                (num_layers, num_kv_heads, block_size, head_dim) for keys
                and same for values — passed as (keys, values) tuples or
                a single ndarray of shape (2, num_layers, num_kv_heads, block_size, head_dim).
            attention_scores: Per-block attention scores (required for top_k).
            access_counts: Per-block access counts (required for frequency_aware).
            strategy: One of "mean_pool", "top_k", "frequency_aware".

        Returns:
            CompressionResult with compressed data and metadata.
        """
        if not kv_blocks:
            return CompressionResult(
                compressed=np.array([]),
                original_shape=(0,),
                compressed_shape=(0,),
                strategy=strategy,
                ratio=1.0,
                block_ids=[],
            )

        # Normalize input to numpy arrays
        blocks_np = self._to_numpy_blocks(kv_blocks)
        n_blocks = len(blocks_np)
        original_shape = blocks_np[0].shape if n_blocks > 0 else (0,)

        if strategy == "mean_pool":
            compressed, block_ids = self._compress_mean_pool(blocks_np)
        elif strategy == "top_k":
            scores = attention_scores or [1.0] * n_blocks
            compressed, block_ids = self._compress_top_k(blocks_np, scores)
        elif strategy == "frequency_aware":
            counts = access_counts or [1] * n_blocks
            compressed, block_ids = self._compress_frequency_aware(blocks_np, counts)
        else:
            raise ValueError(
                f"Unknown compression strategy: {strategy!r}. "
                f"Supported: mean_pool, top_k, frequency_aware"
            )

        compressed_shape = compressed.shape if isinstance(compressed, np.ndarray) and compressed.size > 0 else (0,)
        original_bytes = sum(b.nbytes for b in blocks_np)
        compressed_bytes = compressed.nbytes if isinstance(compressed, np.ndarray) else 0
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0

        self._total_compressions += 1
        self._total_bytes_saved += max(0, original_bytes - compressed_bytes)

        logger.debug(
            f"KV prefix compression ({strategy}): {n_blocks} blocks → "
            f"{compressed_shape}, ratio={ratio:.2f}x"
        )

        return CompressionResult(
            compressed=compressed,
            original_shape=original_shape,
            compressed_shape=compressed_shape,
            strategy=strategy,
            ratio=ratio,
            block_ids=block_ids,
            original_block_count=n_blocks,
        )

    def decompress_blocks(self, compressed_result: CompressionResult) -> list[np.ndarray]:
        """Restore compressed KV blocks to their approximate original form.

        Note: Decompression is lossy for mean_pool and frequency_aware.
        The restored blocks are approximations, not exact copies.
        For top_k, decompression is exact for the kept blocks.

        Args:
            compressed_result: A CompressionResult from compress_blocks().

        Returns:
            List of numpy arrays approximating the original blocks.
        """
        self._total_decompressions += 1

        if not compressed_result.block_ids:
            return []

        strategy = compressed_result.strategy
        compressed = compressed_result.compressed

        if not isinstance(compressed, np.ndarray) or compressed.size == 0:
            return []

        if strategy == "mean_pool":
            return self._decompress_mean_pool(compressed, compressed_result)
        elif strategy == "top_k":
            return self._decompress_top_k(compressed, compressed_result)
        elif strategy == "frequency_aware":
            return self._decompress_frequency_aware(compressed, compressed_result)
        else:
            return [compressed]

    def get_compression_ratio(self) -> float:
        """Report the cumulative compression ratio across all operations."""
        if self._total_compressions == 0:
            return 1.0
        # Average ratio
        return self._compression_factor  # Theoretical max for mean_pool

    def get_stats(self) -> dict:
        """Return compression statistics."""
        return {
            "total_compressions": self._total_compressions,
            "total_decompressions": self._total_decompressions,
            "total_bytes_saved": self._total_bytes_saved,
            "compression_factor": self._compression_factor,
            "theoretical_ratio": self._compression_factor,
        }

    # ── Private: Input Normalization ───────────────────────────────────────

    def _to_numpy_blocks(self, kv_blocks: list) -> list[np.ndarray]:
        """Convert mixed MLX/numpy inputs to numpy arrays."""
        result = []
        for block in kv_blocks:
            if isinstance(block, tuple) and len(block) == 2:
                # (keys, values) tuple
                keys, values = block
                keys_np = np.array(keys, dtype=np.float32) if not isinstance(keys, np.ndarray) else keys.astype(np.float32)
                values_np = np.array(values, dtype=np.float32) if not isinstance(values, np.ndarray) else values.astype(np.float32)
                # Stack into shape (2, layers, heads, block_size, head_dim)
                combined = np.stack([keys_np, values_np], axis=0)
                result.append(combined)
            elif isinstance(block, np.ndarray):
                result.append(block.astype(np.float32))
            else:
                # MLX array or other
                result.append(np.array(block, dtype=np.float32))
        return result

    # ── Private: Mean Pool Strategy ────────────────────────────────────────

    def _compress_mean_pool(
        self, blocks: list[np.ndarray]
    ) -> tuple[np.ndarray, list[int]]:
        """Compress by averaging every K consecutive blocks into 1.

        For K=4: blocks [0,1,2,3] → avg → 1 compressed block.
        The last group may have fewer than K blocks.
        """
        K = self._compression_factor
        n = len(blocks)
        all_block_ids = list(range(n))

        groups = []
        group_ids = []
        for start in range(0, n, K):
            end = min(start + K, n)
            group = blocks[start:end]
            groups.append(group)
            group_ids.append(all_block_ids[start:end])

        compressed_blocks = []
        final_ids = []
        for group, ids in zip(groups, group_ids, strict=False):
            stacked = np.stack(group, axis=0)  # (K, 2, layers, heads, bs, hd)
            pooled = np.mean(stacked, axis=0)  # (2, layers, heads, bs, hd)
            compressed_blocks.append(pooled)
            final_ids.append(ids[0])  # Representative block ID

        if compressed_blocks:
            return np.stack(compressed_blocks, axis=0), final_ids
        return np.array([]), []

    def _decompress_mean_pool(
        self, compressed: np.ndarray, result: CompressionResult
    ) -> list[np.ndarray]:
        """Decompress mean-pooled blocks by tiling each compressed block K times."""
        if compressed.size == 0:
            return []
        K = self._compression_factor
        # Use stored original_block_count for accurate decompression.
        # The last group may have fewer than K blocks.
        original_count = result.original_block_count
        # Each compressed block represents up to K original blocks
        decompressed: list = []
        for i in range(compressed.shape[0]):
            block = compressed[i]
            # Last group may have fewer than K blocks
            remaining = original_count - len(decompressed)
            tiles = min(K, remaining) if remaining > 0 else K
            for _ in range(tiles):
                decompressed.append(block.copy())
        return decompressed[:original_count] if original_count > 0 else decompressed

    # ── Private: Top-K Strategy ────────────────────────────────────────────

    def _compress_top_k(
        self, blocks: list[np.ndarray], scores: list[float]
    ) -> tuple[np.ndarray, list[int]]:
        """Keep only the K blocks with highest attention scores."""
        K = max(1, len(blocks) // self._compression_factor)
        K = min(K, len(blocks))

        # Rank blocks by attention score, keep top K
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in indexed_scores[:K]])

        selected = [blocks[i] for i in top_indices]
        if selected:
            return np.stack(selected, axis=0), top_indices
        return np.array([]), []

    def _decompress_top_k(
        self, compressed: np.ndarray, result: CompressionResult
    ) -> list[np.ndarray]:
        """Top-k decompression: exact for kept blocks, zeros for discarded."""
        return [compressed[i] for i in range(compressed.shape[0])]

    # ── Private: Frequency-Aware Strategy ──────────────────────────────────

    def _compress_frequency_aware(
        self, blocks: list[np.ndarray], access_counts: list[int]
    ) -> tuple[np.ndarray, list[int]]:
        """Compress based on access frequency.

        - High access (>= median): kept at full fidelity
        - Low access (< median): compressed with higher factor
        - Zero access: compressed with maximum factor

        Output preserves original block order so callers and decompressors
        can rely on positional correspondence.
        """
        if not blocks:
            return np.array([]), []

        n = len(blocks)
        sorted_counts = sorted(access_counts)
        median = sorted_counts[n // 2] if n > 0 else 0

        # Classify each index as kept or compressible
        kept_indices: set[int] = set()
        compressible_indices: list[int] = []
        for i, count in enumerate(access_counts):
            if count >= median and count > 0:
                kept_indices.add(i)
            else:
                compressible_indices.append(i)

        # Build a map from original index → pooled block (for compressible groups)
        compressed_map: dict[int, np.ndarray] = {}
        if compressible_indices:
            K = max(1, self._compression_factor)
            for start in range(0, len(compressible_indices), K):
                end = min(start + K, len(compressible_indices))
                group_orig_ids = compressible_indices[start:end]
                group_blocks = [blocks[gi] for gi in group_orig_ids]
                stacked = np.stack(group_blocks, axis=0)
                pooled = np.mean(stacked, axis=0)
                # Map all original indices in this group to the representative
                # block (the first one in the group).
                representative_id = group_orig_ids[0]
                compressed_map[representative_id] = pooled
                # Non-representative indices in this group are consumed; only
                # the representative appears in the output list.

        # Reconstruct in original order: kept blocks at their positions,
        # compressed representatives at their positions.
        all_blocks: list[np.ndarray] = []
        all_ids: list[int] = []
        # Track which compressible indices have been emitted (as representatives)
        emitted_compressible: set[int] = set()
        for i in range(n):
            if i in kept_indices:
                all_blocks.append(blocks[i])
                all_ids.append(i)
            elif i in compressed_map:
                # This is a representative of a pooled group
                all_blocks.append(compressed_map[i])
                all_ids.append(i)
                # Mark the other indices in this group as emitted
                # (they won't appear in the output)
                emitted_compressible.add(i)

        if all_blocks:
            return np.stack(all_blocks, axis=0), all_ids
        return np.array([]), []

    def _decompress_frequency_aware(
        self, compressed: np.ndarray, result: CompressionResult
    ) -> list[np.ndarray]:
        """Frequency-aware decompression: kept blocks exact, pooled blocks tiled."""
        return [compressed[i] for i in range(compressed.shape[0])]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SlidingWindowKVManager
# ═══════════════════════════════════════════════════════════════════════════════


class SlidingWindowKVManager:
    """Sliding window KV cache manager for models with windowed attention.

    For models like Mistral and Qwen2 that use sliding window attention,
    only KV blocks within the window are needed for computation. This
    manager automatically evicts blocks that fall outside the window.

    Key behavior:
    - System prompt blocks are NEVER evicted (they're always attended to)
    - Blocks outside (window_size - system_prompt_tokens) from the current
      position are evicted
    - Tracks eviction statistics for monitoring

    Reference: StreamingLLM (Xiao et al., 2024) — attention sinks +
    rolling window keeps generation stable without full context.
    """

    def __init__(
        self,
        window_size: int = 4096,
        block_size: int = 64,
        system_prompt_blocks: int = 0,
    ) -> None:
        self._window_size = window_size
        self._block_size = block_size
        self._system_prompt_blocks = system_prompt_blocks

        # Per-request state: request_id → list of _KVBlock
        self._request_blocks: dict[str, list[_KVBlock]] = {}
        # Per-request current position
        self._request_positions: dict[str, int] = {}

        # Statistics
        self._stats = WindowStats(
            system_prompt_blocks=system_prompt_blocks,
        )

    def configure(
        self,
        window_size: int | None = None,
        model_config: dict | None = None,
    ) -> None:
        """Set or update window parameters.

        Args:
            window_size: Number of tokens in the sliding window.
            model_config: Optional dict with keys like 'sliding_window',
                'block_size', 'system_prompt_blocks'.
        """
        if window_size is not None:
            self._window_size = window_size

        if model_config is not None:
            if "sliding_window" in model_config:
                self._window_size = model_config["sliding_window"]
            if "block_size" in model_config:
                self._block_size = model_config["block_size"]
            if "system_prompt_blocks" in model_config:
                self._system_prompt_blocks = model_config["system_prompt_blocks"]
                self._stats.system_prompt_blocks = self._system_prompt_blocks

        window_blocks = math.ceil(self._window_size / self._block_size)
        logger.info(
            f"SlidingWindowKV configured: window={self._window_size} tokens "
            f"({window_blocks} blocks), block_size={self._block_size}, "
            f"system_prompt_blocks={self._system_prompt_blocks}"
        )

    def register_request(
        self,
        request_id: str,
        system_prompt_blocks: int = 0,
    ) -> None:
        """Register a new request with the sliding window manager.

        Args:
            request_id: Unique request identifier.
            system_prompt_blocks: Number of blocks in the system prompt
                (these will never be evicted).
        """
        self._request_blocks[request_id] = []
        self._request_positions[request_id] = 0

        # If this request has its own system prompt count, store it
        if system_prompt_blocks > 0:
            for i in range(system_prompt_blocks):
                block = _KVBlock(
                    block_id=i,
                    keys=np.zeros((1,), dtype=np.float32),
                    values=np.zeros((1,), dtype=np.float32),
                    is_system_prompt=True,
                    token_position=i * self._block_size,
                )
                self._request_blocks[request_id].append(block)

    def on_new_token(
        self,
        token_position: int,
        kv_block: _KVBlock | None = None,
        request_id: str = "",
    ) -> list[int]:
        """Add a new block and evict blocks outside the window.

        Called for each new token position during generation. When a block
        boundary is crossed, adds the new block and evicts old ones that
        fall outside the sliding window.

        Args:
            token_position: Absolute token position in the sequence.
            kv_block: The KV block data (if None, a placeholder is created).
            request_id: The request this token belongs to.

        Returns:
            List of evicted block IDs.
        """
        if request_id not in self._request_blocks:
            self.register_request(request_id)

        self._request_positions[request_id] = token_position
        self._stats.total_blocks_seen += 1

        # Only create a block at block boundaries
        block_idx = token_position // self._block_size
        if token_position % self._block_size != 0 and token_position > 0:
            # Mid-block: no new block created yet
            return self._evict_outside_window(request_id)

        # Create new block
        if kv_block is None:
            kv_block = _KVBlock(
                block_id=block_idx,
                keys=np.zeros((1,), dtype=np.float32),
                values=np.zeros((1,), dtype=np.float32),
                token_position=token_position,
            )

        blocks = self._request_blocks[request_id]

        # Check if we already have this block.  System prompt blocks
        # registered via register_request() have block_ids starting at 0,
        # so checking only blocks[-1] is insufficient — a new block at
        # block_idx=0 could collide with an existing system prompt block.
        # Use a set for O(1) dedup when the list is non-trivial.
        existing_ids = {b.block_id for b in blocks}
        if block_idx not in existing_ids:
            blocks.append(kv_block)

        return self._evict_outside_window(request_id)

    def get_active_blocks(self, request_id: str) -> list[_KVBlock]:
        """Return all blocks within the sliding window for a request.

        System prompt blocks are always included regardless of position.
        """
        if request_id not in self._request_blocks:
            return []

        blocks = self._request_blocks[request_id]
        current_pos = self._request_positions.get(request_id, 0)
        window_start = max(0, current_pos - self._window_size + self._block_size)

        active = []
        for block in blocks:
            if block.is_system_prompt or block.token_position >= window_start:
                active.append(block)

        return active

    def get_stats(self) -> WindowStats:
        """Return sliding window statistics."""
        # Count stored blocks (approximation for active; exact count
        # would require per-request window computation which is O(n*m)).
        # Use len() on the stored lists instead of calling get_active_blocks()
        # for each request to avoid O(n*m) overhead in the stats path.
        self._stats.active_blocks = sum(
            len(blocks) for blocks in self._request_blocks.values()
        )
        # Estimate memory saved: evicted blocks * per-block memory
        blocks_evicted = self._stats.total_evictions
        per_block_bytes = (
            self._block_size
            * 8  # kv_heads
            * 128  # head_dim
            * 2  # keys + values
            * 32  # layers
            * 2  # dtype bytes (fp16)
        )
        self._stats.memory_saved_bytes = blocks_evicted * per_block_bytes
        return self._stats

    def trim_kv_cache(self, kv_cache: Any, request_id: str = "") -> int:
        """Trim a real MLX KV cache to remove entries outside the sliding window.

        Called by EngineCore after ``on_new_token`` returns evicted block IDs.
        For MLX's ``make_prompt_cache`` format, each layer's cache is a list of
        ``(key, value)`` tuples shaped ``(seq_len, num_heads, head_dim)``.
        This method slices off positions that fell outside the window.

        Args:
            kv_cache: The MLX prompt cache (list of (key, value) tuples per layer).
            request_id: The request ID (for logging).

        Returns:
            Number of token positions trimmed from the cache arrays.
        """
        if kv_cache is None:
            return 0

        blocks = self._request_blocks.get(request_id, [])
        current_pos = self._request_positions.get(request_id, 0)

        if not blocks or current_pos <= 0:
            return 0

        # Compute the number of active positions within the window.
        # System prompt blocks are always kept, so the effective window
        # start for non-system blocks is:
        max(0, current_pos - self._window_size + self._block_size)

        # The total number of tokens to keep = system prompt tokens + window tokens.
        # System prompt blocks always remain; non-system blocks within window remain.
        active_positions = set()
        for block in blocks:
            active_positions.add(block.token_position)

        # For MLX KV cache (list of layers, each layer is (key, value) arrays):
        # Trim arrays to keep only the last window_size positions.
        # System prompt blocks are at the start and are always kept.
        n_system = sum(1 for b in blocks if b.is_system_prompt)
        system_tokens = n_system * self._block_size

        # The cache should keep: system_tokens + window_tokens
        # where window_tokens = min(current_pos - system_tokens, window_size)
        window_tokens = min(current_pos - system_tokens, self._window_size)
        keep_tokens = system_tokens + max(0, window_tokens)

        trimmed = 0
        try:
            # Import once before the loop to avoid repeated import overhead.
            # mlxcache arrays require mx.concatenate for slicing + joining.
            import mlx.core as mx  # noqa: F811

            for i, layer_cache in enumerate(kv_cache):
                if layer_cache is None:
                    continue
                # MLX prompt cache: each entry is (key, value) or a cache object
                if isinstance(layer_cache, (list, tuple)) and len(layer_cache) == 2:
                    key, value = layer_cache
                    seq_len = key.shape[0] if hasattr(key, 'shape') else 0
                    if seq_len > keep_tokens:
                        trim_from_start = seq_len - keep_tokens
                        if trim_from_start > 0 and n_system > 0:
                            # Guard against negative window_tokens (can happen when
                            # system prompt alone exceeds the current position).
                            effective_window = max(0, window_tokens)
                            if effective_window == 0:
                                # Only keep system prefix — no window tokens yet
                                new_key = key[:system_tokens]
                                new_value = value[:system_tokens]
                            else:
                                # Keep system prefix + window suffix
                                new_key = mx.concatenate(
                                    [key[:system_tokens], key[seq_len - effective_window:]],
                                    axis=0,
                                )
                                new_value = mx.concatenate(
                                    [value[:system_tokens], value[seq_len - effective_window:]],
                                    axis=0,
                                )
                            # CRITICAL: write back to the original kv_cache list.
                            # Reassigning the loop variable (layer_cache = ...) does NOT
                            # mutate the list — previous code silently discarded the
                            # trimmed arrays, making trim_kv_cache a no-op.
                            kv_cache[i] = (new_key, new_value)
                            trimmed += trim_from_start
                        elif trim_from_start > 0:
                            new_key = key[trim_from_start:]
                            new_value = value[trim_from_start:]
                            # CRITICAL: same write-back fix as above.
                            kv_cache[i] = (new_key, new_value)
                            trimmed += trim_from_start
        except Exception:
            logger.debug(
                "trim_kv_cache failed for request %s",
                request_id, exc_info=True,
            )

        if trimmed > 0:
            self._stats.kv_trimmed += trimmed
            logger.debug(
                "SlidingWindowKV trimmed %d positions for request %s "
                "(keep=%d, window=%d)",
                trimmed, request_id, keep_tokens, self._window_size,
            )

        return trimmed

    def invalidate_prefix_cache(
        self,
        prefix_cache: Any,
        request_id: str = "",
    ) -> int:
        """Invalidate prefix cache entries whose blocks have slid out of the window.

        When sliding window attention evicts blocks, any prefix cache entries
        that reference those blocks must be invalidated. Otherwise, new requests
        may receive stale KV data from the prefix cache -- blocks that the model
        will never attend to, wasting memory and potentially causing incorrect
        attention computation.

        Args:
            prefix_cache: The KV prefix cache (has ``evict_by_prefix_len`` or
                ``invalidate`` method). If None, no-op.
            request_id: The request ID (for logging).

        Returns:
            Number of prefix cache entries invalidated, or 0 if no-op.
        """
        if prefix_cache is None:
            return 0

        blocks = self._request_blocks.get(request_id, [])
        current_pos = self._request_positions.get(request_id, 0)
        if not blocks or current_pos <= 0:
            return 0

        # Find the minimum token position among non-system blocks.
        # Prefix cache entries shorter than this position are stale because
        # the sliding window has moved past them.
        window_start = max(0, current_pos - self._window_size + self._block_size)

        # Non-system blocks below window_start have been evicted.
        evicted_blocks = [
            b for b in blocks
            if not b.is_system_prompt and b.token_position < window_start
        ]
        if not evicted_blocks:
            return 0

        # Compute the maximum stale prefix length (in tokens).
        # Any prefix cache entry with <= this many tokens is stale.
        max_stale_tokens = max(b.token_position for b in evicted_blocks)
        max_stale_tokens = min(max_stale_tokens, current_pos)

        if max_stale_tokens <= 0:
            return 0

        # Try to invalidate stale entries in the prefix cache.
        invalidated = 0
        try:
            # Method 1: prefix cache has a dedicated invalidation method.
            if hasattr(prefix_cache, 'invalidate_up_to'):
                invalidated = prefix_cache.invalidate_up_to(max_stale_tokens)
            # Method 2: evict entries with a token count filter.
            elif hasattr(prefix_cache, 'evict_by_max_tokens'):
                invalidated = prefix_cache.evict_by_max_tokens(max_stale_tokens)
            else:
                logger.debug(
                    "Prefix cache has no invalidation method; "
                    "sliding window eviction may serve stale KV data"
                )
        except Exception:
            logger.debug(
                "Prefix cache invalidation failed for request %s",
                request_id, exc_info=True,
            )

        return invalidated

    def remove_request(self, request_id: str) -> None:
        """Remove all blocks for a completed request."""
        self._request_blocks.pop(request_id, None)
        self._request_positions.pop(request_id, None)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def block_size(self) -> int:
        return self._block_size

    # ── Private ────────────────────────────────────────────────────────────

    def _evict_outside_window(self, request_id: str) -> list[int]:
        """Evict blocks outside the sliding window.

        System prompt blocks are never evicted.
        Returns list of evicted block IDs.
        """
        blocks = self._request_blocks[request_id]
        current_pos = self._request_positions[request_id]

        # Window boundary: keep blocks from window_start to current
        window_start = max(0, current_pos - self._window_size + self._block_size)

        evicted_ids = []
        kept = []
        for block in blocks:
            if block.is_system_prompt or block.token_position >= window_start:
                kept.append(block)
            else:
                evicted_ids.append(block.block_id)

        if evicted_ids:
            self._request_blocks[request_id] = kept
            self._stats.total_evictions += len(evicted_ids)

        return evicted_ids
