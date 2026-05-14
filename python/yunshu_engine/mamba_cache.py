"""Yunshu Mamba / Hybrid KV Cache — mixed attention/SSM block cache management.

Addresses audit item §12.2: vLLM supports mixed attention types in the same model
(full attention, sliding window, MLA, Mamba SSM states). Yunshu currently only
supports a single attention type. Hybrid models like Jamba, Zamba, and others
mix Mamba SSM blocks with attention blocks, requiring different cache formats
for each layer.

This module provides:

- **CacheBlockType** enum: identifies the cache format per layer (attention,
  Mamba SSM, sliding window, MLA).
- **HybridKVCache**: manages multiple cache pools, one per block type, with
  per-type block sizes, shapes, and eviction policies.
- **MambaSSMState**: manages Mamba's recurrent state (A, B, C, D matrices
  + discretized parameters), with checkpoint/restore and optional compression.
- **BlockAlignedCacheSplitter**: ensures KV blocks are split at layer
  boundaries and eviction respects complete layer groups.

Integration points:
- Scheduler: ``cache_type`` field in SchedulerConfig, wired into get_stats().
- kv_prefix_cache: per-layer cache type lookup.
- kv_offload: Mamba state offloading (different from KV offloading).

Studied from:
- vLLM's AttentionBackend / cache_engine.py (per-layer cache type dispatch)
- Mamba-SSM's inference cache (recurrent state management)
- SGLang's model runner (mixed attention / SSM support)
"""
from __future__ import annotations

import enum
import io
import logging
import math
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────

# Default block sizes per cache type (tokens per block for attention types,
# state slots for SSM types).
_DEFAULT_BLOCK_SIZES: dict[CacheBlockType, int] = {}  # Forward-filled after enum

# Compression methods for SSM state checkpointing.
_COMPRESS_NONE = 0
_COMPRESS_ZLIB = 1
_COMPRESS_QUANT_8BIT = 2
_COMPRESS_QUANT_4BIT = 3


# ── CacheBlockType Enum ────────────────────────────────────────────────


class CacheBlockType(enum.Enum):
    """Identifies the KV cache format for a model layer.

    Hybrid models (Jamba, Zamba, etc.) interleave different block types
    across layers. Each type has a distinct cache layout and lifecycle.
    """

    ATTENTION = "attention"
    """Standard key-value attention cache (existing behavior).

    Layout: [2, num_blocks, block_size, num_kv_heads, head_dim] (keys, values).
    """

    MAMBA_SSM = "mamba_ssm"
    """Mamba SSM recurrent state cache (not key-value).

    Layout: tuple of (h, A_log, D, conv_states, ssm_states).
    - conv_states: [num_layers, batch, inner_dim, conv_dim]
    - ssm_states: [num_layers, batch, inner_dim, state_dim]
    """

    SLIDING_WINDOW = "sliding_window"
    """Sliding window attention with limited context.

    Layout: same as ATTENTION but only the last `window_size` tokens are kept.
    Oldest entries are overwritten in a circular buffer pattern.
    """

    MLA = "mla"
    """Multi-head latent attention (DeepSeek style).

    Layout: [2, num_blocks, block_size, latent_dim] — compressed KV via
    low-rank projection. Much smaller than standard attention per token.
    """


# Fill default block sizes after enum is defined.
_DEFAULT_BLOCK_SIZES.update({
    CacheBlockType.ATTENTION: 64,
    CacheBlockType.MAMBA_SSM: 1,      # SSM state is per-slot, not per-token
    CacheBlockType.SLIDING_WINDOW: 64,
    CacheBlockType.MLA: 64,
})


# ── MambaSSMState ──────────────────────────────────────────────────────


@dataclass
class MambaSSMState:
    """Manages Mamba's recurrent state for one or more SSM layers.

    Mamba models use a state-space model (SSM) instead of attention for
    some or all layers. The SSM state consists of:

    - **A_log**: log of the state transition matrix (discretized).
    - **D**: skip connection parameter.
    - **conv_states**: convolution state per layer (short-range dependency).
    - **ssm_states**: SSM hidden state per layer (long-range dependency).

    This class supports:
    - Checkpoint/restore for preemption (serialize to bytes).
    - Optional compression (zlib, 8-bit/4-bit quantization).
    - Multi-layer state management.

    Attributes:
        num_layers: Number of SSM layers tracked.
        inner_dim: Mamba inner dimension (d_model * expand).
        state_dim: SSM state dimension (d_model * A_rank).
        conv_dim: Convolution kernel size (typically 4).
        dtype: MLX dtype for state tensors.
    """

    num_layers: int
    inner_dim: int
    state_dim: int
    conv_dim: int = 4
    dtype: mx.Dtype = mx.float16

    def __post_init__(self):
        self._conv_states: list[mx.array] = []
        self._ssm_states: list[mx.array] = []
        self._initialized = False
        self._checkpoint_version = 1

    def initialize(self, batch_size: int = 1) -> None:
        """Initialize SSM states to zeros.

        Args:
            batch_size: Number of parallel sequences.
        """
        self._conv_states = []
        self._ssm_states = []
        for _ in range(self.num_layers):
            self._conv_states.append(
                mx.zeros((batch_size, self.inner_dim, self.conv_dim), dtype=self.dtype)
            )
            self._ssm_states.append(
                mx.zeros((batch_size, self.inner_dim, self.state_dim), dtype=self.dtype)
            )
        self._initialized = True
        self._batch_size = batch_size

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def conv_states(self) -> list[mx.array]:
        """Convolution states per layer."""
        return self._conv_states

    @property
    def ssm_states(self) -> list[mx.array]:
        """SSM hidden states per layer."""
        return self._ssm_states

    def update_layer(
        self,
        layer_idx: int,
        conv_state: mx.array | None = None,
        ssm_state: mx.array | None = None,
    ) -> None:
        """Update state for a specific layer.

        Args:
            layer_idx: Layer index (0-based).
            conv_state: New convolution state (or None to keep existing).
            ssm_state: New SSM hidden state (or None to keep existing).

        Raises:
            IndexError: If layer_idx is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        if conv_state is not None:
            self._conv_states[layer_idx] = conv_state
        if ssm_state is not None:
            self._ssm_states[layer_idx] = ssm_state

    def get_layer(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        """Get (conv_state, ssm_state) for a specific layer.

        Args:
            layer_idx: Layer index (0-based).

        Returns:
            Tuple of (conv_state, ssm_state) arrays.

        Raises:
            IndexError: If layer_idx is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        return self._conv_states[layer_idx], self._ssm_states[layer_idx]

    def memory_bytes(self) -> int:
        """Estimate total memory usage in bytes."""
        total = 0
        for cs in self._conv_states:
            total += cs.size * cs.dtype.size
        for ss in self._ssm_states:
            total += ss.size * ss.dtype.size
        return total

    # ── Checkpoint / Restore ────────────────────────────────────────── 

    def checkpoint(self, compression: str = "none") -> bytes:
        """Serialize SSM state to bytes for preemption/persistence.

        The checkpoint format:
        - Header: version (uint16), num_layers (uint16), inner_dim (uint32),
                  state_dim (uint32), conv_dim (uint16), batch_size (uint16),
                  dtype code (uint8), compression code (uint8)
        - Per-layer data: conv_state + ssm_state tensors
        - Optional compression wrapper

        Args:
            compression: "none", "zlib", "8bit", or "4bit".

        Returns:
            Serialized state as bytes.
        """
        if not self._initialized:
            raise RuntimeError("Cannot checkpoint uninitialized MambaSSMState")

        comp_code = {
            "none": _COMPRESS_NONE,
            "zlib": _COMPRESS_ZLIB,
            "8bit": _COMPRESS_QUANT_8BIT,
            "4bit": _COMPRESS_QUANT_4BIT,
        }.get(compression, _COMPRESS_NONE)

        dtype_code = {
            mx.float16: 0,
            mx.float32: 1,
            mx.bfloat16: 2,
            mx.int8: 3,
            mx.uint8: 4,
        }.get(self.dtype, 0)

        # Pack header
        header = struct.pack(
            ">HHIIHHBB",
            self._checkpoint_version,  # version
            self.num_layers,
            self.inner_dim,
            self.state_dim,
            self.conv_dim,
            self._batch_size,
            dtype_code,
            comp_code,
        )

        buf = io.BytesIO()
        buf.write(header)

        for layer_idx in range(self.num_layers):
            conv, ssm = self.get_layer(layer_idx)
            if comp_code == _COMPRESS_QUANT_4BIT:
                self._write_tensor_4bit(buf, conv)
                self._write_tensor_4bit(buf, ssm)
            elif comp_code == _COMPRESS_QUANT_8BIT:
                self._write_tensor_8bit(buf, conv)
                self._write_tensor_8bit(buf, ssm)
            else:
                self._write_tensor_raw(buf, conv)
                self._write_tensor_raw(buf, ssm)

        raw_data = buf.getvalue()

        if comp_code == _COMPRESS_ZLIB:
            # Prefix with a single byte indicating zlib compression so
            # restore() can detect it before unpacking the header.
            raw_data = b"\x01" + zlib.compress(raw_data)

        return raw_data

    @classmethod
    def restore(cls, data: bytes) -> MambaSSMState:
        """Restore SSM state from checkpoint bytes.

        Args:
            data: Serialized state from checkpoint().

        Returns:
            Restored MambaSSMState instance.
        """
        header_size = struct.calcsize(">HHIIHHBB")

        # Check for zlib wrapper byte prefix
        if data and data[0] == 0x01:
            # Zlib-compressed: skip the 1-byte prefix, decompress, then parse
            data = zlib.decompress(data[1:])
            comp_code = _COMPRESS_ZLIB
        else:
            comp_code = _COMPRESS_NONE

        header = data[:header_size]
        (
            version,
            num_layers,
            inner_dim,
            state_dim,
            conv_dim,
            batch_size,
            dtype_code,
            stored_comp_code,
        ) = struct.unpack(">HHIIHHBB", header)

        # Override comp_code from wrapper byte (the stored value inside the
        # header may differ for non-zlib modes; trust the wrapper).
        if comp_code == _COMPRESS_NONE and stored_comp_code in (
            _COMPRESS_QUANT_8BIT, _COMPRESS_QUANT_4BIT, _COMPRESS_NONE,
        ):
            comp_code = stored_comp_code

        dtype_map = {
            0: mx.float16,
            1: mx.float32,
            2: mx.bfloat16,
            3: mx.int8,
            4: mx.uint8,
        }
        dtype = dtype_map.get(dtype_code, mx.float16)

        state = cls(
            num_layers=num_layers,
            inner_dim=inner_dim,
            state_dim=state_dim,
            conv_dim=conv_dim,
            dtype=dtype,
        )
        state.initialize(batch_size)

        offset = header_size
        for layer_idx in range(num_layers):
            if comp_code == _COMPRESS_QUANT_4BIT:
                conv, offset = cls._read_tensor_4bit(data, offset, dtype)
                ssm, offset = cls._read_tensor_4bit(data, offset, dtype)
            elif comp_code == _COMPRESS_QUANT_8BIT:
                conv, offset = cls._read_tensor_8bit(data, offset, dtype)
                ssm, offset = cls._read_tensor_8bit(data, offset, dtype)
            else:
                conv, offset = cls._read_tensor_raw(data, offset, dtype)
                ssm, offset = cls._read_tensor_raw(data, offset, dtype)
            state.update_layer(layer_idx, conv, ssm)

        return state

    # ── Tensor Serialization Helpers ──────────────────────────────────

    @staticmethod
    def _write_tensor_raw(buf: io.BytesIO, tensor: mx.array) -> None:
        """Write a tensor to buffer with shape header (raw float16/float32)."""
        import numpy as np

        np_arr = np.array(tensor)
        shape = np_arr.shape
        # shape header: ndim (uint16) + shape values (each uint32)
        buf.write(struct.pack(">H", len(shape)))
        for dim in shape:
            buf.write(struct.pack(">I", dim))
        # Data
        buf.write(np_arr.tobytes())

    @staticmethod
    def _read_tensor_raw(data: bytes, offset: int, dtype: mx.Dtype) -> tuple[mx.array, int]:
        """Read a tensor from bytes at offset."""
        import numpy as np

        ndim = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        shape = []
        for _ in range(ndim):
            dim = struct.unpack_from(">I", data, offset)[0]
            offset += 4
            shape.append(dim)

        num_elements = 1
        for d in shape:
            num_elements *= d

        np_dtype = {mx.float16: np.float16, mx.float32: np.float32}.get(dtype, np.float16)
        nbytes = num_elements * np_dtype().itemsize
        np_arr = np.frombuffer(data[offset : offset + nbytes], dtype=np_dtype).reshape(shape)
        offset += nbytes

        return mx.array(np_arr), offset

    @staticmethod
    def _write_tensor_8bit(buf: io.BytesIO, tensor: mx.array) -> None:
        """Write tensor as 8-bit quantized (scale + int8 data)."""
        import numpy as np

        np_arr = np.array(tensor).astype(np.float32)
        scale = np.abs(np_arr).max()
        if scale == 0:
            scale = 1.0
        quantized = np.clip(np.round(np_arr / scale * 127), -128, 127).astype(np.int8)

        shape = np_arr.shape
        buf.write(struct.pack(">H", len(shape)))
        for dim in shape:
            buf.write(struct.pack(">I", dim))
        buf.write(struct.pack(">f", scale))
        buf.write(quantized.tobytes())

    @staticmethod
    def _read_tensor_8bit(data: bytes, offset: int, dtype: mx.Dtype) -> tuple[mx.array, int]:
        """Read 8-bit quantized tensor."""
        import numpy as np

        ndim = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        shape = []
        for _ in range(ndim):
            dim = struct.unpack_from(">I", data, offset)[0]
            offset += 4
            shape.append(dim)

        scale = struct.unpack_from(">f", data, offset)[0]
        offset += 4

        num_elements = 1
        for d in shape:
            num_elements *= d

        nbytes = num_elements  # int8 = 1 byte each
        quantized = np.frombuffer(data[offset : offset + nbytes], dtype=np.int8).reshape(shape)
        offset += nbytes

        dequantized = (quantized.astype(np.float32) * scale / 127.0)
        target_dtype = {mx.float16: np.float16, mx.float32: np.float32}.get(dtype, np.float16)
        return mx.array(dequantized.astype(target_dtype)), offset

    @staticmethod
    def _write_tensor_4bit(buf: io.BytesIO, tensor: mx.array) -> None:
        """Write tensor as 4-bit quantized (scale + packed int4 data)."""
        import numpy as np

        np_arr = np.array(tensor).astype(np.float32)
        scale = np.abs(np_arr).max()
        if scale == 0:
            scale = 1.0
        quantized = np.clip(np.round(np_arr / scale * 7), -7, 7).astype(np.int8)

        shape = np_arr.shape
        buf.write(struct.pack(">H", len(shape)))
        for dim in shape:
            buf.write(struct.pack(">I", dim))
        buf.write(struct.pack(">f", scale))

        # Pack two int4 values per byte
        flat = quantized.flatten()
        # Shift to unsigned: [-7,7] -> [0,14] using +7 offset
        flat = (flat + 7).astype(np.uint8)
        packed = []
        for i in range(0, len(flat), 2):
            hi = flat[i] & 0x0F
            lo = flat[i + 1] & 0x0F if i + 1 < len(flat) else 0
            packed.append((hi << 4) | lo)
        buf.write(bytes(packed))

    @staticmethod
    def _read_tensor_4bit(data: bytes, offset: int, dtype: mx.Dtype) -> tuple[mx.array, int]:
        """Read 4-bit quantized tensor."""
        import numpy as np

        ndim = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        shape = []
        for _ in range(ndim):
            dim = struct.unpack_from(">I", data, offset)[0]
            offset += 4
            shape.append(dim)

        scale = struct.unpack_from(">f", data, offset)[0]
        offset += 4

        num_elements = 1
        for d in shape:
            num_elements *= d

        packed_bytes = (num_elements + 1) // 2
        packed = data[offset : offset + packed_bytes]
        offset += packed_bytes

        # Unpack
        flat = []
        for byte_val in packed:
            hi = ((byte_val >> 4) & 0x0F) - 7  # Undo unsigned offset
            lo = (byte_val & 0x0F) - 7
            flat.extend([hi, lo])
        flat = flat[:num_elements]

        dequantized = (np.array(flat, dtype=np.float32) * scale / 7.0)
        target_dtype = {mx.float16: np.float16, mx.float32: np.float32}.get(dtype, np.float16)
        return mx.array(dequantized.reshape(shape).astype(target_dtype)), offset


# ── CachePoolStats ─────────────────────────────────────────────────────


@dataclass
class CachePoolStats:
    """Statistics for a single cache pool (one block type)."""

    block_type: CacheBlockType
    blocks_total: int = 0
    blocks_used: int = 0
    block_size: int = 0
    memory_bytes: int = 0
    num_layers: int = 0
    allocations: int = 0
    frees: int = 0

    @property
    def blocks_free(self) -> int:
        return self.blocks_total - self.blocks_used

    @property
    def utilization(self) -> float:
        if self.blocks_total == 0:
            return 0.0
        return self.blocks_used / self.blocks_total


# ── CachePool ──────────────────────────────────────────────────────────


class _CachePool:
    """Pool of cache blocks for a specific CacheBlockType.

    Each pool manages its own block allocator with a fixed block size
    and optional eviction policy. Blocks are tracked as free/used lists.

    Attributes:
        block_type: The cache type this pool manages.
        block_size: Tokens per block (or state slots for SSM).
        total_blocks: Maximum number of blocks.
        cache_shape: Shape of a single block's cache tensor.
    """

    def __init__(
        self,
        block_type: CacheBlockType,
        block_size: int,
        total_blocks: int,
        cache_shape: tuple[int, ...] = (),
    ) -> None:
        self.block_type = block_type
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.cache_shape = cache_shape
        self._free_blocks: list[int] = list(range(total_blocks))
        self._used_blocks: set[int] = set()
        self._layer_indices: set[int] = set()
        self._allocations = 0
        self._frees = 0
        # Per-block timestamps for LRU eviction
        self._last_access: dict[int, float] = {}

    def register_layer(self, layer_idx: int) -> None:
        """Register a layer index as managed by this pool."""
        self._layer_indices.add(layer_idx)

    @property
    def num_layers(self) -> int:
        return len(self._layer_indices)

    @property
    def blocks_free(self) -> int:
        return len(self._free_blocks)

    @property
    def blocks_used(self) -> int:
        return len(self._used_blocks)

    def allocate(self, num_blocks: int) -> list[int]:
        """Allocate blocks from the pool.

        Args:
            num_blocks: Number of blocks to allocate.

        Returns:
            List of block indices.

        Raises:
            MemoryError: If not enough free blocks.
        """
        if num_blocks > len(self._free_blocks):
            raise MemoryError(
                f"Not enough free blocks in {self.block_type.value} pool: "
                f"requested {num_blocks}, available {len(self._free_blocks)}"
            )
        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_blocks.pop(0)
            self._used_blocks.add(block_id)
            self._last_access[block_id] = time.monotonic()
            allocated.append(block_id)
        self._allocations += num_blocks
        return allocated

    def free(self, blocks: list[int]) -> int:
        """Free blocks back to the pool.

        Args:
            blocks: Block indices to free.

        Returns:
            Number of blocks actually freed (ignores already-free blocks).
        """
        freed = 0
        for block_id in blocks:
            if block_id in self._used_blocks:
                self._used_blocks.discard(block_id)
                self._free_blocks.append(block_id)
                self._last_access.pop(block_id, None)
                freed += 1
        self._frees += freed
        return freed

    def evict_lru(self, num_blocks: int) -> list[int]:
        """Evict the least-recently-used blocks.

        Args:
            num_blocks: Number of blocks to evict.

        Returns:
            List of evicted block indices.
        """
        if not self._used_blocks:
            return []

        # Sort used blocks by last access time
        sorted_blocks = sorted(
            self._used_blocks,
            key=lambda b: self._last_access.get(b, 0.0),
        )
        to_evict = sorted_blocks[:num_blocks]
        self.free(to_evict)
        return list(to_evict)

    def get_stats(self) -> CachePoolStats:
        """Get pool statistics."""
        element_size = 2  # float16
        elements_per_block = 1
        for dim in self.cache_shape:
            elements_per_block *= dim
        memory_per_block = elements_per_block * element_size

        return CachePoolStats(
            block_type=self.block_type,
            blocks_total=self.total_blocks,
            blocks_used=self.blocks_used,
            block_size=self.block_size,
            memory_bytes=self.blocks_used * memory_per_block,
            num_layers=self.num_layers,
            allocations=self._allocations,
            frees=self._frees,
        )


# ── HybridKVCache ──────────────────────────────────────────────────────


class HybridKVCache:
    """Manages multiple cache pools for hybrid models with mixed layer types.

    Hybrid models (Jamba, Zamba, etc.) interleave attention layers with
    Mamba SSM layers. Each layer type requires a different cache format.
    This class:

    1. Maintains a separate pool per CacheBlockType.
    2. Routes layer-specific allocate/free/lookup to the correct pool.
    3. Provides per-type and aggregate statistics.
    4. Supports preemption-friendly checkpoint/restore for SSM states.

    Usage::

        cache = HybridKVCache(max_blocks=1024)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        blocks = cache.allocate(0, 4)  # Allocates from ATTENTION pool
        ssm_blocks = cache.allocate(1, 2)  # Allocates from MAMBA_SSM pool
        cache.free(0, blocks)
    """

    def __init__(
        self,
        max_blocks: int = 1024,
        default_block_size: int = 64,
    ) -> None:
        """Initialize the hybrid cache manager.

        Args:
            max_blocks: Default max blocks per pool (can be overridden per type).
            default_block_size: Default tokens per block.
        """
        self._max_blocks = max_blocks
        self._default_block_size = default_block_size
        # layer_idx → CacheBlockType
        self._layer_types: dict[int, CacheBlockType] = {}
        # CacheBlockType → _CachePool
        self._pools: dict[CacheBlockType, _CachePool] = {}
        # Per-block-size overrides: CacheBlockType → int
        self._block_size_overrides: dict[CacheBlockType, int] = {}
        # Per-max-blocks overrides: CacheBlockType → int
        self._max_blocks_overrides: dict[CacheBlockType, int] = {}
        # Per-layer cache data: layer_idx → Any
        self._layer_caches: dict[int, Any] = {}
        # Mamba SSM states: layer_idx → MambaSSMState
        self._ssm_states: dict[int, MambaSSMState] = {}

    def register_layer(
        self,
        layer_idx: int,
        block_type: CacheBlockType,
        cache_shape: tuple[int, ...],
        block_size: int | None = None,
        max_blocks: int | None = None,
    ) -> None:
        """Register a layer's cache type and shape.

        This creates the pool for the block type if it doesn't exist,
        and maps the layer index to the correct pool.

        Args:
            layer_idx: Layer index (0-based).
            block_type: Cache format for this layer.
            cache_shape: Shape of a single cache block tensor.
            block_size: Override block size for this type (or None for default).
            max_blocks: Override max blocks for this pool (or None for default).
        """
        self._layer_types[layer_idx] = block_type

        if block_type not in self._pools:
            bs = block_size or self._block_size_overrides.get(
                block_type,
                _DEFAULT_BLOCK_SIZES.get(block_type, self._default_block_size),
            )
            mb = max_blocks or self._max_blocks_overrides.get(
                block_type, self._max_blocks
            )
            self._pools[block_type] = _CachePool(
                block_type=block_type,
                block_size=bs,
                total_blocks=mb,
                cache_shape=cache_shape,
            )

        self._pools[block_type].register_layer(layer_idx)

    def allocate(self, layer_idx: int, num_blocks: int) -> list[int]:
        """Allocate cache blocks for a specific layer.

        Routes to the correct pool based on the layer's registered type.

        Args:
            layer_idx: Layer index (must be registered).
            num_blocks: Number of blocks to allocate.

        Returns:
            List of block indices.

        Raises:
            KeyError: If layer_idx is not registered.
            MemoryError: If the pool is out of blocks.
        """
        block_type = self._get_layer_type(layer_idx)
        pool = self._pools[block_type]
        return pool.allocate(num_blocks)

    def free(self, layer_idx: int, blocks: list[int]) -> int:
        """Free cache blocks for a specific layer.

        Routes to the correct pool based on the layer's registered type.

        Args:
            layer_idx: Layer index (must be registered).
            blocks: Block indices to free.

        Returns:
            Number of blocks actually freed.

        Raises:
            KeyError: If layer_idx is not registered.
        """
        block_type = self._get_layer_type(layer_idx)
        pool = self._pools[block_type]
        return pool.free(blocks)

    def get_cache(self, layer_idx: int) -> Any:
        """Get the cache data for a specific layer.

        For SSM layers, returns the MambaSSMState.
        For attention layers, returns the stored cache data.

        Args:
            layer_idx: Layer index.

        Returns:
            Cache data for the layer, or None if not set.
        """
        block_type = self._get_layer_type(layer_idx)
        if block_type == CacheBlockType.MAMBA_SSM:
            return self._ssm_states.get(layer_idx)
        return self._layer_caches.get(layer_idx)

    def set_cache(self, layer_idx: int, cache_data: Any) -> None:
        """Set the cache data for a specific layer.

        Args:
            layer_idx: Layer index (must be registered).
            cache_data: Cache data to store.
        """
        block_type = self._get_layer_type(layer_idx)
        if block_type == CacheBlockType.MAMBA_SSM:
            if isinstance(cache_data, MambaSSMState):
                self._ssm_states[layer_idx] = cache_data
            else:
                raise TypeError(
                    f"MAMBA_SSM layers require MambaSSMState, got {type(cache_data)}"
                )
        else:
            self._layer_caches[layer_idx] = cache_data

    def get_layer_type(self, layer_idx: int) -> CacheBlockType | None:
        """Get the cache block type for a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            CacheBlockType or None if not registered.
        """
        return self._layer_types.get(layer_idx)

    def _get_layer_type(self, layer_idx: int) -> CacheBlockType:
        """Get the cache block type for a layer, raising if not registered."""
        if layer_idx not in self._layer_types:
            raise KeyError(
                f"Layer {layer_idx} not registered. "
                f"Registered layers: {sorted(self._layer_types.keys())}"
            )
        return self._layer_types[layer_idx]

    def get_pool(self, block_type: CacheBlockType) -> _CachePool | None:
        """Get the cache pool for a specific block type.

        Args:
            block_type: The cache type.

        Returns:
            The _CachePool, or None if no layers of this type are registered.
        """
        return self._pools.get(block_type)

    def get_stats(self) -> dict[str, Any]:
        """Get per-type and aggregate cache statistics.

        Returns:
            Dict with keys:
            - "pools": dict mapping block_type name to CachePoolStats dict
            - "total_blocks_used": int
            - "total_blocks_total": int
            - "total_memory_bytes": int
            - "registered_layers": int
            - "layer_types": dict mapping layer_idx to block_type name
        """
        pool_stats = {}
        total_used = 0
        total_total = 0
        total_memory = 0

        for bt, pool in self._pools.items():
            stats = pool.get_stats()
            pool_stats[bt.value] = {
                "block_type": bt.value,
                "blocks_total": stats.blocks_total,
                "blocks_used": stats.blocks_used,
                "blocks_free": stats.blocks_free,
                "block_size": stats.block_size,
                "memory_bytes": stats.memory_bytes,
                "num_layers": stats.num_layers,
                "utilization": round(stats.utilization, 4),
                "allocations": stats.allocations,
                "frees": stats.frees,
            }
            total_used += stats.blocks_used
            total_total += stats.blocks_total
            total_memory += stats.memory_bytes

        return {
            "pools": pool_stats,
            "total_blocks_used": total_used,
            "total_blocks_total": total_total,
            "total_memory_bytes": total_memory,
            "registered_layers": len(self._layer_types),
            "layer_types": {
                idx: bt.value for idx, bt in sorted(self._layer_types.items())
            },
        }

    def clear(self) -> None:
        """Clear all cache data and reset pools."""
        for pool in self._pools.values():
            pool.free(list(pool._used_blocks))
        self._layer_caches.clear()
        self._ssm_states.clear()

    @property
    def num_layers(self) -> int:
        """Total number of registered layers."""
        return len(self._layer_types)

    @property
    def num_pools(self) -> int:
        """Number of distinct cache pools."""
        return len(self._pools)


# ── BlockAlignedCacheSplitter ──────────────────────────────────────────


@dataclass
class LayerGroup:
    """A group of layers that must be evicted together.

    In hybrid models, some layers share cached state (e.g., attention
    layers sharing the same prefix cache). This class defines a group
    of layer indices that must be kept or evicted atomically.
    """

    group_id: int
    layer_indices: list[int] = field(default_factory=list)
    block_type: CacheBlockType = CacheBlockType.ATTENTION

    @property
    def size(self) -> int:
        return len(self.layer_indices)


class BlockAlignedCacheSplitter:
    """Ensures KV blocks are split at layer boundaries for hybrid models.

    When evicting cache blocks from a hybrid model, it is critical that
    eviction respects layer group boundaries. Partial eviction of a layer
    group leads to inconsistent state (some layers cached, others not).

    This class:
    1. Groups layers by their cache block type.
    2. Defines eviction boundaries at layer group edges.
    3. Validates that eviction operations don't split layer groups.
    4. Supports custom layer grouping (e.g., attention+SSM paired groups).

    Usage::

        splitter = BlockAlignedCacheSplitter(hybrid_cache)
        splitter.define_group(0, [0, 2, 4], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1, 3, 5], CacheBlockType.MAMBA_SSM)

        # Validate eviction is safe
        safe = splitter.is_safe_eviction([0, 2, 4])
        boundary = splitter.next_eviction_boundary(start_block=10)
    """

    def __init__(self, hybrid_cache: HybridKVCache) -> None:
        self._cache = hybrid_cache
        self._groups: dict[int, LayerGroup] = {}
        # layer_idx → group_id (reverse mapping)
        self._layer_to_group: dict[int, int] = {}
        self._next_group_id = 0

    def define_group(
        self,
        group_id: int,
        layer_indices: list[int],
        block_type: CacheBlockType,
    ) -> None:
        """Define a layer group for aligned eviction.

        Args:
            group_id: Unique group identifier.
            layer_indices: Layer indices in this group.
            block_type: Cache type for this group.
        """
        group = LayerGroup(
            group_id=group_id,
            layer_indices=list(layer_indices),
            block_type=block_type,
        )
        self._groups[group_id] = group
        for idx in layer_indices:
            self._layer_to_group[idx] = group_id
        self._next_group_id = max(self._next_group_id, group_id + 1)

    def auto_group_by_type(self) -> list[LayerGroup]:
        """Automatically create groups based on contiguous same-type layers.

        Walks the registered layers and creates groups wherever the
        cache block type changes. Returns the created groups.

        Returns:
            List of LayerGroup instances created.
        """
        sorted_layers = sorted(self._cache._layer_types.keys())
        if not sorted_layers:
            return []

        groups = []
        current_type = self._cache._layer_types[sorted_layers[0]]
        current_layers = [sorted_layers[0]]

        for li in sorted_layers[1:]:
            lt = self._cache._layer_types[li]
            if lt != current_type:
                # Flush current group
                gid = self._next_group_id
                self.define_group(gid, current_layers, current_type)
                groups.append(self._groups[gid])
                current_type = lt
                current_layers = [li]
            else:
                current_layers.append(li)

        # Flush last group
        gid = self._next_group_id
        self.define_group(gid, current_layers, current_type)
        groups.append(self._groups[gid])

        return groups

    def get_group(self, group_id: int) -> LayerGroup | None:
        """Get a layer group by ID."""
        return self._groups.get(group_id)

    def get_group_for_layer(self, layer_idx: int) -> LayerGroup | None:
        """Get the layer group containing a specific layer."""
        gid = self._layer_to_group.get(layer_idx)
        if gid is None:
            return None
        return self._groups.get(gid)

    def is_safe_eviction(self, layer_indices: list[int]) -> bool:
        """Check if evicting a set of layers is safe (doesn't split groups).

        Eviction is safe when either:
        - All layers in each affected group are included, or
        - No layers from a group are included.

        Args:
            layer_indices: Layers to be evicted.

        Returns:
            True if eviction is safe.
        """
        affected_groups: set[int] = set()
        for li in layer_indices:
            gid = self._layer_to_group.get(li)
            if gid is not None:
                affected_groups.add(gid)

        for gid in affected_groups:
            group = self._groups[gid]
            evicted_from_group = sum(
                1 for li in group.layer_indices if li in set(layer_indices)
            )
            # Partial eviction is unsafe
            if evicted_from_group > 0 and evicted_from_group < group.size:
                return False

        return True

    def get_complete_eviction_set(self, layer_indices: list[int]) -> list[int]:
        """Expand a partial eviction set to be group-complete.

        Given a set of layers to evict, expands it to include all layers
        from any partially-included group. This ensures eviction is always
        safe.

        Args:
            layer_indices: Partial set of layers to evict.

        Returns:
            Expanded set including all layers from affected groups.
        """
        result = set(layer_indices)
        for li in list(layer_indices):
            gid = self._layer_to_group.get(li)
            if gid is not None:
                group = self._groups[gid]
                result.update(group.layer_indices)
        return sorted(result)

    def next_eviction_boundary(self, start_block: int) -> int:
        """Find the next block-aligned eviction boundary after a position.

        For hybrid caches, this returns the block position that aligns
        with the start of the next layer group, ensuring that eviction
        doesn't split a group mid-way.

        Args:
            start_block: Starting block position.

        Returns:
            The block position of the next safe eviction boundary.
        """
        # Map block positions to layer groups
        sorted_layers = sorted(self._cache._layer_types.keys())
        if not sorted_layers:
            return start_block

        # Estimate blocks per layer group
        pos = 0
        for li in sorted_layers:
            bt = self._cache._layer_types[li]
            pool = self._cache._pools.get(bt)
            if pool is None:
                continue
            group_blocks = pool.blocks_used
            if pos + group_blocks > start_block:
                return pos + group_blocks
            pos += group_blocks

        return pos

    @property
    def num_groups(self) -> int:
        """Number of registered layer groups."""
        return len(self._groups)

    def get_stats(self) -> dict[str, Any]:
        """Get splitter statistics."""
        return {
            "num_groups": self.num_groups,
            "groups": {
                gid: {
                    "block_type": group.block_type.value,
                    "layer_indices": group.layer_indices,
                    "size": group.size,
                }
                for gid, group in self._groups.items()
            },
        }
