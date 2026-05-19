from __future__ import annotations
"""Yunshu KV Cache Serialization — persist and transfer KV cache state.

Binary format:
  MAGIC (4B) | VERSION (2B) | HEADER (variable) | BLOCK DATA (variable)

Supports:
- Per-block serialize/deserialize (for cross-node transfer)
- Full table serialize/deserialize (for snapshot/restore)
- File save/load (for warm-start from disk)
- numpy conversion layer (MLX arrays <-> bytes via numpy bridge)

Compression modes:
  "none"    — raw bytes
  "numpy"   — numpy.save / numpy.load (uses zlib internally)
  "safetensors" — reserved for future use
"""


import io
import struct
from dataclasses import dataclass

import numpy as np

from .block import KVBlock
from .block_table import BlockTable

# Binary format constants
MAGIC: int = 0x5953_4B56  # "YSKV" in big-endian
VERSION: int = 1

# Header field: uint32 magic, uint16 version
_FILE_HEADER_FMT = ">IH"  # big-endian: 4B magic + 2B version
_FILE_HEADER_SIZE = struct.calcsize(_FILE_HEADER_FMT)

# Per-block header:
#   block_id (I), block_hash (Q, 0=None), ref_count (I), block_size (I),
#   ndim_key (B), shape_key (ndim * I), ndim_val (B), shape_val (ndim * I),
#   dtype_key_len (B), dtype_key_str, dtype_val_len (B), dtype_val_str,
#   key_nbytes (I), value_nbytes (I)
_BLOCK_META_PREFIX = ">IQII"  # block_id, block_hash, ref_count, block_size
_BLOCK_META_PREFIX_SIZE = struct.calcsize(_BLOCK_META_PREFIX)


def _mx_to_numpy(arr) -> np.ndarray:
    """Convert MLX array to numpy array."""
    # Support both real MLX arrays and plain numpy arrays (for testing)
    if isinstance(arr, np.ndarray):
        return arr
    # MLX -> numpy
    return np.array(arr)


def _numpy_to_mx(np_arr: np.ndarray):
    """Convert numpy array back to MLX array (or keep as numpy in pure-test)."""
    try:
        import mlx.core as mx
        return mx.array(np_arr)
    except ImportError:
        return np_arr


def _dtype_to_str(dtype) -> str:
    """Convert numpy dtype to a serializable string."""
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, np.dtype):
        s = dtype.str
        return s if isinstance(s, str) else s.decode("ascii")
    return str(dtype)


def _str_to_dtype(s: str) -> np.dtype:
    """Convert serialized dtype string back to numpy dtype."""
    return np.dtype(s)


def _encode_shape(ndim: int, shape: tuple[int, ...]) -> bytes:
    """Encode array shape as bytes."""
    return struct.pack(f">{ndim}I", *shape)


def _decode_shape(data: bytes, offset: int, ndim: int) -> tuple[tuple[int, ...], int]:
    """Decode array shape from bytes. Returns (shape, bytes_consumed)."""
    size = ndim * 4
    shape = struct.unpack_from(f">{ndim}I", data, offset)
    return tuple(shape), size


def _compress_raw(raw_bytes: bytes) -> bytes:
    """No compression passthrough."""
    return raw_bytes


def _decompress_raw(compressed: bytes) -> bytes:
    return compressed


def _compress_numpy(raw_bytes: bytes) -> bytes:
    """Compress using numpy's built-in zlib via savez_compressed to a buffer."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, data=arr)
    return buf.getvalue()


def _decompress_numpy(compressed: bytes) -> bytes:
    """Decompress numpy compressed data."""
    buf = io.BytesIO(compressed)
    loaded = np.load(buf)
    try:
        return loaded["data"].tobytes()
    finally:
        loaded.close()


# Dispatch tables
_COMPRESS = {
    "none": _compress_raw,
    "numpy": _compress_numpy,
    "safetensors": _compress_raw,  # safetensors handles framing at block level
}
_DECOMPRESS = {
    "none": _decompress_raw,
    "numpy": _decompress_numpy,
    "safetensors": _decompress_raw,  # safetensors handles framing at block level
}


@dataclass
class SerializedBlock:
    """Holds deserialized block metadata + tensor data."""
    block: KVBlock
    key_data: np.ndarray
    value_data: np.ndarray


class KVCacheSerializer:
    """Serialize and deserialize KV cache blocks and tables.

    Args:
        compression: "none" for raw bytes, "numpy" for zlib compression.
    """

    def __init__(self, compression: str = "none") -> None:
        if compression not in ("none", "numpy", "safetensors"):
            raise ValueError(f"Unsupported compression mode: {compression!r}")
        self.compression = compression

    # ── Single block ──────────────────────────────────────────────

    def serialize_block(
        self,
        block: KVBlock,
        key_data,
        value_data,
    ) -> bytes:
        """Serialize a single block's key and value tensors.

        Args:
            block: KVBlock metadata.
            key_data: Key tensor (MLX or numpy array).
            value_data: Value tensor (MLX or numpy array).

        Returns:
            bytes containing block metadata + tensor data.
        """
        key_np = _mx_to_numpy(key_data)
        val_np = _mx_to_numpy(value_data)

        key_dtype_str = _dtype_to_str(key_np.dtype)
        val_dtype_str = _dtype_to_str(val_np.dtype)

        block_hash = block.block_hash if block.block_hash is not None else 0
        block_hash_flag = 1 if block.block_hash is not None else 0

        # Build metadata
        parts = []
        # Fixed prefix
        parts.append(struct.pack(_BLOCK_META_PREFIX, block.block_id, block_hash, block.ref_count, block.block_size if hasattr(block, 'block_size') else 0))
        # Hash presence flag
        parts.append(struct.pack(">B", block_hash_flag))
        # Key array info
        parts.append(struct.pack(">B", key_np.ndim))
        parts.append(_encode_shape(key_np.ndim, key_np.shape))
        parts.append(struct.pack(">B", len(key_dtype_str)))
        parts.append(key_dtype_str.encode("ascii"))
        # Value array info
        parts.append(struct.pack(">B", val_np.ndim))
        parts.append(_encode_shape(val_np.ndim, val_np.shape))
        parts.append(struct.pack(">B", len(val_dtype_str)))
        parts.append(val_dtype_str.encode("ascii"))
        # Data sizes
        key_bytes = key_np.tobytes()
        val_bytes = val_np.tobytes()
        parts.append(struct.pack(">II", len(key_bytes), len(val_bytes)))

        # Raw data — always use the same flat layout so deserialize_block can
        # parse it without knowing the compression mode.  (Safetensors framing
        # is handled at the table level by the compress/decompress dispatch.)
        parts.append(key_bytes)
        parts.append(val_bytes)

        return b"".join(parts)

    def deserialize_block(self, data: bytes) -> tuple[KVBlock, object, object]:
        """Deserialize a single block from bytes.

        Returns:
            (KVBlock, key_array, value_array) — arrays are numpy or MLX depending on availability.
        """
        offset = 0

        # Fixed prefix
        block_id, block_hash_raw, ref_count, _block_size = struct.unpack_from(
            _BLOCK_META_PREFIX, data, offset
        )
        offset += _BLOCK_META_PREFIX_SIZE

        # Hash presence flag
        (hash_flag,) = struct.unpack_from(">B", data, offset)
        offset += 1
        block_hash = block_hash_raw if hash_flag else None

        # Key array info
        (key_ndim,) = struct.unpack_from(">B", data, offset)
        offset += 1
        key_shape, consumed = _decode_shape(data, offset, key_ndim)
        offset += consumed
        (key_dtype_len,) = struct.unpack_from(">B", data, offset)
        offset += 1
        key_dtype_str = data[offset : offset + key_dtype_len].decode("ascii")
        offset += key_dtype_len
        key_dtype = _str_to_dtype(key_dtype_str)

        # Value array info
        (val_ndim,) = struct.unpack_from(">B", data, offset)
        offset += 1
        val_shape, consumed = _decode_shape(data, offset, val_ndim)
        offset += consumed
        (val_dtype_len,) = struct.unpack_from(">B", data, offset)
        offset += 1
        val_dtype_str = data[offset : offset + val_dtype_len].decode("ascii")
        offset += val_dtype_len
        val_dtype = _str_to_dtype(val_dtype_str)

        # Data sizes
        (key_nbytes, val_nbytes) = struct.unpack_from(">II", data, offset)
        offset += 8

        # Read array data
        key_raw = data[offset : offset + key_nbytes]
        offset += key_nbytes
        val_raw = data[offset : offset + val_nbytes]
        offset += val_nbytes

        key_np = np.frombuffer(key_raw, dtype=key_dtype).reshape(key_shape).copy()
        val_np = np.frombuffer(val_raw, dtype=val_dtype).reshape(val_shape).copy()

        # Reconstruct KVBlock
        block = KVBlock(block_id=block_id, ref_count=ref_count, block_hash=block_hash)

        # Convert to MLX if available
        key_out = _numpy_to_mx(key_np)
        val_out = _numpy_to_mx(val_np)

        return block, key_out, val_out

    # ── Full table ────────────────────────────────────────────────

    def serialize_table(
        self,
        table: BlockTable,
        key_cache,
        value_cache,
    ) -> bytes:
        """Serialize entire block table with all referenced blocks.

        Args:
            table: BlockTable with block references.
            key_cache: Key cache tensor (MLX or numpy) shaped [num_blocks, ...].
            value_cache: Value cache tensor (MLX or numpy) shaped [num_blocks, ...].

        Returns:
            bytes with file header + table header + per-block data.
        """
        key_np = _mx_to_numpy(key_cache)
        val_np = _mx_to_numpy(value_cache)

        blocks = table.get_blocks()
        num_blocks = len(blocks)
        block_size = table.block_size

        # File header
        file_header = struct.pack(_FILE_HEADER_FMT, MAGIC, VERSION)

        # Table header:
        #   num_blocks (I), block_size (I),
        #   key_dtype_len (B), key_dtype_str,
        #   val_dtype_len (B), val_dtype_str,
        #   key_ndim (B), key_shape, val_ndim (B), val_shape
        key_dtype_str = _dtype_to_str(key_np.dtype)
        val_dtype_str = _dtype_to_str(val_np.dtype)

        table_header_parts = [
            struct.pack(">II", num_blocks, block_size),
            struct.pack(">B", len(key_dtype_str)),
            key_dtype_str.encode("ascii"),
            struct.pack(">B", len(val_dtype_str)),
            val_dtype_str.encode("ascii"),
            struct.pack(">B", key_np.ndim),
            _encode_shape(key_np.ndim, key_np.shape),
            struct.pack(">B", val_np.ndim),
            _encode_shape(val_np.ndim, val_np.shape),
        ]
        table_header = b"".join(table_header_parts)

        # Per-block data: block metadata + slice of key/value arrays
        block_data_parts = []
        for block in blocks:
            bid = block.block_id
            if bid < key_np.shape[0]:
                k_slice = key_np[bid]
                v_slice = val_np[bid]
            else:
                # Block not in cache tensor — write zeros
                k_slice = np.zeros(key_np.shape[1:], dtype=key_np.dtype)
                v_slice = np.zeros(val_np.shape[1:], dtype=val_np.dtype)

            block_bytes = self.serialize_block(block, k_slice, v_slice)
            # Prepend block data length for framing
            block_data_parts.append(struct.pack(">I", len(block_bytes)))
            block_data_parts.append(block_bytes)

        # Combine with optional compression on the body
        body = table_header + b"".join(block_data_parts)
        compressed_body = _COMPRESS[self.compression](body)

        # Final: file_header + compressed_body_length + compressed_body
        return file_header + struct.pack(">I", len(compressed_body)) + compressed_body

    def deserialize_table(self, data: bytes) -> tuple[BlockTable, object, object]:
        """Deserialize a block table from bytes.

        Returns:
            (BlockTable, key_array, value_array)
        """
        offset = 0

        # File header
        magic, version = struct.unpack_from(_FILE_HEADER_FMT, data, offset)
        offset += _FILE_HEADER_SIZE
        if magic != MAGIC:
            raise ValueError(f"Invalid magic number: 0x{magic:08X}, expected 0x{MAGIC:08X}")
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}, expected {VERSION}")

        # Compressed body length
        (body_len,) = struct.unpack_from(">I", data, offset)
        offset += 4

        # Decompress body
        compressed_body = data[offset : offset + body_len]
        body = _DECOMPRESS[self.compression](compressed_body)
        boff = 0  # body offset

        # Table header
        num_blocks, block_size = struct.unpack_from(">II", body, boff)
        boff += 8

        (key_dtype_len,) = struct.unpack_from(">B", body, boff)
        boff += 1
        key_dtype_str = body[boff : boff + key_dtype_len].decode("ascii")
        boff += key_dtype_len
        key_dtype = _str_to_dtype(key_dtype_str)

        (val_dtype_len,) = struct.unpack_from(">B", body, boff)
        boff += 1
        val_dtype_str = body[boff : boff + val_dtype_len].decode("ascii")
        boff += val_dtype_len
        val_dtype = _str_to_dtype(val_dtype_str)

        (key_ndim,) = struct.unpack_from(">B", body, boff)
        boff += 1
        key_shape, consumed = _decode_shape(body, boff, key_ndim)
        boff += consumed

        (val_ndim,) = struct.unpack_from(">B", body, boff)
        boff += 1
        val_shape, consumed = _decode_shape(body, boff, val_ndim)
        boff += consumed

        # Allocate output arrays
        # shape[0] is the max block index — we'll size to hold all blocks
        blocks_list: list[KVBlock] = []
        key_slices: dict[int, np.ndarray] = {}
        val_slices: dict[int, np.ndarray] = {}

        max_block_id = 0

        for _ in range(num_blocks):
            # Block frame length
            (frame_len,) = struct.unpack_from(">I", body, boff)
            boff += 4
            frame_data = body[boff : boff + frame_len]
            boff += frame_len

            block, k_arr, v_arr = self.deserialize_block(frame_data)
            # k_arr, v_arr are numpy or MLX; convert to numpy for assembly
            blocks_list.append(block)
            key_slices[block.block_id] = _mx_to_numpy(k_arr)
            val_slices[block.block_id] = _mx_to_numpy(v_arr)
            if block.block_id > max_block_id:
                max_block_id = block.block_id

        # Assemble full key/value arrays sized to fit all block_ids
        # Use the shape from the header but replace dim-0 with max_block_id+1
        out_key_shape = (max_block_id + 1,) + key_shape[1:]
        out_val_shape = (max_block_id + 1,) + val_shape[1:]

        out_key = np.zeros(out_key_shape, dtype=key_dtype)
        out_val = np.zeros(out_val_shape, dtype=val_dtype)

        for bid, ks in key_slices.items():
            out_key[bid] = ks
        for bid, vs in val_slices.items():
            out_val[bid] = vs

        # Build BlockTable
        table = BlockTable(block_size=block_size)
        for block in blocks_list:
            table.append_block(block)

        # Convert to MLX if available
        key_out = _numpy_to_mx(out_key)
        val_out = _numpy_to_mx(out_val)

        return table, key_out, val_out

    # ── File I/O ──────────────────────────────────────────────────

    def save_to_file(self, path: str, table: BlockTable, key_cache, value_cache) -> None:
        """Write serialized table to disk."""
        data = self.serialize_table(table, key_cache, value_cache)
        with open(path, "wb") as f:
            f.write(data)

    def load_from_file(self, path: str) -> tuple[BlockTable, object, object]:
        """Load serialized table from disk.

        Returns:
            (BlockTable, key_array, value_array)
        """
        with open(path, "rb") as f:
            data = f.read()
        return self.deserialize_table(data)
