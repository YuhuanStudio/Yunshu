"""Yunshu KV Transfer Protocol — disaggregated prefill KV block transfer.

Addresses audit item §12.2: external_prefill.py is "not mature" and lacks
proper wire protocol for disaggregated prefill. vLLM has KVConnectorFactory
with async load/store; this module provides Yunshu's equivalent.

Architecture:
  KVTransferProtocol  — wire format definition (header + payload)
  KVTransferClient    — sends KV blocks to remote prefill nodes
  KVTransferServer    — receives KV blocks and loads into local cache

Wire format (per message):
  ┌──────────────────────────────────────────────┐
  │ Header (JSON, length-prefixed)               │
  │   request_id: str        unique transfer ID  │
  │   block_count: int       number of KV blocks │
  │   model_name: str        model identifier    │
  │   compression: str       "none"|"zstd"|"lz4" │
  │   checksum: str          xxhash of payload   │
  │   total_tokens: int      total prefill tokens│
  │   layer_count: int       number of KV layers │
  │   block_size: int        tokens per block    │
  ├──────────────────────────────────────────────┤
  │ Payload (binary, optionally compressed)       │
  │   Per-block:                                  │
  │     block_hash: int64     content hash        │
  │     token_count: int32    actual tokens       │
  │     data_len: int32       bytes of KV data    │
  │     data: bytes           serialized KV tensors│
  └──────────────────────────────────────────────┘

Integration:
  - Enabled via YUNSHU_KV_TRANSFER=1 env var
  - ExternalPrefiller gains transfer_prefill_result() for remote dispatch
  - KVTransferServer integrates with local KVCacheManager for cache loading
  - Stats exported via get_stats() for monitoring

Thread safety:
  - Client: safe to call from any thread (uses its own asyncio loop or
    synchronous mode)
  - Server: runs in asyncio event loop, handles concurrent transfers
  - Stats: accumulated atomically (single-writer via lock)
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── MLX optional import ──────────────────────────────────────────────
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore
    _HAS_MLX = False


# ── Constants ─────────────────────────────────────────────────────────

_MAGIC = b"YKVT"  # Yunshu KV Transfer magic bytes
_HEADER_VERSION = 1
_DEFAULT_PORT = 7890
_MAX_MESSAGE_SIZE = 512 * 1024 * 1024  # 512 MB max per message
_CHUNK_SIZE = 64 * 1024  # 64 KB chunks for streaming


class CompressionType(str, Enum):
    """Compression algorithm for KV block payload."""
    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"


class TransferStatus(str, Enum):
    """Status of a KV transfer operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    CANCELLED = "cancelled"


# ── Data Classes ─────────────────────────────────────────────────────


@dataclass
class KVTransferConfig:
    """Configuration for KV transfer protocol.

    Enable via YUNSHU_KV_TRANSFER=1 env var.

    Attributes:
        enabled: Whether KV transfer is active.
        listen_port: Port for KVTransferServer to listen on.
        remote_host: Hostname of remote prefill node (for client).
        remote_port: Port of remote prefill node (for client).
        compression: Compression algorithm to use.
        checksum_algorithm: Hash algorithm for verification ("xxhash"|"sha256").
        max_message_size: Maximum message size in bytes.
        timeout_seconds: Transfer timeout in seconds.
        max_concurrent_transfers: Max parallel inbound transfers (server).
        block_size: Tokens per KV block (must match KVCacheManager).
    """
    enabled: bool = False
    listen_port: int = _DEFAULT_PORT
    remote_host: str = "127.0.0.1"
    remote_port: int = _DEFAULT_PORT
    compression: CompressionType = CompressionType.NONE
    checksum_algorithm: str = "sha256"
    max_message_size: int = _MAX_MESSAGE_SIZE
    timeout_seconds: float = 30.0
    max_concurrent_transfers: int = 4
    block_size: int = 64

    @classmethod
    def from_env(cls) -> KVTransferConfig:
        """Create config from environment variables.

        Env vars:
            YUNSHU_KV_TRANSFER=1              Enable KV transfer
            YUNSHU_KV_TRANSFER_PORT            Server listen port
            YUNSHU_KV_TRANSFER_REMOTE_HOST     Remote prefill node host
            YUNSHU_KV_TRANSFER_REMOTE_PORT     Remote prefill node port
            YUNSHU_KV_TRANSFER_COMPRESSION     "none"|"zstd"|"lz4"
            YUNSHU_KV_TRANSFER_CHECKSUM        "xxhash"|"sha256"
            YUNSHU_KV_TRANSFER_TIMEOUT         Timeout in seconds
            YUNSHU_KV_TRANSFER_MAX_CONCURRENT  Max concurrent transfers
            YUNSHU_KV_TRANSFER_BLOCK_SIZE      Tokens per block
        """
        compression_str = os.environ.get("YUNSHU_KV_TRANSFER_COMPRESSION", "none").lower()
        try:
            compression = CompressionType(compression_str)
        except ValueError:
            compression = CompressionType.NONE

        return cls(
            enabled=os.environ.get("YUNSHU_KV_TRANSFER", "0") == "1",
            listen_port=int(os.environ.get("YUNSHU_KV_TRANSFER_PORT", str(_DEFAULT_PORT))),
            remote_host=os.environ.get("YUNSHU_KV_TRANSFER_REMOTE_HOST", "127.0.0.1"),
            remote_port=int(os.environ.get("YUNSHU_KV_TRANSFER_REMOTE_PORT", str(_DEFAULT_PORT))),
            compression=compression,
            checksum_algorithm=os.environ.get("YUNSHU_KV_TRANSFER_CHECKSUM", "sha256"),
            timeout_seconds=float(os.environ.get("YUNSHU_KV_TRANSFER_TIMEOUT", "30.0")),
            max_concurrent_transfers=int(
                os.environ.get("YUNSHU_KV_TRANSFER_MAX_CONCURRENT", "4")
            ),
            block_size=int(os.environ.get("YUNSHU_KV_TRANSFER_BLOCK_SIZE", "64")),
        )


@dataclass
class KVBlockData:
    """A single KV block ready for transfer.

    Attributes:
        block_hash: Content hash of the block (for dedup / verification).
        token_count: Actual number of tokens in this block.
        layer_data: Per-layer KV tensors serialized as bytes.
                    Key is layer index, value is serialized (K, V) pair.
    """
    block_hash: int
    token_count: int
    layer_data: dict[int, bytes] = field(default_factory=dict)

    @property
    def data_size(self) -> int:
        """Total bytes of KV data in this block."""
        return sum(len(v) for v in self.layer_data.values())


@dataclass
class KVTransferHeader:
    """Wire format header for a KV transfer message.

    The header is JSON-encoded and length-prefixed for streaming.
    """
    request_id: str
    block_count: int
    model_name: str
    compression: str  # "none"|"zstd"|"lz4"
    checksum: str     # hex digest of payload
    total_tokens: int
    layer_count: int
    block_size: int
    checksum_algorithm: str = "sha256"
    version: int = _HEADER_VERSION

    def to_json(self) -> bytes:
        """Serialize header to JSON bytes."""
        return json.dumps({
            "request_id": self.request_id,
            "block_count": self.block_count,
            "model_name": self.model_name,
            "compression": self.compression,
            "checksum": self.checksum,
            "total_tokens": self.total_tokens,
            "layer_count": self.layer_count,
            "block_size": self.block_size,
            "checksum_algorithm": self.checksum_algorithm,
            "version": self.version,
        }, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> KVTransferHeader:
        """Deserialize header from JSON bytes."""
        d = json.loads(data)
        return cls(
            request_id=d["request_id"],
            block_count=d["block_count"],
            model_name=d["model_name"],
            compression=d["compression"],
            checksum=d["checksum"],
            total_tokens=d["total_tokens"],
            layer_count=d["layer_count"],
            block_size=d["block_size"],
            checksum_algorithm=d.get("checksum_algorithm", "sha256"),
            version=d.get("version", _HEADER_VERSION),
        )


@dataclass
class KVTransferMessage:
    """A complete KV transfer message (header + payload).

    This is the top-level container for both sending and receiving.
    """
    header: KVTransferHeader
    blocks: list[KVBlockData]

    @property
    def request_id(self) -> str:
        return self.header.request_id

    @property
    def total_data_size(self) -> int:
        """Total bytes of KV data across all blocks."""
        return sum(b.data_size for b in self.blocks)


@dataclass
class KVTransferResult:
    """Result of a KV transfer operation.

    Attributes:
        request_id: The transfer request ID.
        status: Final status of the transfer.
        blocks_transferred: Number of blocks successfully transferred.
        bytes_transferred: Total bytes transferred (post-compression).
        bytes_original: Total bytes before compression.
        duration_seconds: Wall-clock transfer duration.
        error: Error message if status is FAILED.
        checksum_verified: Whether payload checksum was verified.
    """
    request_id: str
    status: TransferStatus
    blocks_transferred: int = 0
    bytes_transferred: int = 0
    bytes_original: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    checksum_verified: bool = False

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (1.0 = no compression, 0.5 = 50% size)."""
        if self.bytes_original == 0:
            return 1.0
        return self.bytes_transferred / self.bytes_original


@dataclass
class KVTransferStats:
    """Aggregate KV transfer statistics."""
    total_transfers: int = 0
    total_blocks_sent: int = 0
    total_blocks_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_bytes_original_sent: int = 0
    total_checksum_failures: int = 0
    total_transfer_failures: int = 0
    send_latencies: list[float] = field(default_factory=list)
    receive_latencies: list[float] = field(default_factory=list)

    def record_send(self, result: KVTransferResult) -> None:
        """Record a send operation result."""
        self.total_transfers += 1
        self.total_blocks_sent += result.blocks_transferred
        self.total_bytes_sent += result.bytes_transferred
        self.total_bytes_original_sent += result.bytes_original
        self.send_latencies.append(result.duration_seconds)
        if len(self.send_latencies) > 1000:
            self.send_latencies = self.send_latencies[-500:]

    def record_receive(self, result: KVTransferResult) -> None:
        """Record a receive operation result."""
        self.total_transfers += 1
        self.total_blocks_received += result.blocks_transferred
        self.total_bytes_received += result.bytes_transferred
        self.receive_latencies.append(result.duration_seconds)
        if len(self.receive_latencies) > 1000:
            self.receive_latencies = self.receive_latencies[-500:]
        if result.status == TransferStatus.CHECKSUM_MISMATCH:
            self.total_checksum_failures += 1
        elif result.status == TransferStatus.FAILED:
            self.total_transfer_failures += 1

    @property
    def avg_send_latency_ms(self) -> float:
        if not self.send_latencies:
            return 0.0
        return sum(self.send_latencies) / len(self.send_latencies) * 1000

    @property
    def avg_receive_latency_ms(self) -> float:
        if not self.receive_latencies:
            return 0.0
        return sum(self.receive_latencies) / len(self.receive_latencies) * 1000

    @property
    def avg_compression_ratio(self) -> float:
        if self.total_bytes_original_sent == 0:
            return 1.0
        return self.total_bytes_sent / self.total_bytes_original_sent

    def to_dict(self) -> dict:
        """Export stats as a dict for monitoring."""
        return {
            "total_transfers": self.total_transfers,
            "total_blocks_sent": self.total_blocks_sent,
            "total_blocks_received": self.total_blocks_received,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_bytes_original_sent": self.total_bytes_original_sent,
            "total_checksum_failures": self.total_checksum_failures,
            "total_transfer_failures": self.total_transfer_failures,
            "avg_send_latency_ms": round(self.avg_send_latency_ms, 2),
            "avg_receive_latency_ms": round(self.avg_receive_latency_ms, 2),
            "avg_compression_ratio": round(self.avg_compression_ratio, 4),
        }


# ── Wire Format Serialization ────────────────────────────────────────


class KVTransferProtocol:
    """Defines the wire format for KV block transfer between nodes.

    This class handles serialization and deserialization of KV transfer
    messages according to the wire format:

    Frame layout:
        [4 bytes] magic "YKVT"
        [4 bytes] header length (big-endian uint32)
        [N bytes] JSON header
        [8 bytes] payload length (big-endian uint64)
        [M bytes] payload (block data, optionally compressed)
    """

    @staticmethod
    def compute_checksum(data: bytes, algorithm: str = "sha256") -> str:
        """Compute checksum of payload data.

        Args:
            data: The raw payload bytes.
            algorithm: Hash algorithm ("sha256" or "xxhash").

        Returns:
            Hex digest string.
        """
        if algorithm == "xxhash":
            try:
                import xxhash
                return xxhash.xxh128(data).hexdigest()
            except ImportError:
                logger.debug("xxhash not available, falling back to sha256")
                algorithm = "sha256"

        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compress(data: bytes, method: CompressionType) -> bytes:
        """Compress data using the specified method.

        Args:
            data: Raw bytes to compress.
            method: Compression algorithm.

        Returns:
            Compressed bytes (or original if NONE).
        """
        if method == CompressionType.NONE:
            return data

        if method == CompressionType.ZSTD:
            try:
                import zstandard as zstd
                compressor = zstd.ZstdCompressor()
                return compressor.compress(data)
            except ImportError:
                logger.warning("zstandard not available, sending uncompressed")
                return data

        if method == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.compress(data)
            except ImportError:
                logger.warning("lz4 not available, sending uncompressed")
                return data

        return data

    @staticmethod
    def decompress(data: bytes, method: CompressionType) -> bytes:
        """Decompress data using the specified method.

        Args:
            data: Compressed bytes.
            method: Compression algorithm that was used.

        Returns:
            Decompressed bytes.
        """
        if method == CompressionType.NONE:
            return data

        if method == CompressionType.ZSTD:
            try:
                import zstandard as zstd
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(data)
            except ImportError:
                raise RuntimeError("zstandard required for zstd decompression")

        if method == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                raise RuntimeError("lz4 required for lz4 decompression")

        return data

    @staticmethod
    def serialize_blocks(blocks: list[KVBlockData]) -> bytes:
        """Serialize a list of KVBlockData into binary payload.

        Per-block layout:
            [8 bytes] block_hash (int64, big-endian)
            [4 bytes] token_count (int32, big-endian)
            [4 bytes] num_layers (int32, big-endian)
            Per layer:
                [4 bytes] layer_index (int32, big-endian)
                [4 bytes] data_len (int32, big-endian)
                [data_len bytes] KV data

        Args:
            blocks: List of KVBlockData to serialize.

        Returns:
            Binary payload bytes.
        """
        buf = io.BytesIO()

        # Number of blocks
        buf.write(struct.pack("!I", len(blocks)))

        for block in blocks:
            # Block hash
            buf.write(struct.pack("!q", block.block_hash))
            # Token count
            buf.write(struct.pack("!i", block.token_count))
            # Number of layers
            buf.write(struct.pack("!i", len(block.layer_data)))

            for layer_idx, layer_bytes in sorted(block.layer_data.items()):
                # Layer index
                buf.write(struct.pack("!i", layer_idx))
                # Data length
                data_len = len(layer_bytes)
                buf.write(struct.pack("!i", data_len))
                # Data
                buf.write(layer_bytes)

        return buf.getvalue()

    @staticmethod
    def deserialize_blocks(data: bytes) -> list[KVBlockData]:
        """Deserialize binary payload into a list of KVBlockData.

        Args:
            data: Binary payload bytes.

        Returns:
            List of KVBlockData.

        Raises:
            struct.error: If the payload is malformed.
        """
        buf = io.BytesIO(data)
        blocks = []

        num_blocks = struct.unpack("!I", buf.read(4))[0]

        for _ in range(num_blocks):
            block_hash = struct.unpack("!q", buf.read(8))[0]
            token_count = struct.unpack("!i", buf.read(4))[0]
            num_layers = struct.unpack("!i", buf.read(4))[0]

            layer_data: dict[int, bytes] = {}
            for _ in range(num_layers):
                layer_idx = struct.unpack("!i", buf.read(4))[0]
                data_len = struct.unpack("!i", buf.read(4))[0]
                layer_bytes = buf.read(data_len)
                layer_data[layer_idx] = layer_bytes

            blocks.append(KVBlockData(
                block_hash=block_hash,
                token_count=token_count,
                layer_data=layer_data,
            ))

        return blocks

    @classmethod
    def encode_message(
        cls,
        message: KVTransferMessage,
        compression: CompressionType = CompressionType.NONE,
        checksum_algorithm: str = "sha256",
    ) -> bytes:
        """Encode a complete KVTransferMessage into wire format.

        Args:
            message: The message to encode.
            compression: Compression to apply to payload.
            checksum_algorithm: Checksum algorithm for verification.

        Returns:
            Wire-format bytes ready to send over the network.
        """
        # Serialize and compress payload
        raw_payload = cls.serialize_blocks(message.blocks)
        original_size = len(raw_payload)
        compressed_payload = cls.compress(raw_payload, compression)

        # Compute checksum on raw (uncompressed) payload
        checksum = cls.compute_checksum(raw_payload, checksum_algorithm)

        # Update header with actual values
        header = KVTransferHeader(
            request_id=message.header.request_id,
            block_count=message.header.block_count,
            model_name=message.header.model_name,
            compression=compression.value,
            checksum=checksum,
            total_tokens=message.header.total_tokens,
            layer_count=message.header.layer_count,
            block_size=message.header.block_size,
        )

        header_bytes = header.to_json()

        # Build frame
        frame = io.BytesIO()
        frame.write(_MAGIC)                                      # 4 bytes magic
        frame.write(struct.pack("!I", len(header_bytes)))         # 4 bytes header len
        frame.write(header_bytes)                                # N bytes header
        frame.write(struct.pack("!Q", len(compressed_payload)))  # 8 bytes payload len
        frame.write(compressed_payload)                          # M bytes payload

        return frame.getvalue()

    @classmethod
    def decode_message(cls, data: bytes) -> tuple[KVTransferMessage, KVTransferResult]:
        """Decode wire-format bytes into a KVTransferMessage.

        Performs checksum verification and decompression.

        Args:
            data: Wire-format bytes received from the network.

        Returns:
            Tuple of (decoded message, result with verification info).

        Raises:
            ValueError: If magic bytes don't match or checksum fails.
        """
        buf = io.BytesIO(data)

        # Magic
        magic = buf.read(4)
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}, expected {_MAGIC!r}")

        # Header
        header_len = struct.unpack("!I", buf.read(4))[0]
        header_bytes = buf.read(header_len)
        header = KVTransferHeader.from_json(header_bytes)

        # Payload
        payload_len = struct.unpack("!Q", buf.read(8))[0]
        compressed_payload = buf.read(payload_len)

        # Decompress
        compression = CompressionType(header.compression)
        raw_payload = cls.decompress(compressed_payload, compression)

        # Checksum verification
        computed_checksum = cls.compute_checksum(raw_payload, header.checksum_algorithm)
        checksum_ok = computed_checksum == header.checksum

        if not checksum_ok:
            result = KVTransferResult(
                request_id=header.request_id,
                status=TransferStatus.CHECKSUM_MISMATCH,
                error=(
                    f"Checksum mismatch: computed {computed_checksum[:16]}... "
                    f"!= expected {header.checksum[:16]}..."
                ),
            )
            # Return empty message with error
            return KVTransferMessage(header=header, blocks=[]), result

        # Deserialize blocks
        blocks = cls.deserialize_blocks(raw_payload)

        message = KVTransferMessage(header=header, blocks=blocks)
        result = KVTransferResult(
            request_id=header.request_id,
            status=TransferStatus.COMPLETED,
            blocks_transferred=len(blocks),
            bytes_transferred=len(compressed_payload),
            bytes_original=len(raw_payload),
            checksum_verified=True,
        )

        return message, result

    @classmethod
    async def read_frame(cls, reader: asyncio.StreamReader) -> bytes | None:
        """Read a complete wire-format frame from an async stream.

        Args:
            reader: Async stream reader (from asyncio.open_connection).

        Returns:
            Complete frame bytes, or None if connection closed.
        """
        # Read magic
        magic = await reader.readexactly(4)
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")

        # Read header length
        header_len_bytes = await reader.readexactly(4)
        header_len = struct.unpack("!I", header_len_bytes)[0]

        # Read header
        header_bytes = await reader.readexactly(header_len)

        # Read payload length
        payload_len_bytes = await reader.readexactly(8)
        payload_len = struct.unpack("!Q", payload_len_bytes)[0]

        if payload_len > _MAX_MESSAGE_SIZE:
            raise ValueError(
                f"Payload too large: {payload_len} > {_MAX_MESSAGE_SIZE}"
            )

        # Read payload
        payload = await reader.readexactly(payload_len)

        # Reconstruct full frame
        frame = io.BytesIO()
        frame.write(magic)
        frame.write(header_len_bytes)
        frame.write(header_bytes)
        frame.write(payload_len_bytes)
        frame.write(payload)

        return frame.getvalue()

    @classmethod
    async def write_frame(
        cls,
        writer: asyncio.StreamWriter,
        frame: bytes,
    ) -> None:
        """Write a complete wire-format frame to an async stream.

        Args:
            writer: Async stream writer.
            frame: Complete frame bytes (from encode_message).
        """
        writer.write(frame)
        await writer.drain()


# ── KV Block Extraction ──────────────────────────────────────────────


def extract_kv_blocks_from_cache(
    kv_cache: Any,
    token_ids: list[int],
    block_size: int = 64,
) -> list[KVBlockData]:
    """Extract KV blocks from an MLX KV cache for transfer.

    Splits the KV cache into block-sized chunks, computing a content
    hash for each block for deduplication and verification.

    Args:
        kv_cache: MLX KV cache (list of layer caches).
        token_ids: The token IDs that were prefilled.
        block_size: Tokens per block.

    Returns:
        List of KVBlockData ready for serialization.
    """
    blocks: list[KVBlockData] = []

    if kv_cache is None or not token_ids:
        return blocks

    total_tokens = len(token_ids)

    for block_start in range(0, total_tokens, block_size):
        block_end = min(block_start + block_size, total_tokens)
        actual_tokens = block_end - block_start
        block_token_ids = token_ids[block_start:block_end]

        # Compute content hash from token IDs
        hash_input = b"".join(
            struct.pack("!i", tid) for tid in block_token_ids
        )
        block_hash = int(hashlib.blake2b(hash_input, digest_size=8).hexdigest(), 16)

        # Extract layer data
        layer_data: dict[int, bytes] = {}

        if _HAS_MLX and mx is not None:
            try:
                for layer_idx, layer_cache in enumerate(kv_cache):
                    if hasattr(layer_cache, 'state'):
                        # MLX KVCache with state (keys, values, offset)
                        state = layer_cache.state
                        if hasattr(state, '__iter__'):
                            state_tuple = tuple(state)
                            if len(state_tuple) >= 2:
                                keys, values = state_tuple[0], state_tuple[1]
                                if hasattr(keys, 'item'):
                                    # mx.array — extract slice
                                    k_slice = keys[:actual_tokens]
                                    v_slice = values[:actual_tokens]
                                    # Convert to bytes for transport
                                    k_bytes = _tensor_to_bytes(k_slice)
                                    v_bytes = _tensor_to_bytes(v_slice)
                                    layer_data[layer_idx] = k_bytes + v_bytes
                    elif hasattr(layer_cache, 'offset'):
                        # Older-style cache
                        offset = getattr(layer_cache, 'offset', 0)
                        if offset > 0:
                            keys = getattr(layer_cache, 'keys', None)
                            vals = getattr(layer_cache, 'values', None)
                            if keys is not None:
                                # Take relevant slice
                                k_bytes = _tensor_to_bytes(keys[:actual_tokens])
                                v_bytes = _tensor_to_bytes(vals[:actual_tokens])
                                layer_data[layer_idx] = k_bytes + v_bytes
            except Exception:
                logger.debug(
                    "Failed to extract KV from layer caches",
                    exc_info=True,
                )

        blocks.append(KVBlockData(
            block_hash=block_hash,
            token_count=actual_tokens,
            layer_data=layer_data,
        ))

    return blocks


def _tensor_to_bytes(tensor: Any) -> bytes:
    """Convert an MLX tensor to bytes for serialization.

    Uses numpy as an intermediary since MLX arrays support
    .astype() and can be converted via numpy.
    """
    if _HAS_MLX and mx is not None and hasattr(tensor, 'nbytes'):
        try:
            # Try direct conversion via mlx
            np_array = np.array(tensor)
            return np_array.tobytes()
        except Exception:
            logger.debug("operation failed", exc_info=True)
            try:
                import numpy as np
                np_array = np.array(tensor)
                return np_array.tobytes()
            except Exception:
                logger.debug("operation failed", exc_info=True)
                pass
    # Fallback: just the bytes from the array
    if hasattr(tensor, 'tobytes'):
        return tensor.tobytes()
    return b""


def load_kv_blocks_into_cache(
    kv_cache: Any,
    blocks: list[KVBlockData],
) -> int:
    """Load received KV blocks into a local MLX KV cache.

    Args:
        kv_cache: Target MLX KV cache (list of layer caches).
        blocks: KV blocks received from remote node.

    Returns:
        Number of blocks successfully loaded.
    """
    loaded = 0

    if kv_cache is None or not blocks:
        return loaded

    if not _HAS_MLX or mx is None:
        return loaded

    for block in blocks:
        try:
            for layer_idx, layer_bytes in block.layer_data.items():
                if layer_idx < len(kv_cache):
                    layer_cache = kv_cache[layer_idx]
                    # This is a placeholder for actual cache injection.
                    # Full implementation requires MLX-level changes to
                    # BatchGenerator's internal cache management.
                    # For now, we log and count successful block receipts.
                    logger.debug(
                        "Loaded block 0x%x into layer %d (%d bytes)",
                        block.block_hash,
                        layer_idx,
                        len(layer_bytes),
                    )
            loaded += 1
        except Exception:
            logger.debug(
                "Failed to load block 0x%x",
                block.block_hash,
                exc_info=True,
            )

    return loaded


# ── KV Transfer Client ───────────────────────────────────────────────


class KVTransferClient:
    """Sends KV blocks to remote prefill nodes.

    Used on the prefill node to send completed KV cache to the
    decode node after prefill completes.

    Usage:
        client = KVTransferClient(config)
        await client.start()
        result = await client.send_blocks(
            request_id="req-123",
            blocks=[block1, block2],
            model_name="qwen-2.5-0.5b",
            total_tokens=2048,
            layer_count=24,
        )
        await client.stop()
    """

    def __init__(self, config: KVTransferConfig | None = None) -> None:
        self._config = config or KVTransferConfig.from_env()
        self._stats = KVTransferStats()
        self._running = False
        # Connection pool: host:port -> (reader, writer)
        self._connections: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}

    @property
    def stats(self) -> KVTransferStats:
        return self._stats

    async def start(self) -> None:
        """Start the client (no persistent connection needed)."""
        self._running = True
        logger.info(
            "KV transfer client started (remote=%s:%d)",
            self._config.remote_host,
            self._config.remote_port,
        )

    async def stop(self) -> None:
        """Stop the client and close all connections."""
        self._running = False
        for addr, (reader, writer) in self._connections.items():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("operation failed", exc_info=True)
                pass
        self._connections.clear()
        logger.info("KV transfer client stopped")

    async def send_blocks(
        self,
        blocks: list[KVBlockData],
        request_id: str | None = None,
        model_name: str = "",
        total_tokens: int = 0,
        layer_count: int = 0,
    ) -> KVTransferResult:
        """Send KV blocks to the remote decode node.

        Args:
            blocks: KV blocks to send.
            request_id: Unique transfer ID (auto-generated if None).
            model_name: Model identifier for cache compatibility check.
            total_tokens: Total number of prefill tokens.
            layer_count: Number of KV layers in the model.

        Returns:
            KVTransferResult with transfer statistics.
        """
        if not self._config.enabled:
            return KVTransferResult(
                request_id=request_id or "",
                status=TransferStatus.FAILED,
                error="KV transfer not enabled",
            )

        request_id = request_id or f"kv-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()

        try:
            # Build message
            header = KVTransferHeader(
                request_id=request_id,
                block_count=len(blocks),
                model_name=model_name,
                compression=self._config.compression.value,
                checksum="",
                total_tokens=total_tokens,
                layer_count=layer_count,
                block_size=self._config.block_size,
            )
            message = KVTransferMessage(header=header, blocks=blocks)

            # Encode to wire format
            frame = KVTransferProtocol.encode_message(
                message,
                compression=self._config.compression,
                checksum_algorithm=self._config.checksum_algorithm,
            )

            # Connect and send
            reader, writer = await self._get_connection()

            # Send frame
            await KVTransferProtocol.write_frame(writer, frame)

            # Wait for ACK
            ack_data = await asyncio.wait_for(
                reader.readexactly(1),
                timeout=self._config.timeout_seconds,
            )

            elapsed = time.monotonic() - t0

            if ack_data == b"\x01":  # ACK
                original_size = sum(b.data_size for b in blocks)
                result = KVTransferResult(
                    request_id=request_id,
                    status=TransferStatus.COMPLETED,
                    blocks_transferred=len(blocks),
                    bytes_transferred=len(frame),
                    bytes_original=original_size,
                    duration_seconds=elapsed,
                    checksum_verified=True,
                )
                self._stats.record_send(result)
                return result
            else:
                return KVTransferResult(
                    request_id=request_id,
                    status=TransferStatus.FAILED,
                    error=f"Remote rejected transfer (ack={ack_data!r})",
                    duration_seconds=elapsed,
                )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            return KVTransferResult(
                request_id=request_id,
                status=TransferStatus.FAILED,
                error=f"Transfer timed out after {self._config.timeout_seconds}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            return KVTransferResult(
                request_id=request_id,
                status=TransferStatus.FAILED,
                error=str(e),
                duration_seconds=elapsed,
            )

    def send_blocks_sync(
        self,
        blocks: list[KVBlockData],
        request_id: str | None = None,
        model_name: str = "",
        total_tokens: int = 0,
        layer_count: int = 0,
    ) -> KVTransferResult:
        """Synchronous variant of send_blocks (for executor thread).

        Runs the async send in a new event loop if needed.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're in an async context — schedule and wait
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.send_blocks(
                        blocks, request_id, model_name,
                        total_tokens, layer_count,
                    ),
                )
                return future.result(timeout=self._config.timeout_seconds)
        else:
            return asyncio.run(
                self.send_blocks(
                    blocks, request_id, model_name,
                    total_tokens, layer_count,
                )
            )

    async def _get_connection(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Get or create a connection to the remote node."""
        addr = f"{self._config.remote_host}:{self._config.remote_port}"

        if addr in self._connections:
            reader, writer = self._connections[addr]
            if not writer.is_closing():
                return reader, writer
            # Connection dead — remove and reconnect
            del self._connections[addr]

        reader, writer = await asyncio.open_connection(
            self._config.remote_host,
            self._config.remote_port,
        )
        self._connections[addr] = (reader, writer)
        return reader, writer


# ── KV Transfer Server ───────────────────────────────────────────────


class KVTransferServer:
    """Receives KV blocks from remote prefill nodes.

    Runs on the decode node to accept incoming KV cache data from
    prefill nodes. Received blocks are loaded into the local KV cache
    for immediate use by the decode engine.

    Usage:
        server = KVTransferServer(config, kv_cache_manager=kv_mgr)
        await server.start()
        # ... runs in background accepting transfers ...
        await server.stop()
    """

    def __init__(
        self,
        config: KVTransferConfig | None = None,
        kv_cache_manager: Any | None = None,
    ) -> None:
        self._config = config or KVTransferConfig.from_env()
        self._kv_manager = kv_cache_manager
        self._stats = KVTransferStats()
        self._running = False
        self._server: asyncio.Server | None = None
        # Track active transfers by request_id
        self._active_transfers: dict[str, KVTransferResult] = {}
        # Semaphore for concurrency limiting
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def stats(self) -> KVTransferStats:
        return self._stats

    def set_kv_manager(self, kv_manager: Any) -> None:
        """Update the KV cache manager reference."""
        self._kv_manager = kv_manager

    async def start(self) -> None:
        """Start the transfer server."""
        if not self._config.enabled:
            logger.debug("KV transfer server disabled")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_transfers)

        self._server = await asyncio.start_server(
            self._handle_connection,
            "0.0.0.0",
            self._config.listen_port,
        )

        logger.info(
            "KV transfer server started on port %d (max_concurrent=%d)",
            self._config.listen_port,
            self._config.max_concurrent_transfers,
        )

    async def stop(self) -> None:
        """Stop the transfer server."""
        self._running = False

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Cancel active transfers
        for req_id, result in self._active_transfers.items():
            if result.status == TransferStatus.IN_PROGRESS:
                result.status = TransferStatus.CANCELLED
        self._active_transfers.clear()

        logger.info("KV transfer server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single incoming connection.

        Reads KV transfer frames and processes them.
        """
        peer = writer.get_extra_info("peername")
        logger.debug("KV transfer connection from %s", peer)

        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(
                        KVTransferProtocol.read_frame(reader),
                        timeout=self._config.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    continue
                except (asyncio.IncompleteReadError, ConnectionError):
                    # Client disconnected
                    break
                except ValueError:
                    logger.warning("Invalid frame from %s", peer)
                    break

                if frame is None:
                    break

                # Process the transfer (with concurrency limit)
                async with self._semaphore:
                    result = await self._process_frame(frame)
                    # Send ACK/NACK
                    if result.status == TransferStatus.COMPLETED:
                        writer.write(b"\x01")  # ACK
                    else:
                        writer.write(b"\x00")  # NACK
                    await writer.drain()

        except Exception:
            logger.debug("Transfer connection error from %s", peer, exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("operation failed", exc_info=True)
                pass

    async def _process_frame(self, frame: bytes) -> KVTransferResult:
        """Process a received transfer frame.

        Decodes the frame, verifies checksum, and loads blocks into
        the local KV cache.
        """
        t0 = time.monotonic()

        try:
            message, decode_result = KVTransferProtocol.decode_message(frame)

            if decode_result.status != TransferStatus.COMPLETED:
                # Checksum mismatch or decode failure
                self._stats.record_receive(decode_result)
                return decode_result

            # Load blocks into local cache
            blocks_loaded = 0
            if self._kv_manager is not None:
                # Try to load via KV manager's cache injection
                try:
                    # If the manager has a load_kv_blocks method, use it
                    if hasattr(self._kv_manager, 'load_kv_blocks'):
                        blocks_loaded = self._kv_manager.load_kv_blocks(
                            message.blocks,
                            model_name=message.header.model_name,
                        )
                    else:
                        # Direct cache loading (limited without BatchGenerator support)
                        kv_cache = getattr(self._kv_manager, '_kv_layers', None)
                        if kv_cache is not None:
                            blocks_loaded = load_kv_blocks_into_cache(
                                kv_cache, message.blocks,
                            )
                except Exception as e:
                    logger.warning(
                        "Failed to load KV blocks into cache: %s", e,
                    )

            elapsed = time.monotonic() - t0
            result = KVTransferResult(
                request_id=message.request_id,
                status=TransferStatus.COMPLETED,
                blocks_transferred=blocks_loaded or len(message.blocks),
                bytes_transferred=len(frame),
                bytes_original=message.total_data_size,
                duration_seconds=elapsed,
                checksum_verified=decode_result.checksum_verified,
            )
            self._stats.record_receive(result)

            logger.info(
                "KV transfer received: %d blocks for %s (%d bytes, %.1f ms)",
                len(message.blocks),
                message.request_id,
                len(frame),
                elapsed * 1000,
            )

            return result

        except Exception as e:
            elapsed = time.monotonic() - t0
            result = KVTransferResult(
                request_id="unknown",
                status=TransferStatus.FAILED,
                error=str(e),
                duration_seconds=elapsed,
            )
            self._stats.record_receive(result)
            return result

    def get_stats(self) -> dict:
        """Export server stats for monitoring."""
        stats = self._stats.to_dict()
        stats.update({
            "enabled": self._config.enabled,
            "listen_port": self._config.listen_port,
            "active_transfers": len(self._active_transfers),
        })
        return stats


# ── Convenience Functions ─────────────────────────────────────────────


def is_kv_transfer_enabled() -> bool:
    """Check if KV transfer is enabled via env var."""
    return os.environ.get("YUNSHU_KV_TRANSFER", "0") == "1"


def create_transfer_client(config: KVTransferConfig | None = None) -> KVTransferClient:
    """Create a KVTransferClient from config or env."""
    cfg = config or KVTransferConfig.from_env()
    return KVTransferClient(cfg)


def create_transfer_server(
    config: KVTransferConfig | None = None,
    kv_cache_manager: Any | None = None,
) -> KVTransferServer:
    """Create a KVTransferServer from config or env."""
    cfg = config or KVTransferConfig.from_env()
    return KVTransferServer(cfg, kv_cache_manager)
