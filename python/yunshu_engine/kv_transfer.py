from __future__ import annotations

"""Yunshu KV Transfer Protocol — disaggregated prefill KV block transfer.

Provides a proper wire protocol for disaggregated prefill (external_prefill.py
lacks one): a connector with async load/store of KV blocks between nodes.

Architecture:
  KVTransferProtocol — wire format definition (header + payload)
  KVTransferClient — sends KV blocks to remote prefill nodes
  KVTransferServer — receives KV blocks and loads into local cache

Wire format (per message):
  ┌──────────────────────────────────────────────┐
  │ Header (JSON, length-prefixed) │
  │ request_id: str unique transfer ID │
  │ block_count: int number of KV blocks │
  │ model_name: str model identifier │
  │ compression: str "none"|"zstd"|"lz4" │
  │ checksum: str xxhash of payload │
  │ total_tokens: int total prefill tokens│
  │ layer_count: int number of KV layers │
  │ block_size: int tokens per block │
  ├──────────────────────────────────────────────┤
  │ Payload (binary, optionally compressed) │
  │ Per-block: │
  │ block_hash: uint64 content hash │
  │ token_count: int32 actual tokens │
  │ data_len: int32 bytes of KV data │
  │ data: bytes serialized KV tensors│
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

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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
_MAX_HEADER_SIZE = 1024 * 1024  # 1 MB max frame header (: DoS bound on header_len)
_CHUNK_SIZE = 64 * 1024  # 64 KB chunks for streaming


class CompressionType(StrEnum):
    """Compression algorithm for KV block payload."""

    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"


class TransferStatus(StrEnum):
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
        compression_str = os.environ.get(
            "YUNSHU_KV_TRANSFER_COMPRESSION", "none"
        ).lower()
        try:
            compression = CompressionType(compression_str)
        except ValueError:
            compression = CompressionType.NONE

        return cls(
            enabled=os.environ.get("YUNSHU_KV_TRANSFER", "0") == "1",
            listen_port=int(
                os.environ.get("YUNSHU_KV_TRANSFER_PORT", str(_DEFAULT_PORT))
            ),
            remote_host=os.environ.get("YUNSHU_KV_TRANSFER_REMOTE_HOST", "127.0.0.1"),
            remote_port=int(
                os.environ.get("YUNSHU_KV_TRANSFER_REMOTE_PORT", str(_DEFAULT_PORT))
            ),
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
    checksum: str  # hex digest of payload
    total_tokens: int
    layer_count: int
    block_size: int
    checksum_algorithm: str = "sha256"
    version: int = _HEADER_VERSION

    def to_json(self) -> bytes:
        """Serialize header to JSON bytes."""
        return json.dumps(
            {
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
            },
            separators=(",", ":"),
        ).encode("utf-8")

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
        completed_at: Monotonic timestamp when the transfer reached a terminal
            state.  Used by cleanup_expired_transfers() to determine wall-clock
            age (not duration_seconds, which is just the transfer time).
    """

    request_id: str
    status: TransferStatus
    blocks_transferred: int = 0
    bytes_transferred: int = 0
    bytes_original: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    checksum_verified: bool = False
    completed_at: float = 0.0

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
                logger.warning(
                    "xxhash requested but not available — falling back to sha256. "
                    "Checksum will NOT match a remote node using xxhash. "
                    "Install xxhash on all nodes for cross-node transfers."
                )
                algorithm = "sha256"

        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compress(data: bytes, method: CompressionType) -> tuple[bytes, CompressionType]:
        """Compress data using the specified method.

        Args:
            data: Raw bytes to compress.
            method: Compression algorithm.

        Returns:
            ``(payload, effective_method)`` — the (possibly uncompressed) bytes
            AND the method that was *actually* applied. Build caveat:
            when the requested codec is unavailable we fall back to NONE, and
            the caller MUST record ``effective_method`` in the wire header.
            Recording the requested method instead made the receiver attempt to
            decompress raw bytes → hard RuntimeError (codec missing) or silent
            corruption (codec present, raw input not a valid frame).
        """
        if method == CompressionType.NONE:
            return data, CompressionType.NONE

        if method == CompressionType.ZSTD:
            try:
                import zstandard as zstd

                compressor = zstd.ZstdCompressor()
                return compressor.compress(data), CompressionType.ZSTD
            except ImportError:
                logger.warning("zstandard not available, sending uncompressed")
                return data, CompressionType.NONE

        if method == CompressionType.LZ4:
            try:
                import lz4.frame

                return lz4.frame.compress(data), CompressionType.LZ4
            except ImportError:
                logger.warning("lz4 not available, sending uncompressed")
                return data, CompressionType.NONE

        return data, CompressionType.NONE

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
                raise RuntimeError(
                    "zstandard required for zstd decompression"
                ) from None

        if method == CompressionType.LZ4:
            try:
                import lz4.frame

                return lz4.frame.decompress(data)
            except ImportError:
                raise RuntimeError("lz4 required for lz4 decompression") from None

        return data

    @staticmethod
    def serialize_blocks(blocks: list[KVBlockData]) -> bytes:
        """Serialize a list of KVBlockData into binary payload.

        Per-block layout:
            [8 bytes] block_hash (uint64, big-endian)
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
            # Block hash (unsigned 64-bit to accommodate blake2b outputs)
            buf.write(struct.pack("!Q", block.block_hash & 0xFFFFFFFFFFFFFFFF))
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
        if num_blocks > 100_000:
            raise ValueError(f"Unreasonable num_blocks: {num_blocks}")

        for _ in range(num_blocks):
            block_hash = struct.unpack("!Q", buf.read(8))[0]
            token_count = struct.unpack("!i", buf.read(4))[0]
            num_layers = struct.unpack("!i", buf.read(4))[0]
            if num_layers < 0:
                raise ValueError(f"Negative num_layers: {num_layers}")
            if num_layers > 1000:
                raise ValueError(f"Unreasonable num_layers: {num_layers}")

            layer_data: dict[int, bytes] = {}
            for _ in range(num_layers):
                layer_idx = struct.unpack("!i", buf.read(4))[0]
                data_len = struct.unpack("!i", buf.read(4))[0]
                if data_len < 0:
                    raise ValueError(f"Negative data_len: {data_len}")
                if data_len > 64 * 1024 * 1024:  # 64 MB per layer
                    raise ValueError(f"Unreasonable data_len: {data_len}")
                layer_bytes = buf.read(data_len)
                layer_data[layer_idx] = layer_bytes

            blocks.append(
                KVBlockData(
                    block_hash=block_hash,
                    token_count=token_count,
                    layer_data=layer_data,
                )
            )

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
        len(raw_payload)
        # Record the EFFECTIVE method (compress() downgrades to NONE
        # when the codec is missing) so the receiver doesn't try to decompress
        # raw bytes.
        compressed_payload, effective_compression = cls.compress(
            raw_payload, compression
        )

        # Compute checksum on raw (uncompressed) payload
        checksum = cls.compute_checksum(raw_payload, checksum_algorithm)

        # Update header with actual values
        header = KVTransferHeader(
            request_id=message.header.request_id,
            block_count=message.header.block_count,
            model_name=message.header.model_name,
            compression=effective_compression.value,
            checksum=checksum,
            total_tokens=message.header.total_tokens,
            layer_count=message.header.layer_count,
            block_size=message.header.block_size,
            # Record the algorithm the checksum was ACTUALLY computed with. The
            # dataclass default is "sha256"; omitting this made an xxhash request
            # (YUNSHU_KV_TRANSFER_CHECKSUM=xxhash) store an xxhash digest under a
            # "sha256" label → the receiver recomputed sha256 → guaranteed
            # CHECKSUM_MISMATCH → every block discarded (decode silently re-prefilled).
            checksum_algorithm=checksum_algorithm,
        )

        header_bytes = header.to_json()

        # Build frame
        frame = io.BytesIO()
        frame.write(_MAGIC)  # 4 bytes magic
        frame.write(struct.pack("!I", len(header_bytes)))  # 4 bytes header len
        frame.write(header_bytes)  # N bytes header
        frame.write(struct.pack("!Q", len(compressed_payload)))  # 8 bytes payload len
        frame.write(compressed_payload)  # M bytes payload

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
        if header.version != _HEADER_VERSION:
            raise ValueError(
                f"KV transfer protocol version mismatch: received v{header.version}, "
                f"expected v{_HEADER_VERSION}"
            )

        # Payload
        payload_len = struct.unpack("!Q", buf.read(8))[0]
        compressed_payload = buf.read(payload_len)

        # Decompress
        compression = CompressionType(header.compression)
        raw_payload = cls.decompress(compressed_payload, compression)

        # Checksum verification
        computed_checksum = cls.compute_checksum(raw_payload, header.checksum_algorithm)
        checksum_ok = computed_checksum == header.checksum
        # If xxhash was requested but we fell back to sha256, the checksums
        # won't match because the sender used xxhash and we used sha256.
        # Retry with sha256 (the universal fallback) to avoid false mismatches.
        if not checksum_ok and header.checksum_algorithm == "xxhash":
            sha256_checksum = cls.compute_checksum(raw_payload, "sha256")
            if sha256_checksum == header.checksum:
                logger.warning(
                    "xxhash checksum mismatch but sha256 matches — sender likely "
                    "fell back to sha256. Install xxhash on all nodes."
                )
                checksum_ok = True

        if not checksum_ok:
            result = KVTransferResult(
                request_id=header.request_id,
                status=TransferStatus.CHECKSUM_MISMATCH,
                error=(
                    f"Checksum mismatch: computed {computed_checksum[:16]}... "
                    f"!= expected {header.checksum[:16]}..."
                ),
                completed_at=time.monotonic(),
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
            completed_at=time.monotonic(),
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

        # bound header_len before allocating. The payload length IS guarded
        # (below), but header_len (a !I, up to 4 GiB) was not — a corrupt/malicious peer
        # connecting to the 0.0.0.0-bound server could send magic + 0xFFFFFFFF and force a
        # multi-GB readexactly buffer → memory-pressure SIGABRT on a 36GB Mac. The header is
        # a small JSON dict (model_name/block_size/layer_count); 1 MiB is generous.
        if header_len > _MAX_HEADER_SIZE:
            raise ValueError(f"Header too large: {header_len} > {_MAX_HEADER_SIZE}")

        # Read header
        header_bytes = await reader.readexactly(header_len)

        # Read payload length
        payload_len_bytes = await reader.readexactly(8)
        payload_len = struct.unpack("!Q", payload_len_bytes)[0]

        if payload_len > _MAX_MESSAGE_SIZE:
            raise ValueError(f"Payload too large: {payload_len} > {_MAX_MESSAGE_SIZE}")

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
        hash_input = b"".join(struct.pack("!i", tid) for tid in block_token_ids)
        block_hash = int(hashlib.blake2b(hash_input, digest_size=8).hexdigest(), 16)

        # Extract layer data
        layer_data: dict[int, bytes] = {}

        if _HAS_MLX and mx is not None:
            try:
                for layer_idx, layer_cache in enumerate(kv_cache):
                    if hasattr(layer_cache, "state"):
                        # MLX KVCache with state (keys, values, offset)
                        state = layer_cache.state
                        if hasattr(state, "__iter__"):
                            state_tuple = tuple(state)
                            if len(state_tuple) >= 2:
                                keys, values = state_tuple[0], state_tuple[1]
                                if hasattr(keys, "item"):
                                    # mx.array — extract slice for THIS block.
                                    # the SEQUENCE axis of an mlx-lm KVCache
                                    # is -2 (shape (B, n_kv_heads, seq, head_dim)), NOT 0.
                                    # Slicing axis 0 returned the whole cache for block 0
                                    # and an empty (0,…) tensor for every later block →
                                    # corrupt/dropped KV on transfer. [..., s:e, :] slices
                                    # the seq axis robustly regardless of leading dims.
                                    k_slice = keys[..., block_start:block_end, :]
                                    v_slice = values[..., block_start:block_end, :]
                                    # Convert to bytes for transport
                                    k_bytes = _tensor_to_bytes(k_slice)
                                    v_bytes = _tensor_to_bytes(v_slice)
                                    layer_data[layer_idx] = k_bytes + v_bytes
                    elif hasattr(layer_cache, "offset"):
                        # Older-style cache
                        offset = getattr(layer_cache, "offset", 0)
                        if offset > 0:
                            keys = getattr(layer_cache, "keys", None)
                            vals = getattr(layer_cache, "values", None)
                            if keys is not None:
                                # Take relevant slice for THIS block on the seq axis (-2),
                                # not axis 0.
                                k_bytes = _tensor_to_bytes(
                                    keys[..., block_start:block_end, :]
                                )
                                v_bytes = _tensor_to_bytes(
                                    vals[..., block_start:block_end, :]
                                )
                                layer_data[layer_idx] = k_bytes + v_bytes
            except Exception:
                logger.debug(
                    "Failed to extract KV from layer caches",
                    exc_info=True,
                )

        blocks.append(
            KVBlockData(
                block_hash=block_hash,
                token_count=actual_tokens,
                layer_data=layer_data,
            )
        )

    return blocks


def _tensor_to_bytes(tensor: Any) -> bytes:
    """Convert an MLX tensor to bytes for serialization.

    Uses numpy as an intermediary since MLX arrays support
    .astype() and can be converted via numpy.
    """
    if _HAS_MLX and mx is not None and hasattr(tensor, "nbytes"):
        try:
            import numpy as _np

            # numpy has no bfloat16, so np.array() on a bf16 array raises —
            # cast to float32 first. We record the resulting numpy dtype so the
            # round-trip restores it EXACTLY (the old code hardcoded float16 on
            # read → silent corruption + 2x element count for any bf16/fp32 KV
            # cache). The receiver casts back to its cache dtype.
            src = tensor
            if getattr(tensor, "dtype", None) == getattr(mx, "bfloat16", None):
                src = tensor.astype(mx.float32)
            arr = _np.array(src)
            dt = str(arr.dtype).encode("ascii")
            # Record the ORIGINAL mlx dtype name (e.g. "bfloat16") separately from the
            # numpy transport dtype. numpy has no bfloat16, so a bf16 KV cache must be
            # transported as float32 — but the receiver MUST restore the original dtype,
            # else a bf16 cache comes back as float32: it mismatches the model's bf16
            # weights (dtype-mismatched concat / attention) on every disaggregated handoff.
            mlxdt = str(getattr(tensor, "dtype", "")).split(".")[-1].encode("ascii")
            header = (
                struct.pack("<I", len(dt))
                + dt
                + struct.pack("<I", len(mlxdt))
                + mlxdt
                + struct.pack("<I", arr.ndim)
                + b"".join(struct.pack("<q", int(d)) for d in arr.shape)
            )
            return header + arr.tobytes()
        except Exception:
            logger.debug("tensor_to_bytes via numpy failed", exc_info=True)
    # Fallback: just the bytes from the array
    if hasattr(tensor, "tobytes"):
        return tensor.tobytes()
    return b""


def _read_tensor(data: bytes, offset: int = 0) -> tuple[Any, int]:
    """Read ONE self-describing tensor from ``data`` at ``offset``.

    Returns ``(mx.array, new_offset)``. Self-describing framing lets multiple
    tensors (e.g. K then V) be concatenated and read back sequentially with
    exact dtype + shape. Raises on malformed input.
    """
    import numpy as _np

    (dlen,) = struct.unpack_from("<I", data, offset)
    offset += 4
    dt = data[offset : offset + dlen].decode("ascii")
    offset += dlen
    # Original mlx dtype name (paired with _tensor_to_bytes). Restores bf16 exactly
    # rather than leaving the receiver with the float32 transport dtype.
    (mlen,) = struct.unpack_from("<I", data, offset)
    offset += 4
    mlxdt = data[offset : offset + mlen].decode("ascii")
    offset += mlen
    (ndim,) = struct.unpack_from("<I", data, offset)
    offset += 4
    shape: list[int] = []
    for _ in range(ndim):
        (d,) = struct.unpack_from("<q", data, offset)
        offset += 8
        shape.append(d)
    npdt = _np.dtype(dt)
    count = 1
    for d in shape:
        count *= d
    nbytes = count * npdt.itemsize
    raw = data[offset : offset + nbytes]
    offset += nbytes
    arr = _np.frombuffer(raw, dtype=npdt).reshape(shape)
    out = mx.array(arr)
    # Cast back to the original mlx dtype (e.g. bfloat16) when it differs from the
    # numpy transport dtype, so the restored cache matches the model's weights.
    if mlxdt:
        target = getattr(mx, mlxdt, None)
        if target is not None and target != out.dtype:
            out = out.astype(target)
    return out, offset


def _bytes_to_tensor(data: bytes) -> Any:
    """Deserialize a single self-describing tensor. Returns None on failure."""
    if not data or not _HAS_MLX or mx is None:
        return None
    try:
        tensor, _ = _read_tensor(data, 0)
        return tensor
    except Exception:
        logger.debug("bytes_to_tensor failed (%d bytes)", len(data), exc_info=True)
        return None


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
                    # extract_kv_blocks stores each layer as k_bytes + v_bytes,
                    # two self-describing tensors. Read them back SEPARATELY
                    # with exact dtype + shape (the old code read one float16
                    # blob and split by shape[0]//2 — wrong dtype AND wrong
                    # geometry).
                    k_tensor, off = _read_tensor(layer_bytes, 0)
                    v_tensor, _ = _read_tensor(layer_bytes, off)
                    if k_tensor is not None and v_tensor is not None:
                        # append along the SEQUENCE axis (-2), not
                        # axis 0, and handle a fresh/empty target cache. The old code
                        # concatenated on axis 0 (batch) AND assumed .keys/.values were
                        # non-None — on a decode node's fresh KVCache (.keys is None) the
                        # concat raised TypeError (swallowed → 0 blocks loaded → KV handoff
                        # silently never happened, always re-prefilled).
                        def _seq_append(existing, new):
                            if existing is None:
                                return new
                            return mx.concatenate([existing, new], axis=-2)

                        if isinstance(layer_cache, list):
                            if len(layer_cache) >= 2:
                                layer_cache[0] = _seq_append(layer_cache[0], k_tensor)
                                layer_cache[1] = _seq_append(layer_cache[1], v_tensor)
                        elif hasattr(layer_cache, "keys") and hasattr(
                            layer_cache, "values"
                        ):
                            layer_cache.keys = _seq_append(layer_cache.keys, k_tensor)
                            layer_cache.values = _seq_append(
                                layer_cache.values, v_tensor
                            )
                            # Keep the cache's offset consistent with the loaded length so
                            # the decode node reads/writes at the right position.
                            with contextlib.suppress(Exception):
                                layer_cache.offset = int(layer_cache.keys.shape[-2])
                        elif hasattr(layer_cache, "state"):
                            # mlx-lm's `state` is a (keys, values) tuple
                            # property whose setter does `self.keys, self.values = v`.
                            # The old code did `mx.concatenate([state, k, v])` —
                            # concatenating a TUPLE and then unpacking a single array
                            # through a 2-tuple setter, which cannot work. Append to
                            # each of keys/values and set the tuple back.
                            cur = layer_cache.state
                            if isinstance(cur, (tuple, list)) and len(cur) == 2:
                                _ck, _cv = cur
                                layer_cache.state = (
                                    _seq_append(_ck, k_tensor),
                                    _seq_append(_cv, v_tensor),
                                )
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
        # cache the owning event loop alongside each connection. send_blocks_sync
        # runs every transfer under a fresh throwaway `asyncio.run` loop, so a writer cached
        # by a prior call is bound to a now-dead loop; reusing it raised "got Future attached
        # to a different loop" → every disagg transfer after the first silently failed and
        # decode fell back to re-prefill. Reuse a cached writer ONLY within its own loop.
        self._connections: dict[
            str,
            tuple[
                asyncio.StreamReader, asyncio.StreamWriter, asyncio.AbstractEventLoop
            ],
        ] = {}
        self._conn_lock = threading.Lock()

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
        for _addr, (_reader, writer, _loop) in self._connections.items():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("operation failed", exc_info=True)
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
                completed_at=time.monotonic(),
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
                    completed_at=time.monotonic(),
                )
                self._stats.record_send(result)
                return result
            else:
                return KVTransferResult(
                    request_id=request_id,
                    status=TransferStatus.FAILED,
                    error=f"Remote rejected transfer (ack={ack_data!r})",
                    duration_seconds=elapsed,
                    completed_at=time.monotonic(),
                )

        except TimeoutError:
            elapsed = time.monotonic() - t0
            return KVTransferResult(
                request_id=request_id,
                status=TransferStatus.FAILED,
                error=f"Transfer timed out after {self._config.timeout_seconds}s",
                duration_seconds=elapsed,
                completed_at=time.monotonic(),
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            return KVTransferResult(
                request_id=request_id,
                status=TransferStatus.FAILED,
                error=str(e),
                duration_seconds=elapsed,
                completed_at=time.monotonic(),
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
                        blocks,
                        request_id,
                        model_name,
                        total_tokens,
                        layer_count,
                    ),
                )
                return future.result(timeout=self._config.timeout_seconds)
        else:
            return asyncio.run(
                self.send_blocks(
                    blocks,
                    request_id,
                    model_name,
                    total_tokens,
                    layer_count,
                )
            )

    async def _get_connection(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Get or create a connection to the remote node."""
        addr = f"{self._config.remote_host}:{self._config.remote_port}"

        _cur_loop = asyncio.get_running_loop()
        with self._conn_lock:
            if addr in self._connections:
                reader, writer, cached_loop = self._connections[addr]
                # reuse ONLY within the same event loop. A writer bound to a
                # different (likely dead, throwaway-run) loop would raise on write/drain.
                if cached_loop is _cur_loop and not writer.is_closing():
                    return reader, writer
                # Foreign/dead loop or closing — discard. Don't await wait_closed()
                # on a writer whose loop may be dead (it would hang/raise); best-effort
                # close only.
                with contextlib.suppress(Exception):
                    writer.close()
                del self._connections[addr]

        reader, writer = await asyncio.open_connection(
            self._config.remote_host,
            self._config.remote_port,
        )
        with self._conn_lock:
            # Another coroutine may have opened a connection while we were
            # connecting — close the loser to avoid leaking file descriptors.
            if addr in self._connections:
                existing_reader, existing_writer, existing_loop = self._connections[
                    addr
                ]
                if existing_loop is _cur_loop and not existing_writer.is_closing():
                    # Use the existing (first-writer-wins) and close ours
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                    return existing_reader, existing_writer
                # Existing is stale (foreign loop or closing) — replace
                with contextlib.suppress(Exception):
                    existing_writer.close()
            self._connections[addr] = (reader, writer, _cur_loop)
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
        # a decode-side consumer callback. The mesh transfer server holds NO
        # reference to a decode engine (and the KVCacheManager has neither a load_kv_blocks
        # method nor a usable _kv_layers target), so received blocks landed in a hollow
        # consumer → "loaded 0 of N blocks" → the cross-node KV was discarded. The engine
        # registers a consumer (blocks, model_name) -> loaded_count that reconstructs the KV
        # into ITS prompt-cache store (using load_kv_blocks_into_cache, the proven
        # primitive), keeping this server engine-agnostic.
        self._block_consumer: Any | None = None

    @property
    def stats(self) -> KVTransferStats:
        return self._stats

    def set_kv_manager(self, kv_manager: Any) -> None:
        """Update the KV cache manager reference."""
        self._kv_manager = kv_manager

    def set_block_consumer(self, consumer: Any) -> None:
        """Register the decode-side KV block consumer.

        consumer is a callable (blocks: list[KVBlockData], model_name: str) ->
        int (number of blocks loaded). It reconstructs the received KV into the decode
        engine's reusable cache. Takes precedence over the kv_manager load path.
        """
        self._block_consumer = consumer

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
        for _req_id, result in self._active_transfers.items():
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
            _consecutive_timeouts = 0
            _max_timeouts = 3
            while self._running:
                try:
                    frame = await asyncio.wait_for(
                        KVTransferProtocol.read_frame(reader),
                        timeout=self._config.timeout_seconds,
                    )
                    _consecutive_timeouts = 0
                except TimeoutError:
                    _consecutive_timeouts += 1
                    if _consecutive_timeouts >= _max_timeouts:
                        break
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
                if self._semaphore is not None:
                    async with self._semaphore:
                        result = await self._process_frame(frame)
                else:
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

    async def _process_frame(self, frame: bytes) -> KVTransferResult:
        """Process a received transfer frame.

        Decodes the frame, verifies checksum, and loads blocks into
        the local KV cache. Tracks the transfer in _active_transfers
        so stop() can cancel in-progress work and TTL GC can clean up.
        """
        t0 = time.monotonic()
        request_id = f"err-{uuid.uuid4().hex[:8]}"

        try:
            message, decode_result = KVTransferProtocol.decode_message(frame)
            request_id = message.request_id

            if decode_result.status != TransferStatus.COMPLETED:
                # Checksum mismatch or decode failure
                self._stats.record_receive(decode_result)
                return decode_result

            # Track as in-progress so stop() can cancel
            in_progress = KVTransferResult(
                request_id=request_id,
                status=TransferStatus.IN_PROGRESS,
            )
            self._active_transfers[request_id] = in_progress

            # model-match guard. KV tensors from the prefill node are only valid
            # for an identically-configured model; loading mismatched-shape/dtype blocks
            # would corrupt decode. Reject (NACK) when the receiver's loaded model differs
            # from the sender's. (Was missing — masked only because the load path below is
            # not yet wired; guards against corruption once it is.)
            _local_model = getattr(self._kv_manager, "model_name", None) or getattr(
                self._kv_manager, "_model_name", None
            )
            _sender_model = getattr(message.header, "model_name", None)
            if _local_model and _sender_model and _local_model != _sender_model:
                logger.warning(
                    "KV transfer model mismatch: sender=%s receiver=%s — rejecting %s",
                    _sender_model,
                    _local_model,
                    request_id,
                )
                _fail = KVTransferResult(
                    request_id=request_id,
                    status=TransferStatus.FAILED,
                    error=f"model mismatch: sender {_sender_model} != receiver {_local_model}",
                    completed_at=time.monotonic(),
                )
                self._stats.record_receive(_fail)
                self._active_transfers[request_id] = _fail
                return _fail

            # Load blocks into local cache
            blocks_loaded = 0
            _load_attempted = False
            # prefer the decode-side consumer (reconstructs into the engine's
            # reusable cache via the proven load primitive) — the mesh server itself
            # holds no engine reference.
            if self._block_consumer is not None:
                _load_attempted = True
                try:
                    blocks_loaded = int(
                        self._block_consumer(
                            message.blocks,
                            message.header.model_name,
                        )
                    )
                except Exception as e:
                    logger.warning("KV block consumer failed: %s", e)
            elif self._kv_manager is not None:
                # Try to load via KV manager's cache injection
                _load_attempted = True
                try:
                    # If the manager has a load_kv_blocks method, use it
                    if hasattr(self._kv_manager, "load_kv_blocks"):
                        blocks_loaded = self._kv_manager.load_kv_blocks(
                            message.blocks,
                            model_name=message.header.model_name,
                        )
                    else:
                        # Direct cache loading (limited without BatchGenerator support)
                        kv_cache = getattr(self._kv_manager, "_kv_layers", None)
                        if kv_cache is not None:
                            blocks_loaded = load_kv_blocks_into_cache(
                                kv_cache,
                                message.blocks,
                            )
                except Exception as e:
                    logger.warning(
                        "Failed to load KV blocks into cache: %s",
                        e,
                    )

            elapsed = time.monotonic() - t0
            now = time.monotonic()
            # /975: report honestly. With NO consumer AND no kv_manager this is a pure
            # transfer relay (no cache to inject into) — the WIRE transfer genuinely succeeded,
            # so report the received block count. When a load WAS attempted (consumer or
            # kv_manager) report the REAL loaded count and FAIL when blocks were sent but none
            # loaded. The old `blocks_loaded or len(blocks)` reported a FULL successful transfer
            # even when a real load injected NOTHING → the decode node silently re-prefilled
            # while stats over-reported success.
            _sent = len(message.blocks)
            if not _load_attempted:
                _ok = True
                _reported = _sent
            else:
                _ok = (blocks_loaded > 0) or (_sent == 0)
                _reported = blocks_loaded
            result = KVTransferResult(
                request_id=request_id,
                status=TransferStatus.COMPLETED if _ok else TransferStatus.FAILED,
                blocks_transferred=_reported,
                bytes_transferred=len(frame),
                bytes_original=message.total_data_size,
                duration_seconds=elapsed,
                checksum_verified=decode_result.checksum_verified,
                completed_at=now,
                error=None
                if _ok
                else f"loaded 0 of {_sent} blocks (no KV load path wired on receiver)",
            )
            self._stats.record_receive(result)

            # Update tracking entry
            self._active_transfers[request_id] = result

            logger.info(
                "KV transfer received: %d blocks for %s (%d bytes, %.1f ms)",
                len(message.blocks),
                request_id,
                len(frame),
                elapsed * 1000,
            )

            return result

        except Exception as e:
            elapsed = time.monotonic() - t0
            result = KVTransferResult(
                request_id=request_id,
                status=TransferStatus.FAILED,
                error=str(e),
                duration_seconds=elapsed,
                completed_at=time.monotonic(),
            )
            self._stats.record_receive(result)
            self._active_transfers[request_id] = result
            return result

    def cleanup_expired_transfers(self, ttl_seconds: float = 300.0) -> int:
        """Remove completed/failed transfers older than TTL from tracking.

        This prevents unbounded growth of _active_transfers over time.

        Args:
            ttl_seconds: Maximum age in seconds before expiry (default 5 min).

        Returns:
            Number of expired entries removed.
        """
        if not self._active_transfers:
            return 0

        terminal_states = (
            TransferStatus.COMPLETED,
            TransferStatus.FAILED,
            TransferStatus.CHECKSUM_MISMATCH,
            TransferStatus.CANCELLED,
        )

        now = time.monotonic()
        expired_keys = [
            rid
            for rid, result in self._active_transfers.items()
            if result.status in terminal_states
            and result.completed_at > 0
            and (now - result.completed_at) > ttl_seconds
        ]
        for rid in expired_keys:
            del self._active_transfers[rid]

        removed = len(expired_keys)

        # Safety net: cap the dict size even if completed_at is missing.
        # Collect all non-IN_PROGRESS entries, sort by completed_at (oldest
        # first), and remove enough to get below the cap.
        max_tracked = 1000
        if len(self._active_transfers) > max_tracked:
            removable = [
                rid
                for rid, res in self._active_transfers.items()
                if res.status != TransferStatus.IN_PROGRESS
            ]
            # Sort by completed_at so oldest entries are evicted first.
            # Entries without completed_at (0.0) are treated as oldest.
            removable.sort(key=lambda rid: self._active_transfers[rid].completed_at)
            to_remove = len(self._active_transfers) - max_tracked
            for rid in removable[:to_remove]:
                del self._active_transfers[rid]
                removed += 1
        return removed

    def get_stats(self) -> dict:
        """Export server stats for monitoring."""
        stats = self._stats.to_dict()
        stats.update(
            {
                "enabled": self._config.enabled,
                "listen_port": self._config.listen_port,
                "active_transfers": len(self._active_transfers),
            }
        )
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
