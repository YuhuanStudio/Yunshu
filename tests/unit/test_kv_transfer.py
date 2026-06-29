"""KV Transfer Protocol tests — wire format, client/server, compression, stats.

Tests the KV transfer protocol with mocked networking:
- KVTransferConfig construction and from_env()
- KVBlockData and KVTransferHeader serialization
- KVTransferProtocol encode/decode round-trip
- Checksum verification (correct and mismatch)
- Compression round-trip (none; zstd/lz4 when available)
- KVTransferClient send with mock server
- KVTransferServer receive and stats
- KVTransferStats accumulation
- ExternalPrefiller.transfer_prefill_result integration
- Edge cases: empty blocks, single block, large batch
"""

from __future__ import annotations

import asyncio
import os
import struct
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.external_prefill import ExternalPrefiller, PrefillResult
from yunshu_engine.kv_transfer import (
    CompressionType,
    KVBlockData,
    KVTransferClient,
    KVTransferConfig,
    KVTransferHeader,
    KVTransferMessage,
    KVTransferProtocol,
    KVTransferResult,
    KVTransferServer,
    KVTransferStats,
    TransferStatus,
    create_transfer_client,
    create_transfer_server,
    extract_kv_blocks_from_cache,
    is_kv_transfer_enabled,
    load_kv_blocks_into_cache,
)

# ── Fakes ──


class _FakeModel:
    def __call__(self, input_ids, cache=None, **kwargs):
        return MagicMock()


class _FakeTokenizer:
    eos_token_ids = [3]
    has_thinking = False

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(f"t{t}" for t in tokens)


# ── KVTransferConfig ──


class TestKVTransferConfig:
    def test_default_disabled(self):
        config = KVTransferConfig()
        assert config.enabled is False
        assert config.listen_port == 7890
        assert config.compression == CompressionType.NONE
        assert config.block_size == 64

    def test_from_env_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            config = KVTransferConfig.from_env()
        assert config.enabled is False

    def test_from_env_enabled(self):
        with patch.dict(os.environ, {"YUNSHU_KV_TRANSFER": "1"}):
            config = KVTransferConfig.from_env()
        assert config.enabled is True

    def test_from_env_custom_port(self):
        with patch.dict(
            os.environ,
            {
                "YUNSHU_KV_TRANSFER": "1",
                "YUNSHU_KV_TRANSFER_PORT": "9999",
                "YUNSHU_KV_TRANSFER_REMOTE_HOST": "node-2.local",
                "YUNSHU_KV_TRANSFER_REMOTE_PORT": "8888",
            },
        ):
            config = KVTransferConfig.from_env()
        assert config.listen_port == 9999
        assert config.remote_host == "node-2.local"
        assert config.remote_port == 8888

    def test_from_env_compression_invalid_falls_back(self):
        with patch.dict(
            os.environ,
            {
                "YUNSHU_KV_TRANSFER_COMPRESSION": "invalid",
            },
        ):
            config = KVTransferConfig.from_env()
        assert config.compression == CompressionType.NONE

    def test_from_env_compression_zstd(self):
        with patch.dict(
            os.environ,
            {
                "YUNSHU_KV_TRANSFER_COMPRESSION": "zstd",
            },
        ):
            config = KVTransferConfig.from_env()
        assert config.compression == CompressionType.ZSTD

    def test_from_env_timeout(self):
        with patch.dict(
            os.environ,
            {
                "YUNSHU_KV_TRANSFER_TIMEOUT": "60.0",
            },
        ):
            config = KVTransferConfig.from_env()
        assert config.timeout_seconds == 60.0


# ── KVBlockData ──


class TestKVBlockData:
    def test_basic_construction(self):
        block = KVBlockData(block_hash=0xABCD, token_count=64)
        assert block.block_hash == 0xABCD
        assert block.token_count == 64
        assert block.layer_data == {}

    def test_data_size_empty(self):
        block = KVBlockData(block_hash=1, token_count=0)
        assert block.data_size == 0

    def test_data_size_with_layers(self):
        block = KVBlockData(
            block_hash=1,
            token_count=64,
            layer_data={0: b"abc", 1: b"defg"},
        )
        assert block.data_size == 7

    def test_data_size_large(self):
        data = {i: b"\x00" * 1024 for i in range(24)}
        block = KVBlockData(block_hash=1, token_count=64, layer_data=data)
        assert block.data_size == 24 * 1024


# ── KVTransferHeader ──


class TestKVTransferHeader:
    def test_to_json_roundtrip(self):
        header = KVTransferHeader(
            request_id="req-test",
            block_count=5,
            model_name="qwen-2.5",
            compression="none",
            checksum="abc123",
            total_tokens=320,
            layer_count=24,
            block_size=64,
        )
        json_bytes = header.to_json()
        restored = KVTransferHeader.from_json(json_bytes)

        assert restored.request_id == "req-test"
        assert restored.block_count == 5
        assert restored.model_name == "qwen-2.5"
        assert restored.compression == "none"
        assert restored.checksum == "abc123"
        assert restored.total_tokens == 320
        assert restored.layer_count == 24
        assert restored.block_size == 64

    def test_to_json_is_bytes(self):
        header = KVTransferHeader(
            request_id="r1",
            block_count=0,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=0,
            layer_count=0,
            block_size=64,
        )
        result = header.to_json()
        assert isinstance(result, bytes)
        assert b"r1" in result

    def test_version_default(self):
        header = KVTransferHeader(
            request_id="r1",
            block_count=0,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=0,
            layer_count=0,
            block_size=64,
        )
        assert header.version == 1

    def test_from_json_missing_version_uses_default(self):
        import json

        data = json.dumps(
            {
                "request_id": "r1",
                "block_count": 0,
                "model_name": "m",
                "compression": "none",
                "checksum": "",
                "total_tokens": 0,
                "layer_count": 0,
                "block_size": 64,
            }
        ).encode()
        header = KVTransferHeader.from_json(data)
        assert header.version == 1


# ── Wire Format (KVTransferProtocol) ──


class TestKVTransferProtocol:
    def test_serialize_deserialize_empty(self):
        blocks = []
        data = KVTransferProtocol.serialize_blocks(blocks)
        restored = KVTransferProtocol.deserialize_blocks(data)
        assert restored == []

    def test_serialize_deserialize_single_block(self):
        block = KVBlockData(
            block_hash=0xDEADBEEF,
            token_count=64,
            layer_data={0: b"kv_data_layer_0", 1: b"kv_data_layer_1"},
        )
        data = KVTransferProtocol.serialize_blocks([block])
        restored = KVTransferProtocol.deserialize_blocks(data)

        assert len(restored) == 1
        assert restored[0].block_hash == 0xDEADBEEF
        assert restored[0].token_count == 64
        assert len(restored[0].layer_data) == 2
        assert restored[0].layer_data[0] == b"kv_data_layer_0"
        assert restored[0].layer_data[1] == b"kv_data_layer_1"

    def test_serialize_deserialize_multiple_blocks(self):
        blocks = [
            KVBlockData(
                block_hash=i,
                token_count=64,
                layer_data={0: f"block{i}_layer0".encode()},
            )
            for i in range(10)
        ]
        data = KVTransferProtocol.serialize_blocks(blocks)
        restored = KVTransferProtocol.deserialize_blocks(data)

        assert len(restored) == 10
        for i, block in enumerate(restored):
            assert block.block_hash == i
            assert block.layer_data[0] == f"block{i}_layer0".encode()

    def test_serialize_deserialize_large_batch(self):
        blocks = [
            KVBlockData(
                block_hash=hash(i),
                token_count=64,
                layer_data={j: b"\xab" * 256 for j in range(24)},
            )
            for i in range(100)
        ]
        data = KVTransferProtocol.serialize_blocks(blocks)
        restored = KVTransferProtocol.deserialize_blocks(data)

        assert len(restored) == 100
        for block in restored:
            assert block.data_size == 24 * 256

    def test_checksum_sha256(self):
        data = b"hello world"
        checksum = KVTransferProtocol.compute_checksum(data, "sha256")
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest

    def test_checksum_sha256_deterministic(self):
        data = b"test data"
        c1 = KVTransferProtocol.compute_checksum(data, "sha256")
        c2 = KVTransferProtocol.compute_checksum(data, "sha256")
        assert c1 == c2

    def test_checksum_different_data(self):
        c1 = KVTransferProtocol.compute_checksum(b"data1", "sha256")
        c2 = KVTransferProtocol.compute_checksum(b"data2", "sha256")
        assert c1 != c2

    def test_compress_decompress_none(self):
        data = b"uncompressed data"
        compressed, eff = KVTransferProtocol.compress(data, CompressionType.NONE)
        assert compressed == data
        assert eff == CompressionType.NONE

        decompressed = KVTransferProtocol.decompress(compressed, CompressionType.NONE)
        assert decompressed == data

    def test_encode_decode_roundtrip(self):
        blocks = [
            KVBlockData(
                block_hash=0x1234,
                token_count=64,
                layer_data={0: b"layer0kv", 1: b"layer1kv"},
            ),
            KVBlockData(
                block_hash=0x5678,
                token_count=32,
                layer_data={0: b"short"},
            ),
        ]
        header = KVTransferHeader(
            request_id="roundtrip-test",
            block_count=2,
            model_name="test-model",
            compression="none",
            checksum="",
            total_tokens=96,
            layer_count=2,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)

        frame = KVTransferProtocol.encode_message(
            message,
            compression=CompressionType.NONE,
            checksum_algorithm="sha256",
        )

        # Verify frame starts with magic
        assert frame[:4] == b"YKVT"

        decoded_msg, result = KVTransferProtocol.decode_message(frame)

        assert result.status == TransferStatus.COMPLETED
        assert result.checksum_verified is True
        assert result.blocks_transferred == 2
        assert decoded_msg.request_id == "roundtrip-test"
        assert len(decoded_msg.blocks) == 2
        assert decoded_msg.blocks[0].block_hash == 0x1234
        assert decoded_msg.blocks[0].layer_data[0] == b"layer0kv"
        assert decoded_msg.blocks[1].block_hash == 0x5678

    def test_encode_decode_preserves_model_name(self):
        blocks = [KVBlockData(block_hash=1, token_count=10)]
        header = KVTransferHeader(
            request_id="r1",
            block_count=1,
            model_name="qwen2.5-72b-instruct",
            compression="none",
            checksum="",
            total_tokens=10,
            layer_count=64,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)
        decoded, _ = KVTransferProtocol.decode_message(frame)
        assert decoded.header.model_name == "qwen2.5-72b-instruct"
        assert decoded.header.layer_count == 64

    def test_decode_invalid_magic(self):
        bad_frame = b"XXXX" + struct.pack("!I", 0) + struct.pack("!Q", 0)
        with pytest.raises(ValueError, match="Invalid magic"):
            KVTransferProtocol.decode_message(bad_frame)

    def test_xxhash_checksum_algorithm_roundtrips(self):
        """encode_message(checksum_algorithm='xxhash') must record 'xxhash' in the
        header so the receiver recomputes with the SAME algorithm.

        Regression: the header was rebuilt without checksum_algorithm=, so it kept
        the dataclass default 'sha256' while the digest was xxhash → the receiver
        recomputed sha256 → guaranteed CHECKSUM_MISMATCH → every block discarded
        (decode silently re-prefilled). Triggered by YUNSHU_KV_TRANSFER_CHECKSUM=xxhash.
        """
        blocks = [
            KVBlockData(
                block_hash=0xABCD, token_count=16, layer_data={0: b"payload" * 50}
            ),
        ]
        header = KVTransferHeader(
            request_id="xx-roundtrip",
            block_count=1,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=16,
            layer_count=1,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message, checksum_algorithm="xxhash")

        # The header must carry the algorithm actually used to compute the digest.
        decoded_header_only, _ = KVTransferProtocol.decode_message(frame)
        assert decoded_header_only.header.checksum_algorithm == "xxhash"

        decoded_msg, result = KVTransferProtocol.decode_message(frame)
        assert result.status == TransferStatus.COMPLETED, result.error
        assert result.checksum_verified is True
        assert decoded_msg.blocks[0].layer_data[0] == b"payload" * 50

    def test_checksum_mismatch_detected(self):
        blocks = [KVBlockData(block_hash=1, token_count=10, layer_data={0: b"kv"})]
        header = KVTransferHeader(
            request_id="mismatch",
            block_count=1,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=10,
            layer_count=1,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)

        # Corrupt a byte in the payload (after header area)
        corrupted = bytearray(frame)
        corrupted[-1] ^= 0xFF  # Flip last byte
        corrupted_frame = bytes(corrupted)

        _, result = KVTransferProtocol.decode_message(corrupted_frame)
        assert result.status == TransferStatus.CHECKSUM_MISMATCH
        assert "Checksum mismatch" in result.error

    def test_encode_decode_empty_blocks(self):
        header = KVTransferHeader(
            request_id="empty",
            block_count=0,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=0,
            layer_count=0,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=[])
        frame = KVTransferProtocol.encode_message(message)
        decoded, result = KVTransferProtocol.decode_message(frame)

        assert result.status == TransferStatus.COMPLETED
        assert decoded.blocks == []

    def test_read_frame_from_stream(self):
        blocks = [KVBlockData(block_hash=42, token_count=64, layer_data={0: b"data"})]
        header = KVTransferHeader(
            request_id="stream-test",
            block_count=1,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=64,
            layer_count=1,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(message)

        async def _test():
            # Simulate a stream reader
            reader = asyncio.StreamReader()
            reader.feed_data(frame)
            reader.feed_eof()

            read_frame = await KVTransferProtocol.read_frame(reader)
            assert read_frame == frame

        asyncio.run(_test())


# ── Compression (optional) ──


class TestCompression:
    def test_zstd_roundtrip_if_available(self):
        pytest.importorskip("zstandard")

        data = b"compress this data " * 100
        compressed, eff = KVTransferProtocol.compress(data, CompressionType.ZSTD)
        assert len(compressed) < len(data)
        assert eff == CompressionType.ZSTD

        decompressed = KVTransferProtocol.decompress(compressed, CompressionType.ZSTD)
        assert decompressed == data

    def test_lz4_roundtrip_if_available(self):
        pytest.importorskip("lz4.frame")

        data = b"compress this data " * 100
        compressed, eff = KVTransferProtocol.compress(data, CompressionType.LZ4)
        assert len(compressed) < len(data)
        assert eff == CompressionType.LZ4

        decompressed = KVTransferProtocol.decompress(compressed, CompressionType.LZ4)
        assert decompressed == data

    def test_zstd_encode_decode_full(self):
        pytest.importorskip("zstandard")

        blocks = [
            KVBlockData(
                block_hash=i,
                token_count=64,
                layer_data={0: b"x" * 2048},
            )
            for i in range(10)
        ]
        header = KVTransferHeader(
            request_id="zstd-test",
            block_count=10,
            model_name="m",
            compression="zstd",
            checksum="",
            total_tokens=640,
            layer_count=1,
            block_size=64,
        )
        message = KVTransferMessage(header=header, blocks=blocks)
        frame = KVTransferProtocol.encode_message(
            message,
            compression=CompressionType.ZSTD,
        )
        decoded, result = KVTransferProtocol.decode_message(frame)

        assert result.status == TransferStatus.COMPLETED
        assert decoded.header.compression == "zstd"
        assert len(decoded.blocks) == 10


# ── KVTransferResult ──


class TestKVTransferResult:
    def test_compression_ratio_no_compression(self):
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.COMPLETED,
            bytes_transferred=1000,
            bytes_original=1000,
        )
        assert result.compression_ratio == 1.0

    def test_compression_ratio_with_compression(self):
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.COMPLETED,
            bytes_transferred=500,
            bytes_original=1000,
        )
        assert result.compression_ratio == 0.5

    def test_compression_ratio_zero_original(self):
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.COMPLETED,
            bytes_transferred=0,
            bytes_original=0,
        )
        assert result.compression_ratio == 1.0

    def test_failed_result(self):
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.FAILED,
            error="Connection refused",
        )
        assert result.status == TransferStatus.FAILED
        assert result.error == "Connection refused"


# ── KVTransferStats ──


class TestKVTransferStats:
    def test_empty_stats(self):
        stats = KVTransferStats()
        assert stats.total_transfers == 0
        assert stats.avg_send_latency_ms == 0.0
        assert stats.avg_receive_latency_ms == 0.0
        assert stats.avg_compression_ratio == 1.0

    def test_record_send(self):
        stats = KVTransferStats()
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.COMPLETED,
            blocks_transferred=5,
            bytes_transferred=1000,
            bytes_original=2000,
            duration_seconds=0.1,
        )
        stats.record_send(result)
        assert stats.total_transfers == 1
        assert stats.total_blocks_sent == 5
        assert stats.total_bytes_sent == 1000
        assert stats.total_bytes_original_sent == 2000
        assert stats.avg_send_latency_ms == pytest.approx(100.0)
        assert stats.avg_compression_ratio == 0.5

    def test_record_receive(self):
        stats = KVTransferStats()
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.COMPLETED,
            blocks_transferred=3,
            bytes_transferred=500,
            duration_seconds=0.05,
        )
        stats.record_receive(result)
        assert stats.total_transfers == 1
        assert stats.total_blocks_received == 3
        assert stats.total_bytes_received == 500

    def test_record_receive_checksum_failure(self):
        stats = KVTransferStats()
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.CHECKSUM_MISMATCH,
            error="bad checksum",
        )
        stats.record_receive(result)
        assert stats.total_checksum_failures == 1

    def test_record_receive_failure(self):
        stats = KVTransferStats()
        result = KVTransferResult(
            request_id="r1",
            status=TransferStatus.FAILED,
            error="timeout",
        )
        stats.record_receive(result)
        assert stats.total_transfer_failures == 1

    def test_latency_list_trimming(self):
        stats = KVTransferStats()
        for i in range(1001):
            stats.record_send(
                KVTransferResult(
                    request_id=f"r{i}",
                    status=TransferStatus.COMPLETED,
                    blocks_transferred=1,
                    bytes_transferred=100,
                    duration_seconds=0.001,
                )
            )
        assert len(stats.send_latencies) == 500  # trimmed to last 500

    def test_to_dict(self):
        stats = KVTransferStats()
        stats.record_send(
            KVTransferResult(
                request_id="r1",
                status=TransferStatus.COMPLETED,
                blocks_transferred=1,
                bytes_transferred=100,
                bytes_original=100,
                duration_seconds=0.1,
            )
        )
        d = stats.to_dict()
        assert d["total_transfers"] == 1
        assert d["total_blocks_sent"] == 1
        assert "avg_send_latency_ms" in d
        assert "avg_receive_latency_ms" in d
        assert "avg_compression_ratio" in d


# ── KVTransferClient ──


class TestKVTransferClient:
    def test_client_disabled_returns_failure(self):
        config = KVTransferConfig(enabled=False)
        client = KVTransferClient(config)
        result = asyncio.run(
            client.send_blocks(
                blocks=[KVBlockData(block_hash=1, token_count=10)],
                request_id="test",
            )
        )
        assert result.status == TransferStatus.FAILED
        assert "not enabled" in result.error

    def test_client_stats_initial(self):
        client = KVTransferClient()
        assert client.stats.total_transfers == 0

    def test_client_auto_request_id(self):
        config = KVTransferConfig(enabled=True)
        client = KVTransferClient(config)
        result = asyncio.run(
            client.send_blocks(
                blocks=[KVBlockData(block_hash=1, token_count=10)],
            )
        )
        # When enabled, auto-generated ID should be set
        assert result.request_id  # non-empty


# ── KVTransferServer ──


class TestKVTransferServer:
    def test_server_initial_state(self):
        server = KVTransferServer()
        assert server.stats.total_transfers == 0

    def test_server_set_kv_manager(self):
        server = KVTransferServer()
        mock_mgr = MagicMock()
        server.set_kv_manager(mock_mgr)
        assert server._kv_manager is mock_mgr

    def test_server_disabled_start(self):
        config = KVTransferConfig(enabled=False)
        server = KVTransferServer(config)
        asyncio.run(server.start())
        assert server._server is None

    def test_server_get_stats(self):
        server = KVTransferServer()
        stats = server.get_stats()
        assert stats["enabled"] is False
        assert stats["listen_port"] == 7890
        assert stats["active_transfers"] == 0

    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        config = KVTransferConfig(enabled=True, listen_port=0)  # port 0 = OS picks
        server = KVTransferServer(config)
        await server.start()
        assert server._server is not None
        await server.stop()
        assert server._server is None


# ── Client/Server Integration ──


class TestClientServerIntegration:
    @pytest.mark.asyncio
    async def test_send_receive_roundtrip(self):
        """Full client->server transfer round-trip."""
        # Start server on random port
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()

        # Get the assigned port
        addr = server._server.sockets[0].getsockname()
        port = addr[1]

        # Configure client to connect to server
        client_config = KVTransferConfig(
            enabled=True,
            remote_host="127.0.0.1",
            remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            # Send blocks
            blocks = [
                KVBlockData(
                    block_hash=0xAAAA,
                    token_count=64,
                    layer_data={0: b"kv_layer_0", 1: b"kv_layer_1"},
                ),
                KVBlockData(
                    block_hash=0xBBBB,
                    token_count=32,
                    layer_data={0: b"short"},
                ),
            ]
            result = await client.send_blocks(
                blocks=blocks,
                request_id="integ-test",
                model_name="test-model",
                total_tokens=96,
                layer_count=2,
            )

            assert result.status == TransferStatus.COMPLETED
            assert result.blocks_transferred == 2
            assert result.bytes_transferred > 0
            assert result.checksum_verified is True

            # Check server received
            assert server.stats.total_blocks_received == 2
            assert server.stats.total_bytes_received > 0

        finally:
            await client.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_send_empty_blocks(self):
        """Sending empty block list."""
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True,
            remote_host="127.0.0.1",
            remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            result = await client.send_blocks(
                blocks=[],
                request_id="empty-test",
                model_name="m",
            )
            assert result.status == TransferStatus.COMPLETED
        finally:
            await client.stop()
            await server.stop()

    @pytest.mark.asyncio
    async def test_send_large_batch(self):
        """Sending many blocks."""
        server_config = KVTransferConfig(enabled=True, listen_port=0)
        server = KVTransferServer(server_config)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_config = KVTransferConfig(
            enabled=True,
            remote_host="127.0.0.1",
            remote_port=port,
        )
        client = KVTransferClient(client_config)
        await client.start()

        try:
            blocks = [
                KVBlockData(
                    block_hash=i,
                    token_count=64,
                    layer_data={j: f"L{i}_{j}".encode() for j in range(4)},
                )
                for i in range(50)
            ]
            result = await client.send_blocks(
                blocks=blocks,
                request_id="batch-test",
                model_name="test",
                total_tokens=50 * 64,
                layer_count=4,
            )
            assert result.status == TransferStatus.COMPLETED
            assert result.blocks_transferred == 50
        finally:
            await client.stop()
            await server.stop()


# ── extract_kv_blocks_from_cache ──


class TestExtractKVBlocks:
    def test_none_cache_returns_empty(self):
        blocks = extract_kv_blocks_from_cache(None, [1, 2, 3])
        assert blocks == []

    def test_empty_tokens_returns_empty(self):
        blocks = extract_kv_blocks_from_cache(["fake_cache"], [])
        assert blocks == []

    def test_blocks_correct_count(self):
        # 128 tokens with block_size=64 → 2 blocks
        tokens = list(range(128))
        blocks = extract_kv_blocks_from_cache([], tokens, block_size=64)
        assert len(blocks) == 2
        assert blocks[0].token_count == 64
        assert blocks[1].token_count == 64

    def test_blocks_partial_last_block(self):
        # 100 tokens with block_size=64 → 2 blocks (64 + 36)
        tokens = list(range(100))
        blocks = extract_kv_blocks_from_cache([], tokens, block_size=64)
        assert len(blocks) == 2
        assert blocks[0].token_count == 64
        assert blocks[1].token_count == 36

    def test_block_hashes_differ(self):
        tokens = list(range(128))
        blocks = extract_kv_blocks_from_cache([], tokens, block_size=64)
        assert blocks[0].block_hash != blocks[1].block_hash

    def test_block_hashes_deterministic(self):
        tokens = list(range(64))
        b1 = extract_kv_blocks_from_cache([], tokens, block_size=64)
        b2 = extract_kv_blocks_from_cache([], tokens, block_size=64)
        assert b1[0].block_hash == b2[0].block_hash

    def test_single_token_block(self):
        blocks = extract_kv_blocks_from_cache([], [42], block_size=64)
        assert len(blocks) == 1
        assert blocks[0].token_count == 1


# ── load_kv_blocks_into_cache ──


class TestLoadKVBlocks:
    def test_none_cache_returns_zero(self):
        loaded = load_kv_blocks_into_cache(None, [MagicMock()])
        assert loaded == 0

    def test_empty_blocks_returns_zero(self):
        loaded = load_kv_blocks_into_cache([], [])
        assert loaded == 0


# ── Convenience Functions ──


class TestConvenience:
    def test_is_kv_transfer_enabled_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_kv_transfer_enabled() is False

    def test_is_kv_transfer_enabled_set(self):
        with patch.dict(os.environ, {"YUNSHU_KV_TRANSFER": "1"}):
            assert is_kv_transfer_enabled() is True

    def test_create_transfer_client(self):
        config = KVTransferConfig(enabled=False)
        client = create_transfer_client(config)
        assert isinstance(client, KVTransferClient)

    def test_create_transfer_server(self):
        config = KVTransferConfig(enabled=False)
        server = create_transfer_server(config)
        assert isinstance(server, KVTransferServer)


# ── ExternalPrefiller Integration ──


class TestExternalPrefillerTransfer:
    def test_transfer_disabled_returns_none(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        result = PrefillResult(
            token_ids=[1, 2, 3],
            num_tokens=3,
            kv_cache=["fake"],
        )
        with patch.dict(os.environ, {"YUNSHU_KV_TRANSFER": "0"}):
            transfer_result = prefiller.transfer_prefill_result(result)
        assert transfer_result is None

    def test_transfer_no_kv_cache_returns_none(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        result = PrefillResult(
            token_ids=[1, 2, 3],
            num_tokens=3,
            kv_cache=None,
        )
        with patch.dict(os.environ, {"YUNSHU_KV_TRANSFER": "1"}):
            transfer_result = prefiller.transfer_prefill_result(result)
        assert transfer_result is None

    def test_transfer_enabled_with_kv_cache(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        result = PrefillResult(
            token_ids=list(range(64)),
            num_tokens=64,
            kv_cache=[],  # empty list (no layers, but not None)
        )

        mock_transfer_result = KVTransferResult(
            request_id="test-123",
            status=TransferStatus.COMPLETED,
            blocks_transferred=1,
            bytes_transferred=100,
        )

        with (
            patch.dict(os.environ, {"YUNSHU_KV_TRANSFER": "1"}),
            patch("yunshu_engine.kv_transfer.KVTransferClient") as MockClient,
        ):
            mock_instance = MagicMock()
            mock_instance.send_blocks_sync.return_value = mock_transfer_result
            MockClient.return_value = mock_instance

            transfer_result = prefiller.transfer_prefill_result(
                result,
                request_id="req-test",
                model_name="test-model",
                layer_count=24,
            )

        assert transfer_result is not None
        assert transfer_result.status == TransferStatus.COMPLETED

    def test_transfer_handles_exception(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        result = PrefillResult(
            token_ids=list(range(64)),
            num_tokens=64,
            kv_cache=[],
        )

        with (
            patch.dict(os.environ, {"YUNSHU_KV_TRANSFER": "1"}),
            patch("yunshu_engine.kv_transfer.KVTransferClient") as MockClient,
        ):
            MockClient.side_effect = RuntimeError("connection failed")

            transfer_result = prefiller.transfer_prefill_result(result)

        assert transfer_result is None  # Exception swallowed, returns None


# ── Enums ──


class TestEnums:
    def test_compression_types(self):
        assert CompressionType.NONE.value == "none"
        assert CompressionType.ZSTD.value == "zstd"
        assert CompressionType.LZ4.value == "lz4"

    def test_transfer_statuses(self):
        assert TransferStatus.PENDING.value == "pending"
        assert TransferStatus.IN_PROGRESS.value == "in_progress"
        assert TransferStatus.COMPLETED.value == "completed"
        assert TransferStatus.FAILED.value == "failed"
        assert TransferStatus.CHECKSUM_MISMATCH.value == "checksum_mismatch"
        assert TransferStatus.CANCELLED.value == "cancelled"


# ── Compression fallback tests ──


class TestCompressionFallback:
    def test_compress_zstd_fallback_without_library(self):
        with patch.dict("sys.modules", {"zstandard": None}):
            data = b"test data"
            result, eff = KVTransferProtocol.compress(data, CompressionType.ZSTD)
            # Falls back to uncompressed AND reports NONE so the receiver
            # doesn't try to decompress raw bytes .
            assert result == data
            assert eff == CompressionType.NONE

    def test_decompress_zstd_raises_without_library(self):
        with patch.dict("sys.modules", {"zstandard": None}):
            with pytest.raises(RuntimeError, match="zstandard required"):
                KVTransferProtocol.decompress(b"fake", CompressionType.ZSTD)

    def test_compress_lz4_fallback_without_library(self):
        with patch.dict("sys.modules", {"lz4": None, "lz4.frame": None}):
            data = b"test data"
            result, eff = KVTransferProtocol.compress(data, CompressionType.LZ4)
            assert result == data
            assert eff == CompressionType.NONE

    def test_decompress_lz4_raises_without_library(self):
        with patch.dict("sys.modules", {"lz4": None, "lz4.frame": None}):
            with pytest.raises(RuntimeError, match="lz4 required"):
                KVTransferProtocol.decompress(b"fake", CompressionType.LZ4)


# ── KVTransferMessage ──


class TestKVTransferMessage:
    def test_request_id(self):
        header = KVTransferHeader(
            request_id="msg-test",
            block_count=0,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=0,
            layer_count=0,
            block_size=64,
        )
        msg = KVTransferMessage(header=header, blocks=[])
        assert msg.request_id == "msg-test"

    def test_total_data_size_empty(self):
        header = KVTransferHeader(
            request_id="r",
            block_count=0,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=0,
            layer_count=0,
            block_size=64,
        )
        msg = KVTransferMessage(header=header, blocks=[])
        assert msg.total_data_size == 0

    def test_total_data_size_with_blocks(self):
        header = KVTransferHeader(
            request_id="r",
            block_count=2,
            model_name="m",
            compression="none",
            checksum="",
            total_tokens=128,
            layer_count=1,
            block_size=64,
        )
        blocks = [
            KVBlockData(block_hash=1, token_count=64, layer_data={0: b"a" * 100}),
            KVBlockData(block_hash=2, token_count=64, layer_data={0: b"b" * 200}),
        ]
        msg = KVTransferMessage(header=header, blocks=blocks)
        assert msg.total_data_size == 300


# ── Tensor serialization round-trip ──


class TestTensorSerializationRoundTrip:
    """Guards the dtype+shape-preserving self-describing tensor format.

    The old code hardcoded float16 on read while writing the source dtype,
    silently corrupting bf16/fp32 KV caches (wrong values + 2x element count
    + lost geometry). These tests pin the exact round-trip.
    """

    def _mx(self):
        mx = pytest.importorskip("mlx.core")
        return mx

    @pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
    def test_kv_pair_roundtrip_preserves_dtype_and_shape(self, dtype_name):
        mx = self._mx()
        from yunshu_engine.kv_transfer import _read_tensor, _tensor_to_bytes

        dt = getattr(mx, dtype_name)
        k = (mx.random.normal((5, 2, 8)) * 3).astype(dt)
        v = (mx.random.normal((5, 2, 8)) * 3).astype(dt)
        # extract stores k_bytes + v_bytes concatenated.
        payload = _tensor_to_bytes(k) + _tensor_to_bytes(v)
        kr, off = _read_tensor(payload, 0)
        vr, _ = _read_tensor(payload, off)
        assert kr.shape == (5, 2, 8)
        assert vr.shape == (5, 2, 8)
        # The original mlx dtype MUST survive the round-trip — bf16 is transported as
        # float32 (numpy has no bfloat16) but must come back as bf16, else it mismatches
        # the model's bf16 weights on every disaggregated handoff.
        assert kr.dtype == dt, f"k dtype {kr.dtype} != original {dt}"
        assert vr.dtype == dt, f"v dtype {vr.dtype} != original {dt}"
        assert mx.allclose(
            kr.astype(mx.float32), k.astype(mx.float32), atol=1e-2
        ).item()
        assert mx.allclose(
            vr.astype(mx.float32), v.astype(mx.float32), atol=1e-2
        ).item()

    def test_load_blocks_separates_k_and_v_into_list_cache(self):
        mx = self._mx()
        from yunshu_engine.kv_transfer import (
            _tensor_to_bytes,
            load_kv_blocks_into_cache,
        )

        # real mlx-lm KV tensors are 4D
        # (batch, n_kv_heads, seq, head_dim) — the SEQUENCE axis is -2, not 0. Use that
        # geometry (the old test used a fictional 3D (seq,heads,dim) shape that masked
        # the axis-0 extract/load bug).
        k = mx.ones((1, 2, 3, 4)).astype(mx.float32)
        v = (mx.ones((1, 2, 3, 4)) * 2).astype(mx.float32)
        block = KVBlockData(
            block_hash=1,
            token_count=3,
            layer_data={0: _tensor_to_bytes(k) + _tensor_to_bytes(v)},
        )
        # list-style layer cache: [keys, values], each starting empty along seq (-2).
        layer_cache = [mx.zeros((1, 2, 0, 4)), mx.zeros((1, 2, 0, 4))]
        kv_cache = [layer_cache]
        loaded = load_kv_blocks_into_cache(kv_cache, [block])
        assert loaded == 1
        # K and V must be appended to their OWN slots along the seq axis (-2).
        assert kv_cache[0][0].shape == (1, 2, 3, 4)
        assert kv_cache[0][1].shape == (1, 2, 3, 4)
        assert mx.allclose(kv_cache[0][0], k).item()  # keys are 1.0
        assert mx.allclose(kv_cache[0][1], v).item()  # values are 2.0
