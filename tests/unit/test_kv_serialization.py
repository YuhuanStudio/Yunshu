"""Tests for KV cache serialization — block, table, and file round-trips."""

import os
import tempfile

import numpy as np
import pytest

from yunshu_kv.block import BlockPool, KVBlock
from yunshu_kv.block_table import BlockTable
from yunshu_kv.manager import KVCacheConfig, KVCacheManager
from yunshu_kv.serialization import MAGIC, VERSION, KVCacheSerializer

# ── Helpers ────────────────────────────────────────────────────────


def _make_block(block_id: int, block_hash=None, ref_count: int = 0) -> KVBlock:
    return KVBlock(block_id=block_id, ref_count=ref_count, block_hash=block_hash)


def _make_kv_arrays(num_blocks: int, layers: int = 2, heads: int = 4, dim: int = 8):
    """Create synthetic key/value numpy arrays shaped [num_blocks, layers, heads, dim]."""
    rng = np.random.default_rng(42)
    key = rng.standard_normal((num_blocks, layers, heads, dim)).astype(np.float16)
    val = rng.standard_normal((num_blocks, layers, heads, dim)).astype(np.float16)
    return key, val


def _make_table_and_cache(num_blocks=5, block_size=16, layers=2, heads=4, dim=8):
    """Build a BlockTable with blocks + matching key/value numpy arrays.

    Arrays are sized to cover block_ids 0..max_block_id (block 0 is the null
    block, so we need num_blocks+1 rows).
    """
    pool = BlockPool(num_blocks=num_blocks + 5, block_size=block_size)
    blocks = pool.allocate(num_blocks)
    table = BlockTable(block_size=block_size)
    table.append_blocks(blocks)

    # Block IDs start at 1 (block 0 is null), so we need at least max_id+1 rows.
    max_id = max(b.block_id for b in blocks)
    key, val = _make_kv_arrays(max_id + 1, layers, heads, dim)
    return table, key, val


# ── Single block round-trip ────────────────────────────────────────


class TestSerializeBlock:
    def test_round_trip_basic(self):
        """Serialize then deserialize a block — metadata and data must match."""
        serializer = KVCacheSerializer()
        block = _make_block(block_id=7, block_hash=12345678, ref_count=3)
        key, val = _make_kv_arrays(1)

        data = serializer.serialize_block(block, key[0], val[0])
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored_block, restored_key, restored_val = serializer.deserialize_block(data)

        # Metadata
        assert restored_block.block_id == 7
        assert restored_block.block_hash == 12345678
        assert restored_block.ref_count == 3

        # Data
        np.testing.assert_array_equal(np.array(restored_key), key[0])
        np.testing.assert_array_equal(np.array(restored_val), val[0])

    def test_round_trip_no_hash(self):
        """Block with block_hash=None should round-trip correctly."""
        serializer = KVCacheSerializer()
        block = _make_block(block_id=3, block_hash=None, ref_count=1)
        key, val = _make_kv_arrays(1)

        data = serializer.serialize_block(block, key[0], val[0])
        restored_block, _, _ = serializer.deserialize_block(data)

        assert restored_block.block_hash is None
        assert restored_block.block_id == 3

    def test_different_dtypes(self):
        """Round-trip with float32 arrays."""
        serializer = KVCacheSerializer()
        block = _make_block(block_id=0)
        key = np.ones((4, 8), dtype=np.float32)
        val = np.zeros((4, 8), dtype=np.float32)

        data = serializer.serialize_block(block, key, val)
        _, rk, rv = serializer.deserialize_block(data)

        np.testing.assert_array_equal(np.array(rk), key)
        np.testing.assert_array_equal(np.array(rv), val)

    def test_large_block(self):
        """Round-trip with a realistically shaped tensor."""
        serializer = KVCacheSerializer()
        block = _make_block(block_id=0, block_hash=9999, ref_count=1)
        # Shape similar to a real KV slice: [layers, heads, head_dim]
        key = np.random.randn(32, 8, 128).astype(np.float16)
        val = np.random.randn(32, 8, 128).astype(np.float16)

        data = serializer.serialize_block(block, key, val)
        _, rk, rv = serializer.deserialize_block(data)

        np.testing.assert_array_equal(np.array(rk), key)
        np.testing.assert_array_equal(np.array(rv), val)


# ── Full table round-trip ──────────────────────────────────────────


class TestSerializeTable:
    def test_round_trip_basic(self):
        """Serialize and deserialize a table with multiple blocks."""
        serializer = KVCacheSerializer()
        table, key, val = _make_table_and_cache(num_blocks=3)

        data = serializer.serialize_table(table, key, val)
        assert isinstance(data, bytes)

        restored_table, restored_key, restored_val = serializer.deserialize_table(data)

        assert restored_table.num_blocks == 3
        assert restored_table.block_size == 16

        # Verify each block's data matches the original slice
        for orig, rest in zip(table.get_blocks(), restored_table.get_blocks(), strict=False):
            np.testing.assert_array_equal(
                np.array(restored_key)[rest.block_id], key[orig.block_id]
            )
            np.testing.assert_array_equal(
                np.array(restored_val)[rest.block_id], val[orig.block_id]
            )

        # Block IDs preserved
        original_ids = [b.block_id for b in table.get_blocks()]
        restored_ids = [b.block_id for b in restored_table.get_blocks()]
        assert restored_ids == original_ids

    def test_round_trip_preserves_hashes(self):
        """Block hashes are preserved through serialization."""
        serializer = KVCacheSerializer()
        pool = BlockPool(num_blocks=20, block_size=16)
        blocks = pool.allocate(3)
        pool.cache_block(blocks[0], 111)
        pool.cache_block(blocks[1], 222)
        # blocks[2] has no hash

        table = BlockTable(block_size=16)
        table.append_blocks(blocks)

        key, val = _make_kv_arrays(3)
        data = serializer.serialize_table(table, key, val)
        restored_table, _, _ = serializer.deserialize_table(data)

        restored_blocks = restored_table.get_blocks()
        assert restored_blocks[0].block_hash == 111
        assert restored_blocks[1].block_hash == 222
        assert restored_blocks[2].block_hash is None

    def test_empty_table(self):
        """Serializing an empty table should round-trip."""
        serializer = KVCacheSerializer()
        table = BlockTable(block_size=32)
        key = np.zeros((0, 2, 4, 8), dtype=np.float16)
        val = np.zeros((0, 2, 4, 8), dtype=np.float16)

        data = serializer.serialize_table(table, key, val)
        restored_table, restored_key, restored_val = serializer.deserialize_table(data)

        assert restored_table.num_blocks == 0
        assert restored_table.block_size == 32

    def test_single_block_table(self):
        """Single block table round-trip."""
        serializer = KVCacheSerializer()
        table, key, val = _make_table_and_cache(num_blocks=1)

        data = serializer.serialize_table(table, key, val)
        rt_table, rt_key, rt_val = serializer.deserialize_table(data)

        assert rt_table.num_blocks == 1
        orig_block = table.get_block(0)
        rest_block = rt_table.get_block(0)
        np.testing.assert_array_equal(
            np.array(rt_key)[rest_block.block_id], key[orig_block.block_id]
        )
        np.testing.assert_array_equal(
            np.array(rt_val)[rest_block.block_id], val[orig_block.block_id]
        )


# ── File save/load ─────────────────────────────────────────────────


class TestFileIO:
    def test_save_load_round_trip(self):
        """Write table to file and read it back."""
        serializer = KVCacheSerializer()
        table, key, val = _make_table_and_cache(num_blocks=4)

        with tempfile.NamedTemporaryFile(suffix=".yskv", delete=False) as f:
            path = f.name

        try:
            serializer.save_to_file(path, table, key, val)
            assert os.path.getsize(path) > 0

            rt_table, rt_key, rt_val = serializer.load_from_file(path)

            assert rt_table.num_blocks == 4
            assert rt_table.block_size == 16
            for orig, rest in zip(table.get_blocks(), rt_table.get_blocks(), strict=False):
                np.testing.assert_array_equal(
                    np.array(rt_key)[rest.block_id], key[orig.block_id]
                )
        finally:
            os.unlink(path)

    def test_file_header_magic(self):
        """Serialized file starts with correct magic number."""
        import struct

        serializer = KVCacheSerializer()
        table, key, val = _make_table_and_cache(num_blocks=1)

        with tempfile.NamedTemporaryFile(suffix=".yskv", delete=False) as f:
            path = f.name

        try:
            serializer.save_to_file(path, table, key, val)
            with open(path, "rb") as f:
                header = f.read(6)
            magic, version = struct.unpack(">IH", header)
            assert magic == MAGIC
            assert version == VERSION
        finally:
            os.unlink(path)

    def test_invalid_magic_rejected(self):
        """Deserializing garbage bytes raises ValueError."""
        serializer = KVCacheSerializer()
        with pytest.raises(ValueError, match="Invalid magic"):
            serializer.deserialize_table(b"\x00" * 64)


# ── Compression modes ──────────────────────────────────────────────


class TestCompressionModes:
    def test_numpy_compression_round_trip(self):
        """numpy compression mode round-trips correctly."""
        serializer = KVCacheSerializer(compression="numpy")
        table, key, val = _make_table_and_cache(num_blocks=3)

        data = serializer.serialize_table(table, key, val)
        rt_table, rt_key, rt_val = serializer.deserialize_table(data)

        assert rt_table.num_blocks == 3
        for orig, rest in zip(table.get_blocks(), rt_table.get_blocks(), strict=False):
            np.testing.assert_array_equal(
                np.array(rt_key)[rest.block_id], key[orig.block_id]
            )
            np.testing.assert_array_equal(
                np.array(rt_val)[rest.block_id], val[orig.block_id]
            )

    def test_numpy_compression_smaller(self):
        """numpy compression should produce smaller output for compressible data."""
        ser_none = KVCacheSerializer(compression="none")
        ser_np = KVCacheSerializer(compression="numpy")

        table, key, val = _make_table_and_cache(num_blocks=5)
        # Use constant data — highly compressible
        key[:] = 1.0
        val[:] = 2.0

        raw = ser_none.serialize_table(table, key, val)
        compressed = ser_np.serialize_table(table, key, val)

        assert len(compressed) < len(raw)

    def test_invalid_compression_raises(self):
        with pytest.raises(ValueError, match="Unsupported compression"):
            KVCacheSerializer(compression="brotli")

    def test_safetensors_supported(self):
        s = KVCacheSerializer(compression="safetensors")
        assert s.compression == "safetensors"


# ── KVCacheManager integration ─────────────────────────────────────


class TestKVCacheManagerSerialization:
    def _make_manager_with_tensors(self, num_blocks=20, block_size=16):
        config = KVCacheConfig(
            block_size=block_size,
            num_layers=2,
            num_kv_heads=4,
            head_dim=8,
            enable_caching=True,
        )
        mgr = KVCacheManager(config, num_blocks=num_blocks)
        # Set up KV cache tensors (numpy for testing)
        key, val = _make_kv_arrays(num_blocks - 1, layers=2, heads=4, dim=8)
        mgr.set_kv_tensors(key, val)
        return mgr, key, val

    def test_save_prefix_and_load(self):
        """Save a prefix block and load it back."""
        mgr, key, val = self._make_manager_with_tensors()

        # Manually cache a block
        block = mgr.block_pool.blocks[1]
        mgr.block_pool.cache_block(block, 0xABCD)
        block.ref_count = 1

        with tempfile.NamedTemporaryFile(suffix=".blk", delete=False) as f:
            path = f.name

        try:
            mgr.save_prefix(0xABCD, path)
            assert os.path.getsize(path) > 0

            loaded = mgr.load_prefix(path)
            assert loaded.block_hash == 0xABCD
        finally:
            os.unlink(path)

    def test_save_prefix_missing_raises(self):
        """Saving a non-existent prefix hash raises KeyError."""
        mgr, _, _ = self._make_manager_with_tensors()

        with tempfile.NamedTemporaryFile(suffix=".blk", delete=False) as f:
            path = f.name
        os.unlink(path)

        with pytest.raises(KeyError):
            mgr.save_prefix(0xDEAD, path)

    def test_save_all_cached_and_load(self):
        """Save all cached blocks and reload."""
        mgr, key, val = self._make_manager_with_tensors()

        # Cache several blocks
        mgr.block_pool.cache_block(mgr.block_pool.blocks[1], 111)
        mgr.block_pool.cache_block(mgr.block_pool.blocks[2], 222)
        mgr.block_pool.cache_block(mgr.block_pool.blocks[3], 333)

        with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as f:
            path = f.name

        try:
            mgr.save_all_cached(path)

            # Clear prefix cache
            mgr.block_pool.reset_prefix_cache()
            assert mgr.block_pool.lookup_hash(111) is None

            # Load
            num_loaded = mgr.load_cached(path)
            assert num_loaded == 3

            # Prefix cache restored
            assert mgr.block_pool.lookup_hash(111) is not None
            assert mgr.block_pool.lookup_hash(222) is not None
            assert mgr.block_pool.lookup_hash(333) is not None
        finally:
            os.unlink(path)

    def test_save_all_cached_empty(self):
        """Saving empty prefix cache writes a valid file with count=0."""
        mgr, _, _ = self._make_manager_with_tensors()

        with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as f:
            path = f.name

        try:
            mgr.save_all_cached(path)
            num_loaded = mgr.load_cached(path)
            assert num_loaded == 0
        finally:
            os.unlink(path)


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_multiple_serialize_cycles(self):
        """Data survives multiple serialize/deserialize cycles."""
        serializer = KVCacheSerializer()
        block = _make_block(block_id=42, block_hash=98765, ref_count=5)
        key, val = _make_kv_arrays(1)

        data = serializer.serialize_block(block, key[0], val[0])

        for _ in range(5):
            b, k, v = serializer.deserialize_block(data)
            data = serializer.serialize_block(b, k, v)

        final_b, final_k, final_v = serializer.deserialize_block(data)
        assert final_b.block_id == 42
        assert final_b.block_hash == 98765
        assert final_b.ref_count == 5
        np.testing.assert_array_equal(np.array(final_k), key[0])

    def test_zero_ref_count(self):
        """Block with ref_count=0 round-trips."""
        serializer = KVCacheSerializer()
        block = _make_block(block_id=0, ref_count=0, block_hash=None)
        key = np.array([[1.0, 2.0]], dtype=np.float16)
        val = np.array([[3.0, 4.0]], dtype=np.float16)

        data = serializer.serialize_block(block, key, val)
        rb, _, _ = serializer.deserialize_block(data)
        assert rb.ref_count == 0
        assert rb.block_hash is None

    def test_metadata_preservation_under_numpy_compression(self):
        """Metadata survives numpy compression mode."""
        serializer = KVCacheSerializer(compression="numpy")
        block = _make_block(block_id=99, block_hash=0xCAFEBABE, ref_count=7)
        key, val = _make_kv_arrays(1)

        data = serializer.serialize_block(block, key[0], val[0])
        rb, rk, rv = serializer.deserialize_block(data)

        assert rb.block_id == 99
        assert rb.block_hash == 0xCAFEBABE
        assert rb.ref_count == 7

    def test_bfloat16_block_roundtrip(self):
        """an MLX bfloat16 KV block round-trips losslessly and restores its
        bf16 dtype. numpy has no bf16, so np.array() on a bf16 MLX array raised —
        save_prefix/export crashed for every bf16 model before the fix."""
        mx = pytest.importorskip("mlx.core")
        serializer = KVCacheSerializer()
        block = _make_block(block_id=2, block_hash=12345, ref_count=1)
        key = mx.random.normal((4, 8, 16)).astype(mx.bfloat16)
        val = mx.random.normal((4, 8, 16)).astype(mx.bfloat16)

        data = serializer.serialize_block(block, key, val)
        rb, rk, rv = serializer.deserialize_block(data)

        assert rb.block_id == 2 and rb.block_hash == 12345
        assert rk.dtype == mx.bfloat16 and rv.dtype == mx.bfloat16
        # bf16 -> float32 -> bf16 is exact (float32 represents every bf16 value)
        assert mx.array_equal(key, rk) and mx.array_equal(val, rv)
