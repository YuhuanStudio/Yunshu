"""Parallel PLE row reads are byte-identical to upstream's, including cache and counters."""

import json

import numpy as np
import pytest

ps = pytest.importorskip("mlx_vlm.models.qwen4_exp.ple_storage")
from yunshu_engine import ple_parallel  # noqa: E402

ROW_WIDTH = 160
GROUP = 32
SHARD_ROWS = (7, 5, 9)


def _build(tmp_path, cache_rows):
    rng = np.random.default_rng(0)
    entries, start = [], 0
    for index, count in enumerate(SHARD_ROWS):
        entry = {"row_start": start, "row_count": count}
        for name, dtype, width in (
            ("weight", "<u4", ROW_WIDTH // 8),
            ("scales", "<u2", ROW_WIDTH // GROUP),
            ("biases", "<u2", ROW_WIDTH // GROUP),
        ):
            data = rng.integers(0, 2**15, size=(count, width)).astype(dtype)
            fname = f"s{index}_{name}.bin"
            pad = 13  # a non-zero byte offset inside the file
            (tmp_path / fname).write_bytes(b"\xaa" * pad + data.tobytes())
            entry[name] = {
                "dtype": {"<u4": "U32", "<u2": "BF16"}[dtype],
                "file": fname,
                "offset": pad,
                "shape": [count, width],
            }
        entries.append(entry)
        start += count
    manifest = {
        "version": 2,
        "source_root": ".",
        "row_width": ROW_WIDTH,
        "quantization": {"bits": 4, "group_size": GROUP, "mode": "affine"},
        "shards": entries,
        "cache_rows": cache_rows,
    }
    path = tmp_path / "ple.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture(autouse=True)
def _restore():
    yield
    ple_parallel.install(0)


@pytest.mark.parametrize("cache_rows", [0, 4])
def test_identical_to_upstream(tmp_path, cache_rows):
    manifest = _build(tmp_path, cache_rows)
    reference = ps.QuantizedMMapNGramEmbedding(manifest)
    patched = ps.QuantizedMMapNGramEmbedding(manifest)
    assert ple_parallel.install(4)
    batches = [[3, 11, 20, 0, 6], [11, 3, 7], [20, 20 - 1, 1], [0]]
    ids = np.array(batches[0])
    for batch in batches:
        ids = np.unique(np.array(batch, dtype=np.int64))
        ple_parallel.install(0)
        expected = reference._read_rows(ids)
        ple_parallel.install(4)
        got = patched._read_rows(ids)
        for want, have in zip(expected, got, strict=True):
            assert want.dtype == have.dtype and want.shape == have.shape
            assert np.array_equal(want, have)
    assert (reference._hits, reference._misses, reference._bytes) == (
        patched._hits,
        patched._misses,
        patched._bytes,
    )
    assert list(reference._cache) == list(patched._cache)


def test_out_of_range_and_toggle(tmp_path):
    manifest = _build(tmp_path, 0)
    table = ps.QuantizedMMapNGramEmbedding(manifest)
    assert ple_parallel.install(2) and ple_parallel.active_threads() == 2
    with pytest.raises(IndexError):
        table._read_rows(np.array([sum(SHARD_ROWS)]))
    assert ple_parallel.install(0) is False and ple_parallel.active_threads() == 0
