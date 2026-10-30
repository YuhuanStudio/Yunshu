"""Compare selected sparse blocks for padded batches and singleton caches."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen4_exp import language as q4


class IdentityRotary:
    def apply_rotary(self, x, y, position_ids, unsqueeze_dim):
        return x, y


def cache_for(length, raw):
    cache = q4.QSAKVCache()
    kv = mx.zeros((1, 1, length, 1), dtype=mx.float16)
    cache.update_and_fetch(kv, kv)
    cache.update_indexer(raw, mx.arange(length, dtype=mx.int32)[None])
    return cache


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("batch,length", [(1, 2052), (2, 8196), (3, 8196), (4, 32772)])
def test_selection_matches_singletons(batch, length, ties):
    mx.random.seed(37)
    cfg = SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hidden_size=128,
        rms_norm_eps=1e-6,
    )
    indexer = q4.Qwen4ExpQSAIndexer(cfg, IdentityRotary())
    lengths = [length - 7 * i for i in range(batch)]
    raw = [mx.random.normal((1, n, 128)).astype(mx.float16) for n in lengths]
    rows = [cache_for(n, keys) for n, keys in zip(lengths, raw, strict=True)]
    batched = q4.BatchQSAKVCache.merge(
        [cache_for(n, keys) for n, keys in zip(lengths, raw, strict=True)]
    )
    projected = mx.random.normal((batch, 1, 5 * 128)).astype(mx.float16)
    if ties:
        projected = mx.zeros_like(projected)
    actual = indexer.select_from_projected(projected, batched, None)
    assert actual is not None
    for i, row in enumerate(rows):
        expected = indexer.select_from_projected(projected[i : i + 1], row, None)
        assert expected is not None
        assert mx.array_equal(
            mx.sort(actual.selected_blocks[i]), mx.sort(expected.selected_blocks[0])
        ).item()
