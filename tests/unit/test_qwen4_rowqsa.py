"""Batched Qwen4 decode runs each QSA row as the singleton step (row-exact attention)."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen4_exp import language as q4

from yunshu_engine import qwen4_rowqsa

D = 4


def attention(topk=2, ratio=4):
    return SimpleNamespace(
        indexer=SimpleNamespace(block_topk=topk, compress_ratio=ratio)
    )


def fake_original(self, attention, hidden, cache, position_ids, mask):
    """A step whose result depends on how many rows share it (the upstream defect)."""
    batch = hidden.shape[0]
    keys = hidden.reshape(batch, 1, 1, D)
    full, _ = cache.update_and_fetch(keys, keys)
    return full.sum(axis=(1, 2, 3))[:, None, None] + (0.125 if batch > 1 else 0.0)


def row_cache(length, seed):
    cache = q4.QSAKVCache()
    keys = mx.arange(length * D, dtype=mx.float32).reshape(1, 1, length, D) * seed
    cache.update_and_fetch(keys, keys)
    return cache


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        q4.Qwen4ExpBatchInvariantForward, "_qsa_attention", fake_original
    )
    monkeypatch.setitem(qwen4_rowqsa._STATE, "installed", False)
    monkeypatch.setitem(qwen4_rowqsa._STATE, "rows", 0)
    assert qwen4_rowqsa.install()
    return q4.Qwen4ExpBatchInvariantForward._qsa_attention


def step(fn, attn, hidden, cache):
    return fn(object(), attn, hidden, cache, None, None)


def test_batched_step_equals_singleton_steps(patched):
    rows = [row_cache(16, 1.0), row_cache(13, 2.0)]
    hidden = mx.array([[[1.0, 2, 3, 4]], [[5.0, 6, 7, 8]]])
    batch = q4.BatchQSAKVCache.merge([row_cache(16, 1.0), row_cache(13, 2.0)])
    got = step(patched, attention(), hidden, batch)
    want = mx.concatenate(
        [
            fake_original(None, None, hidden[i : i + 1], rows[i], None, None)
            for i in range(2)
        ]
    )
    assert mx.array_equal(got, want).item()
    # the rebuilt batch cache holds every row's appended token
    for i, entry in enumerate(rows):
        again = batch.extract(i)
        assert again.offset == entry.offset == [17, 14][i]
        assert mx.array_equal(
            again.keys[..., : again.offset, :], entry.keys[..., : entry.offset, :]
        ).item()
    assert qwen4_rowqsa._STATE["rows"] == 1


def test_short_contexts_and_single_rows_keep_the_native_step(patched):
    short = q4.BatchQSAKVCache.merge([row_cache(3, 1.0), row_cache(4, 2.0)])
    hidden = mx.ones((2, 1, D))
    baseline = fake_original(
        None,
        attention(),
        hidden,
        q4.BatchQSAKVCache.merge([row_cache(3, 1.0), row_cache(4, 2.0)]),
        None,
        None,
    )
    assert mx.array_equal(step(patched, attention(), hidden, short), baseline).item()
    one = q4.BatchQSAKVCache.merge([row_cache(20, 1.0)])
    step(patched, attention(), mx.ones((1, 1, D)), one)
    assert qwen4_rowqsa._STATE["rows"] == 0


def test_unpatched_batch_step_differs_from_the_singleton_step():
    rows = [row_cache(16, 1.0), row_cache(13, 2.0)]
    hidden = mx.array([[[1.0, 2, 3, 4]], [[5.0, 6, 7, 8]]])
    batch = q4.BatchQSAKVCache.merge([row_cache(16, 1.0), row_cache(13, 2.0)])
    got = fake_original(None, attention(), hidden, batch, None, None)
    want = mx.concatenate(
        [
            fake_original(None, None, hidden[i : i + 1], rows[i], None, None)
            for i in range(2)
        ]
    )
    assert not mx.array_equal(got, want).item()


def test_install_is_idempotent(patched):
    assert qwen4_rowqsa.install()
    assert q4.Qwen4ExpBatchInvariantForward._qsa_attention is patched
