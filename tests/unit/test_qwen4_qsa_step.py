"""A real QSA step matches independent singleton attention and cache updates."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.qwen4_exp import language as q4
from mlx_vlm.models.qwen4_exp.config import TextConfig

from yunshu_engine import qwen4_rowqsa


def config():
    return dict(
        model_type="qwen4_exp",
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=16,
        ple_layer_ids=[],
        eos_token_id=127,
        ple_embed_dim=128,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_head_dim=64,
        indexer_budget=8,
        indexer_compress_ratio=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )


def make_cache(length):
    cache = q4.QSAKVCache()
    keys = mx.random.normal((1, 1, length, 64)).astype(mx.bfloat16)
    values = mx.random.normal((1, 1, length, 64)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    cache.update_indexer(index_keys, mx.arange(length, dtype=mx.int32)[None])
    return cache


def test_real_sparse_step_and_cache_are_row_exact(monkeypatch):
    mx.random.seed(71)
    attention = q4.Qwen4ExpAttention(TextConfig.from_dict(config()))
    attention.set_dtype(mx.bfloat16)
    mx.eval(attention.parameters())
    owner = q4.Qwen4ExpBatchInvariantForward()
    rows = [make_cache(25), make_cache(20), make_cache(17)]
    batch = q4.BatchQSAKVCache.merge(rows)
    # Extract fresh copies: merge does not mutate the original caches.
    expected_caches = [batch.extract(i) for i in range(3)]
    hidden = mx.random.normal((3, 1, 128)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [
            owner._qsa_attention(attention, hidden[i : i + 1], cache, None, None)
            for i, cache in enumerate(expected_caches)
        ]
    )
    # Emulate a batched indexer choosing a different block: the wrapper must
    # use each singleton selection even when the batched selection differs.
    native_select = attention.indexer.select_from_projected

    def divergent_batch_selection(projected, cache, position_ids):
        result = native_select(projected, cache, position_ids)
        if result is not None and projected.shape[0] > 1:
            result.selected_blocks = mx.concatenate(
                [
                    mx.full_like(result.selected_blocks[..., :1], -1),
                    result.selected_blocks[..., 1:],
                ],
                axis=-1,
            )
        return result

    monkeypatch.setattr(
        attention.indexer, "select_from_projected", divergent_batch_selection
    )
    original = q4.Qwen4ExpBatchInvariantForward._qsa_attention
    monkeypatch.setattr(q4.Qwen4ExpBatchInvariantForward, "_qsa_attention", original)
    monkeypatch.setitem(qwen4_rowqsa._STATE, "installed", False)
    monkeypatch.setitem(qwen4_rowqsa._STATE, "batched", 0)
    assert qwen4_rowqsa.install()

    def forbid_full_extract(_index):
        raise AssertionError("batched attention must not extract full KV rows")

    native_extract = batch.extract
    monkeypatch.setattr(batch, "extract", forbid_full_extract)
    actual = owner._qsa_attention(attention, hidden, batch, None, "left_padded_decode")
    monkeypatch.setattr(batch, "extract", native_extract)
    assert qwen4_rowqsa._STATE["batched"] == 1
    assert mx.array_equal(actual, expected).item()
    for i, expected_cache in enumerate(expected_caches):
        got = batch.extract(i)
        assert got.offset == expected_cache.offset
        for field in ("keys", "values"):
            assert mx.array_equal(
                getattr(got, field)[..., : got.offset, :],
                getattr(expected_cache, field)[..., : expected_cache.offset, :],
            ).item(), field
        for field in ("index_keys", "index_position_ids"):
            assert mx.array_equal(
                getattr(got, field), getattr(expected_cache, field)
            ).item(), field


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_flash_next_shape_sparse_step_matches_singletons(monkeypatch, bits):
    """Exercise the real head and indexer geometry, not just the small fixture."""
    mx.random.seed(91)
    cfg = config()
    cfg.update(
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_n_heads=4,
        indexer_head_dim=128,
        indexer_budget=2048,
    )
    attention = q4.Qwen4ExpAttention(TextConfig.from_dict(cfg))
    attention.set_dtype(mx.bfloat16)
    nn.quantize(attention, group_size=64, bits=bits)
    mx.eval(attention.parameters())
    owner = q4.Qwen4ExpBatchInvariantForward()

    def cache_for(length):
        cache = q4.QSAKVCache()
        k = mx.random.normal((1, 2, length, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, length, 256)).astype(mx.bfloat16)
        index = mx.random.normal((1, length, 128)).astype(mx.bfloat16)
        cache.update_and_fetch(k, v)
        cache.update_indexer(index, mx.arange(length, dtype=mx.int32)[None])
        return cache

    original = q4.Qwen4ExpBatchInvariantForward._qsa_attention
    initial = [cache_for(6146), cache_for(6108)]
    for row in initial:
        original(
            owner, attention, mx.zeros((1, 1, 2560), dtype=mx.bfloat16), row, None, None
        )
    batch = q4.BatchQSAKVCache.merge(initial)
    rows = [batch.extract(i) for i in range(2)]
    monkeypatch.setattr(q4.Qwen4ExpBatchInvariantForward, "_qsa_attention", original)
    monkeypatch.setitem(qwen4_rowqsa._STATE, "installed", False)
    monkeypatch.setitem(qwen4_rowqsa._STATE, "batched", 0)
    assert qwen4_rowqsa.install()
    for _ in range(4):
        hidden = mx.random.normal((2, 1, 2560)).astype(mx.bfloat16)
        expected = mx.concatenate(
            [
                original(owner, attention, hidden[i : i + 1], row, None, None)
                for i, row in enumerate(rows)
            ]
        )
        actual = owner._qsa_attention(
            attention, hidden, batch, None, "left_padded_decode"
        )
        assert mx.array_equal(actual, expected).item()
    assert qwen4_rowqsa._STATE["batched"] == 4
