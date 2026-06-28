"""Tests for SpecPrefill — attention-based sparse prefill."""
from unittest.mock import MagicMock

import mlx.core as mx

from yunshu_engine.spec_prefill import (
    DEFAULT_KEEP_RATE,
    DEFAULT_THRESHOLD,
    _avg_pool1d,
    _find_attention_layers,
    _get_attn_module,
    _OffsetAdjustedRoPE,
    _PositionMappedRoPE,
    cleanup_rope,
    manual_rope,
    select_chunks,
)


class TestSelectChunks:
    def test_returns_all_when_keep_pct_one(self):
        importance = mx.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6])
        result = select_chunks(importance, keep_pct=1.0, chunk_size=2)
        assert result.shape[0] == 6

    def test_selects_top_chunks(self):
        importance = mx.array([0.1, 0.1, 0.9, 0.9, 0.2, 0.2])
        # 3 chunks of 2: [0.1, 0.9, 0.2] → keep top 2 → chunks 1,2
        result = select_chunks(importance, keep_pct=0.67, chunk_size=2)
        indices = result.tolist()
        assert len(indices) >= 4  # At least 2 chunks * 2 tokens

    def test_single_token(self):
        importance = mx.array([1.0])
        result = select_chunks(importance, keep_pct=0.5, chunk_size=1)
        assert len(result) == 1

    def test_empty_importance(self):
        importance = mx.array([])
        result = select_chunks(importance, keep_pct=0.5, chunk_size=1)
        assert len(result) == 0

    def test_chunk_size_larger_than_tokens(self):
        importance = mx.array([0.1, 0.5, 0.3])
        result = select_chunks(importance, keep_pct=0.5, chunk_size=10)
        assert len(result) >= 1

    def test_keeps_at_least_one_chunk(self):
        importance = mx.array([0.1, 0.2, 0.3])
        result = select_chunks(importance, keep_pct=0.01, chunk_size=1)
        assert len(result) >= 1


class TestManualRoPE:
    def test_output_shape_preserved(self):
        x = mx.ones((1, 4, 8, 64))
        positions = mx.arange(8)
        result = manual_rope(x, positions, dims=64)
        assert result.shape == x.shape

    def test_position_zero_no_rotation(self):
        x = mx.ones((1, 2, 4, 16))
        positions = mx.zeros(4)
        result = manual_rope(x, positions, dims=16)
        assert result.shape == (1, 2, 4, 16)

    def test_non_contiguous_positions(self):
        x = mx.ones((1, 2, 3, 32))
        positions = mx.array([0, 5, 100])
        result = manual_rope(x, positions, dims=32)
        assert result.shape == (1, 2, 3, 32)


class TestAvgPool1d:
    def test_identity_kernel_1(self):
        x = mx.array([[1.0, 2.0, 3.0]])
        result = _avg_pool1d(x, kernel_size=1)
        assert result.shape == x.shape

    def test_smoothing(self):
        x = mx.array([[0.0, 0.0, 10.0, 0.0, 0.0]])
        result = _avg_pool1d(x, kernel_size=3)
        # Peak should be smoothed
        assert float(mx.max(result).item()) < 10.0


class TestPositionMappedRoPE:
    def test_stores_config(self):
        original = MagicMock()
        original.dims = 64
        original.base = 10000.0
        original.scale = 1.0
        positions = mx.arange(10)
        pmr = _PositionMappedRoPE(original, positions, cache_start=0)
        assert pmr._dims == 64
        assert pmr._base == 10000.0


class TestOffsetAdjustedRoPE:
    def test_calls_original_with_adjusted_offset(self):
        calls = []
        def track_call(x, offset=0):
            calls.append(offset)
            return mx.ones((1, 2, 4, 16))
        original = track_call
        oar = _OffsetAdjustedRoPE(original, adjustment=100)
        oar(mx.ones((1, 2, 4, 16)), offset=5)
        assert calls == [105]

    def test_zero_adjustment(self):
        calls = []
        def track_call(x, offset=0):
            calls.append(offset)
            return mx.ones((1, 2, 4, 16))
        original = track_call
        oar = _OffsetAdjustedRoPE(original, adjustment=0)
        oar(mx.ones((1, 2, 4, 16)), offset=10)
        assert calls == [10]


class TestFindAttentionLayers:
    def test_finds_self_attn_layers(self):
        model = MagicMock()
        layer1 = MagicMock(spec=['self_attn'])
        layer2 = MagicMock(spec=[])  # no self_attn
        layer3 = MagicMock(spec=['self_attn'])
        model.layers = [layer1, layer2, layer3]
        result = _find_attention_layers(model)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 2

    def test_empty_model(self):
        model = MagicMock()
        model.layers = []
        result = _find_attention_layers(model)
        assert result == []


class TestGetAttnModule:
    def test_gets_self_attn(self):
        layer = MagicMock(spec=['self_attn'])
        result = _get_attn_module(layer)
        assert result is layer.self_attn

    def test_no_self_attn(self):
        layer = MagicMock(spec=[])
        result = _get_attn_module(layer)
        assert result is None


class TestCleanupRope:
    def test_restores_original_rope(self):
        model = MagicMock()
        layer = MagicMock(spec=['self_attn'])
        original_rope = MagicMock()
        layer.self_attn.rope = _OffsetAdjustedRoPE(original_rope, 10)
        model.layers = [layer]

        cleanup_rope(model)
        assert layer.self_attn.rope is original_rope

    def test_noop_for_position_mapped(self):
        model = MagicMock()
        layer = MagicMock(spec=['self_attn'])
        original_rope = MagicMock()
        layer.self_attn.rope = _PositionMappedRoPE(original_rope, mx.arange(10))
        model.layers = [layer]

        cleanup_rope(model)
        assert layer.self_attn.rope is original_rope

    def test_noop_for_normal_rope(self):
        model = MagicMock()
        layer = MagicMock(spec=['self_attn'])
        normal_rope = MagicMock()
        layer.self_attn.rope = normal_rope
        model.layers = [layer]

        cleanup_rope(model)
        assert layer.self_attn.rope is normal_rope


class TestDefaults:
    def test_default_keep_rate(self):
        assert DEFAULT_KEEP_RATE == 0.20

    def test_default_threshold(self):
        assert DEFAULT_THRESHOLD == 8192
