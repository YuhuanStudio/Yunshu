"""Tests for mRoPE (Multi-dimensional Rotary Position Embedding) support."""
import pytest
from unittest.mock import MagicMock

import mlx.core as mx

from yunshu_engine.mrope import (
    MRoPEInfo,
    detect_mrope,
    build_decode_positions,
    build_prefill_positions,
    capture_rope_deltas,
    clear_rope_state,
    BatchRopeDeltaManager,
)


class TestDetectMRoPE:
    def test_no_mrope_in_config(self):
        info = detect_mrope({"model_type": "llama"})
        assert not info.enabled

    def test_mrope_section_in_rope_scaling(self):
        config = {
            "rope_scaling": {
                "mrope_section": [0, 0, 0, 0, 0, 16, 24, 24],
            }
        }
        info = detect_mrope(config)
        assert info.enabled
        assert info.sections == (16, 24, 24)
        assert info.num_dims == 3
        assert "mrope_section" in info.source_key

    def test_mrope_section_in_text_config(self):
        config = {
            "text_config": {
                "rope_scaling": {
                    "mrope_section": "0,0,0,0,0,16,24,24",
                }
            }
        }
        info = detect_mrope(config)
        assert info.enabled
        assert info.sections == (16, 24, 24)

    def test_mrope_section_in_rope_parameters(self):
        config = {
            "rope_parameters": {
                "mrope_section": [16, 24, 24],
            }
        }
        info = detect_mrope(config)
        assert info.enabled
        assert info.sections == (16, 24, 24)

    def test_empty_config(self):
        info = detect_mrope({})
        assert not info.enabled

    def test_text_config_without_mrope(self):
        config = {
            "text_config": {
                "rope_scaling": {"type": "linear"},
            }
        }
        info = detect_mrope(config)
        assert not info.enabled


class TestBuildDecodePositions:
    def test_single_text_request(self):
        positions = build_decode_positions([10], [0.0])
        assert positions.shape == (3, 1, 1)
        # All 3 dims should have the same position for text-only
        for d in range(3):
            assert int(positions[d, 0, 0].item()) == 10

    def test_mixed_batch(self):
        positions = build_decode_positions([5, 10, 15], [0.0, 3.0, 0.0])
        assert positions.shape == (3, 3, 1)
        # First request: text-only, offset 5
        assert int(positions[0, 0, 0].item()) == 5
        # Second request: VLM with delta 3, offset 10+3=13
        assert int(positions[0, 1, 0].item()) == 13
        # Third request: text-only, offset 15
        assert int(positions[0, 2, 0].item()) == 15

    def test_all_vlm_requests(self):
        positions = build_decode_positions([20, 30], [5.0, 8.0])
        assert positions.shape == (3, 2, 1)
        assert int(positions[0, 0, 0].item()) == 25
        assert int(positions[0, 1, 0].item()) == 38


class TestBuildPrefillPositions:
    def test_text_prefill(self):
        positions = build_prefill_positions(100)
        assert positions.shape == (3, 1, 100)
        # Should be 0..99 for all dims
        assert int(positions[0, 0, 0].item()) == 0
        assert int(positions[0, 0, 99].item()) == 99

    def test_single_token_prefill(self):
        positions = build_prefill_positions(1)
        assert positions.shape == (3, 1, 1)
        assert int(positions[0, 0, 0].item()) == 0


class TestCaptureRopeDeltas:
    def test_model_without_deltas(self):
        model = MagicMock(spec=[])
        assert capture_rope_deltas(model) is None

    def test_model_with_deltas(self):
        model = MagicMock(spec=[])
        model._rope_deltas = 3.0
        assert capture_rope_deltas(model) == 3.0

    def test_vlm_language_model(self):
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model._rope_deltas = 7.5
        assert capture_rope_deltas(model) == 7.5


class TestClearRopeState:
    def test_clears_model_state(self):
        model = MagicMock(spec=[])
        model._position_ids = mx.array([1, 2, 3])
        model._rope_deltas = 5.0
        model._batch_rope_deltas = mx.array([1.0, 2.0])

        clear_rope_state(model)

        assert model._position_ids is None
        assert model._rope_deltas is None
        assert model._batch_rope_deltas is None

    def test_clears_language_model_state(self):
        model = MagicMock(spec=[])
        lang = MagicMock(spec=[])
        lang._rope_deltas = 2.0
        model.language_model = lang

        clear_rope_state(model)

        assert lang._rope_deltas is None


class TestBatchRopeDeltaManager:
    def test_register_and_get(self):
        mgr = BatchRopeDeltaManager()
        mgr.register(1, 3.0)
        mgr.register(2, 0.0)
        mgr.register(3, 5.5)

        deltas = mgr.get_batch_deltas([1, 2, 3])
        assert deltas == [3.0, 0.0, 5.5]

    def test_unregister(self):
        mgr = BatchRopeDeltaManager()
        mgr.register(1, 3.0)
        mgr.unregister(1)

        deltas = mgr.get_batch_deltas([1])
        assert deltas == [0.0]

    def test_missing_uid_defaults_zero(self):
        mgr = BatchRopeDeltaManager()
        deltas = mgr.get_batch_deltas([99])
        assert deltas == [0.0]

    def test_clear(self):
        mgr = BatchRopeDeltaManager()
        mgr.register(1, 3.0)
        mgr.register(2, 4.0)
        mgr.clear()

        assert mgr.get_batch_deltas([1, 2]) == [0.0, 0.0]

    def test_overwrite_on_reregister(self):
        mgr = BatchRopeDeltaManager()
        mgr.register(1, 3.0)
        mgr.register(1, 7.0)

        assert mgr.get_batch_deltas([1]) == [7.0]
