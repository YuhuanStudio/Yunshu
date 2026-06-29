"""Tests for yunshu_engine.image_engine — Image generation engine."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from yunshu_engine.image_engine import (
    ImageGenEngine,
    _compute_sigmas,
    _FinalLayer,
    _RopeEmbedder,
    _TimestepEmbedder,
)


class TestImageGenEngineInit:
    def test_default_state(self):
        engine = ImageGenEngine("/models/zimage")
        assert engine.model_name == "zimage"
        assert engine.is_loaded is False

    def test_model_name_nested(self):
        engine = ImageGenEngine("/home/user/models/zimage-v2")
        assert engine.model_name == "zimage-v2"

    def test_model_name_no_slash(self):
        engine = ImageGenEngine("local-image-model")
        assert engine.model_name == "local-image-model"

    def test_resolve_model_id_exact(self):
        engine = ImageGenEngine("/models/zimage")
        assert engine.resolve_model_id("zimage") is True

    def test_resolve_model_id_full_path(self):
        engine = ImageGenEngine("/models/zimage")
        assert engine.resolve_model_id("/models/zimage") is True

    def test_resolve_model_id_case_insensitive(self):
        engine = ImageGenEngine("/models/ZImage")
        assert engine.resolve_model_id("zimage") is True

    def test_resolve_model_id_not_found(self):
        engine = ImageGenEngine("/models/zimage")
        assert engine.resolve_model_id("other-model") is False

    def test_stop_clears_state(self):
        engine = ImageGenEngine("/models/zimage")
        engine._transformer = MagicMock()
        engine._running = True

        async def _do_stop():
            await engine.stop()

        # stop() calls run_in_executor for sync_and_clear_cache — mock at module level
        import yunshu_engine.mlx_executor as mlx_exec

        original = mlx_exec.sync_and_clear_cache
        mlx_exec.sync_and_clear_cache = lambda: None
        try:
            asyncio.new_event_loop().run_until_complete(_do_stop())
        finally:
            mlx_exec.sync_and_clear_cache = original
        assert engine.is_loaded is False
        assert engine._running is False


class TestComputeSigmas:
    def test_returns_correct_length(self):
        sigmas = _compute_sigmas(4)
        # num_steps + 1 for the trailing zero
        assert sigmas.shape[0] == 5

    def test_last_element_is_zero(self):
        sigmas = _compute_sigmas(4)
        assert float(sigmas[-1].item()) == 0.0

    def test_first_element_positive(self):
        sigmas = _compute_sigmas(4)
        assert float(sigmas[0].item()) > 0.0

    def test_different_step_counts(self):
        for n in [1, 4, 8, 16]:
            sigmas = _compute_sigmas(n)
            assert sigmas.shape[0] == n + 1

    def test_without_sigma_shift(self):
        sigmas = _compute_sigmas(4, requires_sigma_shift=False)
        assert sigmas.shape[0] == 5
        # Linear spacing: first should be close to 1.0
        assert abs(float(sigmas[0].item()) - 1.0) < 0.01


class TestTimestepEmbedder:
    def test_output_shape(self):
        embedder = _TimestepEmbedder(out_size=128, mid_size=64, freq_size=32)
        t = mx.array([0.5])
        result = embedder(t)
        assert result.shape[-1] == 128

    def test_sinusoidal(self):
        result = _TimestepEmbedder._sinusoidal(mx.array([0.5]), dim=64)
        assert result.shape[-1] == 64

    def test_different_timesteps(self):
        embedder = _TimestepEmbedder(out_size=256, mid_size=128, freq_size=64)
        for t_val in [0.0, 0.5, 1.0]:
            result = embedder(mx.array([t_val]))
            assert result.shape[-1] == 256


class TestFinalLayer:
    def test_output_shape(self):
        layer = _FinalLayer(hidden_size=64, out_channels=16)
        x = mx.zeros((1, 10, 64))
        c = mx.zeros((1, 64))
        result = layer(x, c)
        assert result.shape == (1, 10, 16)


class TestRopeEmbedder:
    def test_output_shape(self):
        embedder = _RopeEmbedder(theta=256.0, axes_dims=[16, 16], axes_lens=[64, 64])
        ids = mx.zeros((1, 10, 2), dtype=mx.int32)
        result = embedder(ids)
        assert result.ndim >= 2

    def test_default_axes(self):
        embedder = _RopeEmbedder()
        assert embedder.axes_dims == [32, 48, 48]
        assert len(embedder.freqs_cis) == 3


class TestGenerateImageErrors:
    @pytest.mark.asyncio
    async def test_generate_without_load_raises(self):
        engine = ImageGenEngine("/models/zimage")
        with pytest.raises(RuntimeError, match="Engine not started"):
            await engine.generate_image("a cat")

    @pytest.mark.asyncio
    async def test_stream_without_load_raises(self):
        engine = ImageGenEngine("/models/zimage")
        with pytest.raises(RuntimeError, match="Engine not started"):
            async for _ in engine.generate_image_stream("a cat"):
                pass

    @pytest.mark.asyncio
    async def test_generate_rejects_non_multiple_of_16_dimensions(self):
        engine = ImageGenEngine("/models/zimage")
        engine._transformer = MagicMock()
        with pytest.raises(ValueError, match="multiples of 16"):
            await engine.generate_image("a cat", width=333, height=333)

    @pytest.mark.asyncio
    async def test_generate_rejects_multiple_of_8_not_16(self):
        # %8-but-not-%16 (e.g. 1080) used to pass the gate then crash
        # deep in _patchify (patch_size=2). Must now be rejected cleanly.
        engine = ImageGenEngine("/models/zimage")
        engine._transformer = MagicMock()
        with pytest.raises(ValueError, match="multiples of 16"):
            await engine.generate_image("a cat", width=1080, height=1080)

    @pytest.mark.asyncio
    async def test_start_with_preloaded(self):
        engine = ImageGenEngine("/models/zimage")
        engine._transformer = MagicMock()
        await engine.start()
        assert engine._running is True
