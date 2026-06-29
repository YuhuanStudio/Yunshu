"""Tests for TeaCache — timestep embedding aware cache for diffusion acceleration."""

import mlx.core as mx
import numpy as np
import pytest


class TestTeaCacheConfig:
    def test_defaults(self):
        from yunshu_engine.teacache import TeaCacheConfig

        config = TeaCacheConfig()
        assert config.rel_l1_thresh == 0.2
        assert config.model_type == "ZImageTransformer"
        assert config.coefficients is not None
        assert len(config.coefficients) == 5

    def test_custom_threshold(self):
        from yunshu_engine.teacache import TeaCacheConfig

        config = TeaCacheConfig(rel_l1_thresh=0.4)
        assert config.rel_l1_thresh == 0.4

    def test_custom_coefficients(self):
        from yunshu_engine.teacache import TeaCacheConfig

        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        config = TeaCacheConfig(coefficients=coeffs)
        assert config.coefficients == coeffs

    def test_invalid_threshold(self):
        from yunshu_engine.teacache import TeaCacheConfig

        with pytest.raises(ValueError, match="positive"):
            TeaCacheConfig(rel_l1_thresh=0.0)

    def test_invalid_coefficients_length(self):
        from yunshu_engine.teacache import TeaCacheConfig

        with pytest.raises(ValueError, match="5 elements"):
            TeaCacheConfig(coefficients=[1.0, 2.0, 3.0])

    def test_model_type_lookup(self):
        from yunshu_engine.teacache import TeaCacheConfig

        config = TeaCacheConfig(model_type="FluxTransformer2DModel")
        assert config.coefficients is not None
        assert (
            config.coefficients != TeaCacheConfig().coefficients
        )  # Different from default


class TestTeaCacheState:
    def test_initial_state(self):
        from yunshu_engine.teacache import TeaCacheState

        state = TeaCacheState()
        assert state.cnt == 0
        assert state.accumulated_rel_l1_distance == 0.0
        assert state.previous_modulated_input is None
        assert state.previous_residual is None

    def test_reset(self):
        from yunshu_engine.teacache import TeaCacheState

        state = TeaCacheState()
        state.cnt = 5
        state.accumulated_rel_l1_distance = 1.5
        state.previous_modulated_input = mx.zeros((1, 256))
        state.previous_residual = mx.zeros((16, 1, 8, 8))
        state.reset()
        assert state.cnt == 0
        assert state.accumulated_rel_l1_distance == 0.0
        assert state.previous_modulated_input is None


class TestTeaCacheHook:
    def _make_hook(self, thresh=0.2):
        from yunshu_engine.teacache import TeaCacheConfig, TeaCacheHook

        config = TeaCacheConfig(rel_l1_thresh=thresh)
        return TeaCacheHook(config)

    def test_init(self):
        hook = self._make_hook()
        assert hook.config.rel_l1_thresh == 0.2
        assert hook.state.cnt == 0

    def test_reset(self):
        hook = self._make_hook()
        hook.state.cnt = 10
        hook._cache_hits = 5
        hook._cache_misses = 5
        hook.reset()
        assert hook.state.cnt == 0
        assert hook._cache_hits == 0

    def test_should_compute_first_step(self):
        """First step should always compute (no cache)."""
        hook = self._make_hook()
        inp = mx.random.normal((1, 256))
        assert hook.should_compute(inp) is True

    def test_should_compute_identical_inputs(self):
        """Identical consecutive inputs should trigger cache hit."""
        hook = self._make_hook(thresh=0.5)
        inp = mx.ones((1, 256))
        # First call always computes
        hook.should_compute(inp)
        hook.state.cnt = 1
        hook.state.previous_modulated_input = inp
        # Identical input → distance = 0 → should cache
        result = hook.should_compute(inp)
        assert result is False, "Identical inputs should trigger cache"

    def test_should_compute_different_inputs(self):
        """Very different inputs should trigger recomputation."""
        hook = self._make_hook(thresh=0.001)  # Very low threshold
        inp1 = mx.ones((1, 256))
        hook.state.cnt = 1
        hook.state.previous_modulated_input = inp1
        inp2 = mx.ones((1, 256)) * 100.0  # Very different
        result = hook.should_compute(inp2)
        assert result is True, "Very different inputs should recompute"

    def test_cnt_increments_on_cache_hit_path(self):
        """cnt must advance on BOTH the slow (miss) and fast (hit) paths,
        so it is a true step index (the cache-hit early return used to skip it → cnt
        counted only misses)."""
        import mlx.core as mx

        class _FakeTransformer:
            t_scale = 1.0

            def t_embedder(self, t):
                # constant embedding → consecutive modulated inputs identical → cache hit
                return mx.ones((1, 16))

            def __call__(self, x, timestep, sigmas, cap_feats):
                return mx.zeros((1, 4))

        hook = self._make_hook(thresh=1e9)  # huge threshold → always a hit after step 0
        tf = _FakeTransformer()
        x = mx.zeros((1, 4))
        ts = mx.ones((1,))
        # step 0: previous_residual is None → slow path (forced compute) → cnt 0→1
        hook.forward(tf, x, ts, None, None)
        assert hook.state.cnt == 1
        assert hook._cache_misses == 1
        # step 1: identical t-emb + huge threshold → cache HIT → cnt must still advance
        hook.forward(tf, x, ts, None, None)
        assert hook._cache_hits == 1
        assert hook.state.cnt == 2, "cnt must increment on the cache-hit path too"

    def test_get_stats(self):
        hook = self._make_hook()
        hook._cache_hits = 3
        hook._cache_misses = 7
        stats = hook.get_stats()
        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 7
        assert abs(stats["hit_rate"] - 0.3) < 0.01

    def test_get_stats_zero(self):
        hook = self._make_hook()
        stats = hook.get_stats()
        assert stats["hit_rate"] == 0.0


class TestTeaCachePolynomial:
    def test_polynomial_rescaling(self):
        """Test that polynomial rescaling is applied correctly."""
        from yunshu_engine.teacache import TeaCacheConfig

        config = TeaCacheConfig()
        coeffs = config.coefficients
        poly = np.poly1d(coeffs)
        # Should produce finite values for reasonable inputs
        for x in [0.0, 0.1, 0.5, 1.0]:
            result = float(poly(x))
            assert np.isfinite(result), f"Polynomial({x}) = {result}"

    def test_zimage_coefficients(self):
        """Z-Image coefficients should produce reasonable rescaled distances."""
        from yunshu_engine.teacache import TeaCacheConfig

        config = TeaCacheConfig(model_type="ZImageTransformer")
        poly = np.poly1d(config.coefficients)
        # Small input distance should produce small rescaled distance
        result_small = abs(float(poly(0.01)))
        result_large = abs(float(poly(0.5)))
        assert result_small < result_large or result_small > 0, (
            "Polynomial should differentiate between small and large inputs"
        )


class TestTeaCacheIntegration:
    def test_env_var_detection(self):
        """Test that TeaCache is auto-enabled via env var."""
        import os

        from yunshu_engine.image_engine import ImageGenEngine

        # Save and set env var
        old_val = os.environ.get("YUNSHU_TEACACHE")
        try:
            os.environ["YUNSHU_TEACACHE"] = "0.3"
            engine = ImageGenEngine("/fake/path")
            assert engine._teacache_config is not None
            assert engine._teacache_config.rel_l1_thresh == 0.3
        finally:
            if old_val is None:
                os.environ.pop("YUNSHU_TEACACHE", None)
            else:
                os.environ["YUNSHU_TEACACHE"] = old_val

    def test_env_var_disabled(self):
        """Test that TeaCache is disabled when env var is 0."""
        import os

        from yunshu_engine.image_engine import ImageGenEngine

        old_val = os.environ.get("YUNSHU_TEACACHE")
        try:
            os.environ["YUNSHU_TEACACHE"] = "0"
            engine = ImageGenEngine("/fake/path")
            assert engine._teacache_config is None
        finally:
            if old_val is None:
                os.environ.pop("YUNSHU_TEACACHE", None)
            else:
                os.environ["YUNSHU_TEACACHE"] = old_val
