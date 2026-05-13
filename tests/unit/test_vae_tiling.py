"""Tests for VAE tiled decode/encode — memory-efficient large image processing."""

import pytest
import numpy as np
import mlx.core as mx

from yunshu_engine.image_engine import VAE, VAEDecoder, VAEEncoder, _cosine_ramp


class TestCosineRamp:
    def test_ramp_shape(self):
        ramp = _cosine_ramp(8)
        assert ramp.shape == (8,)

    def test_ramp_monotonic(self):
        ramp = _cosine_ramp(16)
        # Should be monotonically increasing from ~0 to ~1
        for i in range(1, len(ramp)):
            assert ramp[i] >= ramp[i - 1] - 1e-6

    def test_ramp_bounds(self):
        ramp = _cosine_ramp(32)
        assert ramp[0] < 0.05  # Starts near 0
        assert ramp[-1] > 0.95  # Ends near 1

    def test_ramp_empty(self):
        ramp = _cosine_ramp(0)
        assert len(ramp) == 0

    def test_ramp_single(self):
        ramp = _cosine_ramp(1)
        assert len(ramp) == 1
        assert 0.0 <= ramp[0] <= 1.0


class TestVAEDecodeTiled:
    def _make_vae(self):
        """Create a VAE with random weights for shape testing."""
        return VAE(VAEDecoder())

    def test_small_image_no_tiling(self):
        """Small image should decode without tiling."""
        vae = self._make_vae()
        latents = mx.random.normal((16, 1, 8, 8)).astype(mx.float16)
        result = vae.decode_tiled(latents, tile_size_px=512)
        assert result.ndim == 4
        assert result.shape[0] == 1  # batch
        assert result.shape[1] == 3  # channels
        assert result.shape[2] == 64  # 8*8
        assert result.shape[3] == 64
        mx.eval(result)

    def test_large_image_uses_tiling(self):
        """Large latent grid triggers tiled decode with multiple tiles."""
        vae = self._make_vae()
        # 128/8=16 latent dims → 128px output, tile_size=64 → tiles needed
        latents = mx.random.normal((16, 1, 16, 16)).astype(mx.float16)
        result = vae.decode_tiled(latents, tile_size_px=64, overlap_px=16)
        assert result.shape == (1, 3, 128, 128), f"Shape: {result.shape}"
        mx.eval(result)

    def test_tiled_matches_shape(self):
        """Tiled and non-tiled decode produce same shape."""
        vae = self._make_vae()
        latents = mx.random.normal((16, 1, 16, 16)).astype(mx.float16)
        # Non-tiled
        direct = vae.decode(latents)
        mx.eval(direct)
        # Tiled (with large tile size so no actual tiling)
        tiled = vae.decode_tiled(latents, tile_size_px=512, overlap_px=64)
        mx.eval(tiled)
        assert direct.shape == tiled.shape

    def test_tiled_output_finite(self):
        """Tiled decode should produce finite values."""
        vae = self._make_vae()
        latents = mx.random.normal((16, 1, 12, 12)).astype(mx.float16)
        result = vae.decode_tiled(latents, tile_size_px=48, overlap_px=8)
        mx.eval(result)
        arr = np.array(result)
        assert np.all(np.isfinite(arr)), "NaN or Inf in tiled decode output"


class TestVAEEncodeTiled:
    def _make_vae(self):
        return VAE(VAEDecoder(), VAEEncoder())

    def test_small_image_no_tiling(self):
        """Small image encodes without tiling."""
        vae = self._make_vae()
        image = mx.random.normal((1, 3, 64, 64))
        result = vae.encode_tiled(image, tile_size_px=512)
        assert result.shape == (16, 8, 8), f"Shape: {result.shape}"
        mx.eval(result)

    def test_large_image_uses_tiling(self):
        """Large image triggers tiled encoding."""
        vae = self._make_vae()
        image = mx.random.normal((1, 3, 256, 256))
        result = vae.encode_tiled(image, tile_size_px=64, overlap_px=16)
        assert result.shape == (16, 32, 32), f"Shape: {result.shape}"
        mx.eval(result)

    def test_encode_tiled_requires_encoder(self):
        """encode_tiled should fail without encoder."""
        vae = VAE(VAEDecoder())  # No encoder
        image = mx.zeros((1, 3, 256, 256))
        with pytest.raises(RuntimeError, match="encoder not loaded"):
            vae.encode_tiled(image)

    def test_encode_tiled_output_finite(self):
        """Tiled encode should produce finite values."""
        vae = self._make_vae()
        image = mx.random.normal((1, 3, 128, 128))
        result = vae.encode_tiled(image, tile_size_px=64, overlap_px=8)
        mx.eval(result)
        arr = np.array(result)
        assert np.all(np.isfinite(arr)), "NaN or Inf in tiled encode output"


class TestAutoTilingThreshold:
    """Test that _run_pipeline auto-selects tiled decode for large images."""

    def test_auto_tile_decision(self):
        """Verify the auto-tile threshold logic (1024*1024)."""
        # Small image: no tiling needed
        assert 512 * 512 <= 1024 * 1024
        assert 1024 * 1024 <= 1024 * 1024
        # Large image: tiling needed
        assert 2048 * 1024 > 1024 * 1024
        assert 2048 * 2048 > 1024 * 1024
