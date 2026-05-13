"""Tests for inpainting pipeline — VAE encoder, mask handling, inpaint gateway."""

import base64
import io
import struct
import pytest
import numpy as np

from PIL import Image as PILImage


def _make_png(width: int, height: int, color: tuple = (128, 64, 32)) -> bytes:
    """Create a simple solid-color PNG."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    pil = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _make_mask_png(width: int, height: int, fill_fraction: float = 0.5) -> bytes:
    """Create a mask PNG: left half white (fill), right half black (keep)."""
    arr = np.zeros((height, width), dtype=np.uint8)
    split = int(width * fill_fraction)
    arr[:, :split] = 255
    pil = PILImage.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ── VAE Encoder unit tests ──


class TestVAEEncoder:
    def test_encoder_init(self):
        from yunshu_engine.image_engine import VAEEncoder
        enc = VAEEncoder()
        assert enc.conv_in is not None
        assert len(enc.down_blocks) == 4
        assert enc.mid_block is not None
        assert enc.conv_out is not None

    def test_encoder_forward(self):
        """Test encoder produces correct output shapes."""
        import mlx.core as mx
        from yunshu_engine.image_engine import VAEEncoder

        enc = VAEEncoder()
        # Dummy forward pass — weights are random but shapes must match
        img = mx.random.normal((1, 3, 64, 64))
        mean, logvar = enc(img)
        # 64/8 = 8 spatial, 16 latent channels
        assert mean.shape == (1, 16, 8, 8), f"mean shape: {mean.shape}"
        assert logvar.shape == (1, 16, 8, 8), f"logvar shape: {logvar.shape}"
        mx.eval(mean, logvar)

    def test_encoder_downsample_ratio(self):
        """Verify encoder downsamples by 8x."""
        import mlx.core as mx
        from yunshu_engine.image_engine import VAEEncoder

        enc = VAEEncoder()
        for size in [128, 256, 512]:
            img = mx.zeros((1, 3, size, size))
            mean, _ = enc(img)
            expected = size // 8
            assert mean.shape[2] == expected, f"H: {mean.shape[2]} != {expected}"
            assert mean.shape[3] == expected, f"W: {mean.shape[3]} != {expected}"
            mx.eval(mean)


class TestVAEEncode:
    def test_vae_encode_requires_encoder(self):
        """VAE.encode() should fail if no encoder loaded."""
        import mlx.core as mx
        from yunshu_engine.image_engine import VAE, VAEDecoder

        vae = VAE(VAEDecoder())  # No encoder
        img = mx.zeros((1, 3, 64, 64))
        with pytest.raises(RuntimeError, match="encoder not loaded"):
            vae.encode(img)

    def test_vae_encode_deterministic_requires_encoder(self):
        """VAE.encode_deterministic() should fail if no encoder."""
        import mlx.core as mx
        from yunshu_engine.image_engine import VAE, VAEDecoder

        vae = VAE(VAEDecoder())
        img = mx.zeros((1, 3, 64, 64))
        with pytest.raises(RuntimeError, match="encoder not loaded"):
            vae.encode_deterministic(img)

    def test_vae_encode_shape(self):
        """Test VAE encode produces correct latent shape."""
        import mlx.core as mx
        from yunshu_engine.image_engine import VAE, VAEDecoder, VAEEncoder

        encoder = VAEEncoder()
        vae = VAE(VAEDecoder(), encoder)
        img = mx.random.normal((1, 3, 128, 128))
        latents = vae.encode(img)
        assert latents.shape == (16, 16, 16), f"Shape: {latents.shape}"
        mx.eval(latents)

    def test_vae_encode_deterministic_shape(self):
        """Test deterministic encode produces correct shape."""
        import mlx.core as mx
        from yunshu_engine.image_engine import VAE, VAEDecoder, VAEEncoder

        encoder = VAEEncoder()
        vae = VAE(VAEDecoder(), encoder)
        img = mx.random.normal((1, 3, 256, 256))
        latents = vae.encode_deterministic(img)
        assert latents.shape == (16, 32, 32), f"Shape: {latents.shape}"
        mx.eval(latents)


# ── Mask loading tests ──


class TestMaskLoading:
    def test_load_mask_shape(self):
        """Test mask is downsampled to latent resolution."""
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        mask_data = _make_mask_png(256, 256)
        mask = engine._load_mask(mask_data, 32, 32)
        assert mask.shape == (1, 1, 32, 32)

    def test_load_mask_values(self):
        """Test mask values are binary 0/1."""
        import mlx.core as mx
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        mask_data = _make_mask_png(256, 256, fill_fraction=0.5)
        mask = engine._load_mask(mask_data, 32, 32)
        arr = np.array(mask)
        unique = set(np.unique(arr))
        assert unique.issubset({0.0, 1.0}), f"Non-binary values: {unique}"

    def test_load_mask_half_white(self):
        """Test left half is white (1.0) and right half is black (0.0)."""
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        mask_data = _make_mask_png(256, 256, fill_fraction=0.5)
        mask = engine._load_mask(mask_data, 32, 32)
        arr = np.array(mask)[0, 0]  # (H, W)
        # Left half should be 1.0 (fill), right half should be 0.0 (keep)
        assert np.all(arr[:, :16] == 1.0), "Left half should be masked (1.0)"
        assert np.all(arr[:, 16:] == 0.0), "Right half should be kept (0.0)"

    def test_load_mask_full(self):
        """Test fully white mask."""
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        arr_full = np.full((128, 128), 255, dtype=np.uint8)
        pil = PILImage.fromarray(arr_full, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        mask = engine._load_mask(buf.getvalue(), 16, 16)
        np_arr = np.array(mask)
        assert np.all(np_arr == 1.0)

    def test_load_mask_empty(self):
        """Test fully black mask."""
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        arr_empty = np.zeros((128, 128), dtype=np.uint8)
        pil = PILImage.fromarray(arr_empty, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        mask = engine._load_mask(buf.getvalue(), 16, 16)
        np_arr = np.array(mask)
        assert np.all(np_arr == 0.0)


# ── Image loading tests ──


class TestLoadImageTensor:
    def test_load_png_shape(self):
        """Test loading PNG to tensor."""
        import mlx.core as mx
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        png = _make_png(128, 64)
        tensor, h, w = engine._load_image_to_tensor(png)
        assert tensor.shape == (1, 3, 64, 128), f"Shape: {tensor.shape}"
        assert h == 64
        assert w == 128
        mx.eval(tensor)

    def test_load_png_range(self):
        """Test pixel values are in [-1, 1]."""
        import mlx.core as mx
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        # Black image → should be -1.0
        arr_black = np.zeros((64, 64, 3), dtype=np.uint8)
        pil = PILImage.fromarray(arr_black, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        tensor, _, _ = engine._load_image_to_tensor(buf.getvalue())
        np_arr = np.array(tensor)
        assert np.allclose(np_arr, -1.0, atol=0.01)

    def test_load_png_white(self):
        """White image → should be +1.0."""
        from yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        arr_white = np.full((64, 64, 3), 255, dtype=np.uint8)
        pil = PILImage.fromarray(arr_white, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        tensor, _, _ = engine._load_image_to_tensor(buf.getvalue())
        np_arr = np.array(tensor)
        assert np.allclose(np_arr, 1.0, atol=0.01)


# ── Gateway endpoint tests ──


class TestInpaintEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from yunshu_gateway.main import create_app
        app = create_app()
        return TestClient(app)

    def test_inpaint_no_model(self, client):
        """Inpaint should 404 when no image engine loaded."""
        img_b64 = base64.b64encode(_make_png(64, 64)).decode()
        mask_b64 = base64.b64encode(_make_mask_png(64, 64)).decode()
        resp = client.post("/v1/images/inpaint", json={
            "image": img_b64,
            "prompt": "fill with blue sky",
            "mask": mask_b64,
            "size": "64x64",
        })
        # 404 because no image engine is loaded
        assert resp.status_code in (404, 503)

    def test_inpaint_invalid_base64(self, client):
        """Inpaint should 400 with invalid base64."""
        resp = client.post("/v1/images/inpaint", json={
            "image": "not-valid-base64!!!",
            "prompt": "test",
        })
        assert resp.status_code == 400

    def test_inpaint_request_model(self):
        """Test request model validation."""
        from yunshu_gateway.routers.images import ImageInpaintRequest

        req = ImageInpaintRequest(
            image="abc",
            prompt="fill",
            mask="def",
            size="512x512",
            num_inference_steps=8,
            denoise_strength=0.8,
        )
        assert req.denoise_strength == 0.8
        assert req.num_inference_steps == 8


# ── Weight remapping tests ──


class TestVAEWeightRemap:
    def test_remap_encoder_weights(self):
        """Test encoder weight remapping with OIHW→OHWI transpose."""
        import mlx.core as mx
        from yunshu_engine.image_engine import _remap_vae_weights

        # Simulate encoder weights in OIHW format
        raw = {
            "encoder.conv_in.weight": mx.zeros((128, 3, 3, 3)),
            "encoder.conv_in.bias": mx.zeros((128,)),
            "encoder.mid_block.resnets.0.conv1.weight": mx.zeros((512, 512, 3, 3)),
            "encoder.conv_out.weight": mx.zeros((32, 512, 3, 3)),
            "decoder.conv_in.weight": mx.zeros((512, 16, 3, 3)),  # Should be excluded
        }
        result = _remap_vae_weights(raw, "encoder")
        keys = [k for k, v in result]
        assert "conv_in.weight" in keys
        assert "conv_out.weight" in keys
        # Verify transposed shape: (128, 3, 3, 3) OIHW → (128, 3, 3, 3) OHWI
        conv_in = next(v for k, v in result if k == "conv_in.weight")
        assert conv_in.shape == (128, 3, 3, 3), f"Expected OHWI, got {conv_in.shape}"
        # Decoder weight should not be in result
        assert not any("decoder" in k for k, v in result)

    def test_remap_decoder_weights(self):
        """Test decoder weight remapping still works with component param."""
        import mlx.core as mx
        from yunshu_engine.image_engine import _remap_vae_weights

        raw = {
            "decoder.conv_in.weight": mx.zeros((512, 16, 3, 3)),
            "decoder.conv_in.bias": mx.zeros((512,)),
            "encoder.conv_in.weight": mx.zeros((128, 3, 3, 3)),  # Should be excluded
        }
        result = _remap_vae_weights(raw, "decoder")
        keys = [k for k, v in result]
        assert "conv_in.weight" in keys
        # Encoder should not appear
        assert len(result) == 2  # weight + bias
