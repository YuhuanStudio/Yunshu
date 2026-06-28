"""(HIGH): inpaint did NOT preserve the unmasked region in pixel space.

The kept latents end at the clean VAE encoding (final sigma_next=0), so decode(latents)
returns decode(encode(source)) in the kept region — a LOSSY VAE round-trip that softens /
colour-shifts / detail-degrades the WHOLE untouched area on EVERY inpaint, contradicting the
router + _load_mask "unmasked region preserved from the original" contract. Real inpainting
composites final = generated*mask + source*(1-mask) in PIXEL space after decode. W1048 adds
that paste-back via _composite_inpaint, threaded with a pixel-resolution mask.
"""
from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np

from yunshu_engine.image_engine import ImageGenEngine


def test_composite_keeps_source_exact_in_unmasked_region():
    gen = mx.ones((1, 3, 4, 4)) * 0.9    # freshly generated pixels
    src = mx.ones((1, 3, 4, 4)) * -0.5   # original source pixels
    m = np.zeros((1, 1, 4, 4), dtype=np.float32)
    m[0, 0, :2, :2] = 1.0                # top-left 2x2 = inpaint, rest = keep
    out = np.array(ImageGenEngine._composite_inpaint(gen, src, mx.array(m)))
    # masked (inpaint) region → generated
    assert np.allclose(out[0, :, :2, :2], 0.9)
    # kept region → source, byte-exact (NOT a VAE round-trip)
    assert np.allclose(out[0, :, 2:, :], -0.5)
    assert np.allclose(out[0, :, :2, 2:], -0.5)


def test_composite_all_keep_returns_source():
    gen = mx.ones((1, 3, 2, 2)) * 0.7
    src = mx.ones((1, 3, 2, 2)) * 0.1
    mask = mx.zeros((1, 1, 2, 2))  # keep everything
    out = np.array(ImageGenEngine._composite_inpaint(gen, src, mask))
    assert np.allclose(out, 0.1)


def test_composite_all_inpaint_returns_generated():
    gen = mx.ones((1, 3, 2, 2)) * 0.7
    src = mx.ones((1, 3, 2, 2)) * 0.1
    mask = mx.ones((1, 1, 2, 2))  # inpaint everything
    out = np.array(ImageGenEngine._composite_inpaint(gen, src, mask))
    assert np.allclose(out, 0.7)


def test_composite_broadcasts_mixed_dtype():
    # real pipeline: decode output may be float16, source float32, mask float16
    gen = (mx.ones((1, 3, 2, 2)) * 0.5).astype(mx.float16)
    src = (mx.ones((1, 3, 2, 2)) * -0.5).astype(mx.float32)
    mask = mx.array(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)).astype(mx.float16)
    out = np.array(ImageGenEngine._composite_inpaint(gen, src, mask))
    assert np.allclose(out[0, 0, 0, 0], 0.5)   # inpaint
    assert np.allclose(out[0, 0, 0, 1], -0.5)  # keep


def test_inpaint_pipeline_wires_pixel_mask_and_paste_back():
    src = inspect.getsource(ImageGenEngine._run_inpaint_pipeline)
    # builds a pixel-resolution mask (height,width) alongside the latent mask
    assert "_mask_px = self._load_mask(mask_data, height, width)" in src
    assert "_mask_px = self._load_mask(mask_bytes, height, width)" in src
    # and applies the paste-back AFTER decode, BEFORE _to_png
    decode = src.index("self._vae.decode")
    composite = src.index("_composite_inpaint", decode)
    to_png = src.index("_to_png", composite)
    assert decode < composite < to_png, "paste-back must run after decode, before _to_png"
    assert "if _mask_px is not None:" in src  # full-canvas (no mask) path skips paste-back
