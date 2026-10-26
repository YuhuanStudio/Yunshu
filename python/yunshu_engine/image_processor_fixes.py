# Patches upstream (MIT): mlx-vlm Gemma4ImageProcessor.preprocess and EmbeddingGemma2ImageProcessor.preprocess read
# PIL images as channels-last (Blaizzy/mlx-vlm 8dd7746 and 1e64ba4; tracked in vendor.json).
"""PIL images are always height x width x channels; the processors must not infer that from the shape.

``infer_channel_dimension_format`` only sees a shape. A 3-pixel-tall RGB image is ``(3, W, 3)``, which
reads as channels-first: nothing fails, the image rows become channels and the vision tower receives
scrambled pixels with an unchanged soft-token count. A 1-pixel-tall image reads as one channel and its
patches come out 16*16*1 wide. mlx-vlm 0.7.8 has this in the Gemma 4 and EmbeddingGemma 2 image
processors (fixed upstream in 0.7.9 by 8dd7746 / 1e64ba4, which needs mlx 0.32.4, so the fix is carried
here until the pin moves).

The fix tags the array that comes from a PIL image, and the inference answers "channels-last" for a tagged
three-dimensional array; raw array inputs still go through the upstream inference. Both modules call the
two transformers helpers as module globals, so wrapping them there changes nothing else.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODULES = (
    "mlx_vlm.models.gemma4.processing_gemma4",
    "mlx_vlm.models.embedding_gemma2.image_processing_embedding_gemma2",
)

try:  # numpy is a hard dependency of mlx-vlm; keep the import lazy for a bare engine
    import numpy as _np

    class PilPixels(_np.ndarray):
        """An image array that came from a PIL image: height x width x channels by construction."""

except ImportError:  # pragma: no cover
    PilPixels = None  # type: ignore[assignment,misc]


def tag_pil(array, source):
    """``array`` viewed as ``PilPixels`` when ``source`` is a PIL image and the array is 3-D."""
    from PIL import Image

    if isinstance(source, Image.Image) and getattr(array, "ndim", 0) == 3:
        return array.view(PilPixels)
    return array


def channels_last_if_pil(image, infer):
    """Channel layout of ``image``: LAST for a PIL-derived array, ``infer(image)`` otherwise."""
    if PilPixels is not None and isinstance(image, PilPixels):
        from transformers.image_utils import ChannelDimension

        return ChannelDimension.LAST
    return infer(image)


def install() -> list[str]:
    """Wrap the two helpers in each processor module (idempotent). Returns the patched module names."""
    import importlib

    patched: list[str] = []
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # the family is optional in this mlx-vlm build
            logger.debug("image processor fix skipped for %s: %s", name, exc)
            continue
        if getattr(module, "_yunshu_pil_channels_last", False):
            continue
        to_numpy = getattr(module, "to_numpy_array", None)
        infer = getattr(module, "infer_channel_dimension_format", None)
        if to_numpy is None or infer is None:
            continue

        def _to_numpy(image, *args, _orig=to_numpy, **kwargs):
            return tag_pil(_orig(image, *args, **kwargs), image)

        def _infer(image, *args, _orig=infer, **kwargs):
            if PilPixels is not None and isinstance(image, PilPixels):
                return channels_last_if_pil(image, _orig)
            return _orig(image, *args, **kwargs)

        setattr(module, "to_numpy_array", _to_numpy)  # noqa: B010 - a module attribute, not typed
        setattr(module, "infer_channel_dimension_format", _infer)  # noqa: B010
        setattr(module, "_yunshu_pil_channels_last", True)  # noqa: B010
        patched.append(name)
    return patched
