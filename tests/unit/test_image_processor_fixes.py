"""PIL images keep their height x width x channels layout through the Gemma 4 / EmbeddingGemma 2 processors
(mlx-vlm 8dd7746 / 1e64ba4): a 1- or 3-pixel-tall image must not be read as channels-first."""

import numpy as np
import pytest

pytest.importorskip("mlx_vlm")
from PIL import Image  # noqa: E402

from yunshu_engine import image_processor_fixes  # noqa: E402


@pytest.fixture(autouse=True)
def _installed():
    image_processor_fixes.install()


def _pixels(height, width=4):
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


@pytest.mark.parametrize("height", [1, 3], ids=["1px-tall", "3px-tall"])
def test_gemma4_short_pil_images_stay_channels_last(height):
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor

    pixels = _pixels(height)
    processor = Gemma4ImageProcessor(do_resize=False, do_rescale=False)
    data, _ = processor(Image.fromarray(pixels))
    np.testing.assert_array_equal(
        np.asarray(data["pixel_values"][0]), pixels.transpose(2, 0, 1)
    )


@pytest.mark.parametrize("height", [1, 3], ids=["1px-tall", "3px-tall"])
def test_embedding_gemma2_short_pil_images_stay_channels_last(height):
    from mlx_vlm.models.embedding_gemma2.image_processing_embedding_gemma2 import (
        EmbeddingGemma2ImageProcessor,
    )

    processor = EmbeddingGemma2ImageProcessor()
    out = processor(Image.new("RGB", (900, height)))
    # a patch is 16*16*3 values; a channels-first reading of a 1-pixel-tall image gives 16*16*1
    # patches and of a 3-pixel-tall image swaps rows and channels before the resize
    assert np.asarray(out["pixel_values"]).shape[-1] == 16 * 16 * 3


def test_raw_arrays_still_use_the_upstream_inference():
    from mlx_vlm.models.gemma4 import processing_gemma4 as module

    chw = np.zeros((3, 8, 8), dtype=np.uint8)
    hwc = np.zeros((8, 8, 3), dtype=np.uint8)
    first = module.infer_channel_dimension_format(chw)
    last = module.infer_channel_dimension_format(hwc)
    assert first != last


def test_install_is_idempotent():
    from mlx_vlm.models.gemma4 import processing_gemma4 as module

    before = module.infer_channel_dimension_format
    assert image_processor_fixes.install() == []
    assert module.infer_channel_dimension_format is before
