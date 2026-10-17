"""CPU contract for the real-load screenshot receipt."""

from scripts.research.docs016_console import validate


def test_capture_requires_identical_inventory_and_real_load():
    images = {"en/overview.webp": [2880, 1800]}
    assert validate(images, images, 3) == []
    assert validate(images, {}, 3)
    assert validate(images, images, 2)
