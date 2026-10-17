"""CPU contract for the real-load screenshot receipt."""

from scripts.research.docs016_console import validate


def test_capture_requires_identical_inventory_and_real_load():
    images = {"en/overview.webp": [2880, 1800]}
    assert validate(images, images, 3) == []
    assert validate(images, {}, 3)
    assert validate(images, images, 2)


def test_long_session_requires_real_depth_cache_and_complete_replies():
    from scripts.research.docs016_console import validate_long_usage

    usage = {
        "prompt_tokens": 32768,
        "completion_tokens": 2048,
        "prompt_tokens_details": {"cached_tokens": 32768},
    }
    assert validate_long_usage(usage, True) == []
    assert validate_long_usage({**usage, "completion_tokens": 100}, True)
    assert validate_long_usage({**usage, "prompt_tokens": 1024}, False)
    assert validate_long_usage({**usage, "prompt_tokens_details": {}}, True)


def test_capture_swap_reading_fails_closed():
    import pytest
    from scripts.research.docs016_console import swap_mb

    assert swap_mb("used = 10.25M") == 10.25
    assert swap_mb("used = 1.25G") == 1280
    with pytest.raises(ValueError):
        swap_mb("unknown")
