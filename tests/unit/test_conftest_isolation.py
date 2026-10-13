"""Unit tests run with HOME outside the developer's real home directory."""

from pathlib import Path


def test_home_is_isolated_from_the_real_home():
    home = Path.home()
    assert "yunshu-unit-devroot-" in str(home), home
