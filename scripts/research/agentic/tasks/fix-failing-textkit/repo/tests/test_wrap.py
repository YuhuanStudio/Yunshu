import pytest

from textkit import wrap


def test_exact_width_fits():
    assert wrap("aaa bbb", 7) == ["aaa bbb"]


def test_breaks_before_overflow():
    assert wrap("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]


def test_long_word_own_line():
    assert wrap("a bigwordhere b", 5) == ["a", "bigwordhere", "b"]


def test_bad_width():
    with pytest.raises(ValueError):
        wrap("x", 0)
