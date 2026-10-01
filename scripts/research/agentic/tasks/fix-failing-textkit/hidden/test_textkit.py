import pytest

from textkit import memoize, render, slugify, wrap


def test_wrap_fit_and_break():
    assert wrap("aaa bbb", 7) == ["aaa bbb"]
    assert wrap("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]
    assert wrap("aaa bbb", 6) == ["aaa", "bbb"]
    assert wrap("one two three four five", 9) == ["one two", "three", "four five"]
    assert wrap("  spaced   out\n words\t here ", 11) == ["spaced out", "words here"]
    assert wrap("a bigwordhere b", 5) == ["a", "bigwordhere", "b"]
    assert wrap("", 10) == [] and wrap("   ", 10) == []
    assert wrap("abcde", 5) == ["abcde"]
    assert wrap("ab cd", 5) == ["ab cd"]
    with pytest.raises(ValueError):
        wrap("x", 0)


def test_wrap_lines_never_exceed_width_unless_single_word():
    text = "the quick brown fox jumps over the lazy dog " * 5
    for w in (5, 10, 17, 30):
        for line in wrap(text, w):
            assert len(line) <= w or " " not in line


def test_slugify():
    assert slugify("Hello, World") == "hello-world"
    assert slugify("  Café Déjà Vu!  ") == "cafe-deja-vu"
    assert slugify("a --- b") == "a-b"
    assert slugify("---x---") == "x"
    assert (
        slugify("Ünïcödé Straße") == "unicode-strasse"
        or slugify("Ünïcödé Straße") == "unicode-strae"
    )
    assert slugify("!!!") == ""
    assert slugify("Ça va? Très bien.") == "ca-va-tres-bien"
    assert slugify("v1.2.3 (beta)") == "v1-2-3-beta"


def test_memoize_kwargs_and_args():
    calls = []

    @memoize
    def f(a, b=0, *, c=1):
        calls.append((a, b, c))
        return a + b + c

    assert f(1, b=2) == 4
    assert f(1, b=3) == 5
    assert f(1, b=2) == 4
    assert (
        f(1, 2) == 4
    )  # positional and keyword calls are allowed to share or not, but must be right
    assert f(1, b=2, c=5) == 8
    assert f(1, c=5, b=2) == 8
    assert (
        len([x for x in calls if x == (1, 2, 5)]) == 1
    )  # keyword order does not matter


def test_memoize_basics():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4 and f(2) == 4 and f(3) == 6
    assert calls == [2, 3]
    f.cache_clear()
    assert f(2) == 4 and calls == [2, 3, 2]
    assert f.__name__ == "f"


def test_table_unchanged():
    out = render([["ab", 5], ["c", 120]], ["name", "n"])
    assert out.splitlines() == ["name  n", "----  ---", "ab      5", "c     120"]
