from textkit import render


def test_render():
    out = render([["ab", 5], ["c", 120]], ["name", "n"])
    assert out.splitlines() == ["name  n", "----  ---", "ab      5", "c     120"]
