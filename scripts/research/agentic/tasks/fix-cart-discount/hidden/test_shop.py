import pytest

from shop import Cart


def test_remove_all_units_drops_line():
    c = Cart()
    c.add("a", 100, 2)
    c.add("b", 50)
    c.remove("a", 2)
    assert c.line_count() == 1
    assert c.quantity("a") == 0
    assert c.subtotal_cents() == 50


def test_remove_partial_keeps_line():
    c = Cart()
    c.add("a", 100, 3)
    c.remove("a")
    assert c.line_count() == 1 and c.quantity("a") == 2


def test_remove_more_than_present_drops_line():
    c = Cart()
    c.add("a", 100, 1)
    c.remove("a", 5)
    assert c.line_count() == 0


def test_remove_unknown_raises():
    with pytest.raises(KeyError):
        Cart().remove("zzz")


def test_percent_rounds_half_up():
    c = Cart()
    c.add("a", 5995)
    assert c.total_cents("SAVE10") == 5396  # 5395.5 -> 5396


def test_percent_ordinary():
    c = Cart()
    c.add("a", 1999, 3)
    assert c.total_cents("SAVE10") == 5397  # 5397.3 -> 5397


def test_percent_rounding_directions():
    c = Cart()
    c.add("a", 99)
    assert c.total_cents("SAVE25") == 74  # 74.25 -> 74
    c2 = Cart()
    c2.add("a", 1001)
    assert c2.total_cents("SAVE25") == 751  # 750.75 -> 751


def test_other_behaviour_unchanged():
    c = Cart()
    c.add("a", 1000)
    assert c.total_cents("FIVEOFF") == 500
    assert c.total_cents("NOPE") == 1000
    assert c.total_cents() == 1000
    with pytest.raises(ValueError):
        c.add("z", 1, 0)
