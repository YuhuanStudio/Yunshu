from shop import Cart


def test_add_and_total():
    c = Cart()
    c.add("a", 250, 2)
    c.add("b", 100)
    assert c.subtotal_cents() == 600
    assert c.line_count() == 2


def test_fixed_discount():
    c = Cart()
    c.add("a", 1000)
    assert c.total_cents("FIVEOFF") == 500


def test_unknown_code():
    c = Cart()
    c.add("a", 1000)
    assert c.total_cents("NOPE") == 1000
