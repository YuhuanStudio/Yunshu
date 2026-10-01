from yunshu_gateway.token_compare import tokens_equal


def test_non_ascii_does_not_raise():
    assert tokens_equal("é", "configured-secret") is False
    assert tokens_equal("é\udcff", "é\udcff") is True


def test_valid_and_empty():
    assert tokens_equal("abc", "abd") is False
    assert tokens_equal("abc", "abc")
    assert not tokens_equal("", "")
    assert not tokens_equal(None, "x")
