"""`yunshu serve` auth startup helpers: --auth-token auto and weak/missing credential warnings."""

from yunshu_cli.serve import auth_startup_warnings, expand_auto_token


def test_auto_token_is_random_and_long():
    a, gen_a = expand_auto_token("auto")
    b, _ = expand_auto_token("AUTO")
    assert gen_a and a and b and a != b
    assert len(a) >= 32


def test_explicit_or_missing_token_is_untouched():
    assert expand_auto_token("s3cret-value-1234") == ("s3cret-value-1234", False)
    assert expand_auto_token(None) == (None, False)


def test_short_token_warns():
    w = auth_startup_warnings("127.0.0.1", "short", False)
    assert len(w) == 1 and "5 characters" in w[0]
    assert auth_startup_warnings("127.0.0.1", "x" * 16, False) == []


def test_non_loopback_without_any_auth_warns():
    assert auth_startup_warnings("0.0.0.0", None, False)
    assert auth_startup_warnings("0.0.0.0", None, False, auth_disabled=True)
    assert auth_startup_warnings("127.0.0.1", None, False) == []


def test_managed_keys_count_as_auth():
    # stored API keys protect the server even without a static token
    assert auth_startup_warnings("0.0.0.0", None, True) == []
