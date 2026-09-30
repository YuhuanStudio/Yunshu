from textkit import memoize


def test_caches_positional():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4 and f(2) == 4
    assert calls == [2]


def test_kwargs_distinguish_calls():
    @memoize
    def f(a, b=0):
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=3) == 4
