import pytest

mx = pytest.importorskip("mlx.core")
from yunshu_engine import const_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _restore():
    yield
    const_cache.uninstall()


def test_arange_memoized_and_identical():
    view = const_cache._CachedMx()
    a = view.arange(0, 8, dtype=mx.int32)
    b = view.arange(0, 8, dtype=mx.int32)
    assert a is b
    assert mx.array_equal(a, mx.arange(0, 8, dtype=mx.int32)).item()
    assert view.arange(8) is view.arange(8)
    assert view.arange(8) is not view.arange(8, dtype=mx.int64)


def test_arrays_and_large_ranges_are_not_cached():
    view = const_cache._CachedMx()
    assert view.arange(0, 5, 1.0) is not view.arange(0, 5, 1.0)
    big = const_cache.MAX_LEN + 1
    assert view.arange(big) is not view.arange(big)
    assert view.arange(0.0, 1.0, 0.25).shape == (4,)


def test_other_attributes_forward_and_install():
    view = const_cache._CachedMx()
    assert view.zeros((2,)).shape == (2,)
    assert const_cache.install()
    from mlx_vlm.models.qwen4_exp import language

    assert isinstance(language.mx, const_cache._CachedMx)
    assert const_cache.install()
