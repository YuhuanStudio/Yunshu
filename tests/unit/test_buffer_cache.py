"""The generator's clear_cache keeps the buffer cache until it grows past the limit."""

import pytest

mx = pytest.importorskip("mlx.core")
ar = pytest.importorskip("mlx_vlm.generate.ar")

from yunshu_engine.kernels import buffer_cache  # noqa: E402


def test_default_pool_scales_with_physical_ram():
    gib = 1 << 30
    assert buffer_cache.auto_limit_gib(128 * gib) == 6.0
    assert buffer_cache.auto_limit_gib(32 * gib) == pytest.approx(1.6)
    assert buffer_cache.auto_limit_gib(8 * gib) == pytest.approx(0.4)
    assert buffer_cache.auto_limit_gib(0) == 0.0


@pytest.fixture
def restore():
    original = ar.mx
    yield
    ar.mx = original
    buffer_cache._STATE["limit"] = 0


def _fill_cache():
    x = mx.ones((1024, 1024, 8))
    mx.eval(x * 2)
    del x
    mx.synchronize()


def test_small_cache_survives_generator_clear(restore):
    mx.clear_cache()
    buffer_cache.install(1.0)
    _fill_cache()
    held = mx.get_cache_memory()
    assert held > 0
    ar.mx.clear_cache()
    assert mx.get_cache_memory() == held
    assert ar.mx.array is mx.array


def test_cache_over_the_limit_is_cleared(restore):
    mx.clear_cache()
    buffer_cache.install(1e-6)
    _fill_cache()
    ar.mx.clear_cache()
    assert mx.get_cache_memory() == 0


def test_zero_limit_clears_every_time(restore):
    mx.clear_cache()
    buffer_cache.install(0)
    _fill_cache()
    ar.mx.clear_cache()
    assert mx.get_cache_memory() == 0
