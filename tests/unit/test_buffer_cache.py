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
    state = dict(buffer_cache._STATE)
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    yield
    buffer_cache._RESTORE_BUFFERS.clear()
    ar.mx = original
    mx.set_cache_limit(previous)
    buffer_cache._STATE.clear()
    buffer_cache._STATE.update(state)


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


def test_over_limit_pool_is_reclaimed_on_allocation_without_full_clear(restore):
    mx.clear_cache()
    buffer_cache.install(1e-6)
    _fill_cache()
    held = mx.get_cache_memory()
    assert held > 0
    ar.mx.clear_cache()
    assert mx.get_cache_memory() == held
    scratch = mx.empty((16,), dtype=mx.float32)
    mx.eval(scratch)
    assert mx.get_cache_memory() <= buffer_cache._STATE["limit"]


def test_zero_limit_clears_every_time(restore):
    mx.clear_cache()
    buffer_cache.install(0)
    _fill_cache()
    ar.mx.clear_cache()
    assert mx.get_cache_memory() == 0


def test_tight_weights_get_small_pool():
    gib = 1 << 30
    total = 128 * gib
    assert buffer_cache.auto_limit_gib_for_weights(total, 70 * gib) == 0.5
    assert buffer_cache.auto_limit_gib_for_weights(total, 16 * gib) == 6.0
    assert buffer_cache.auto_limit_gib_for_weights(total, None) == 6.0
    assert buffer_cache.auto_limit_gib_for_weights(8 * gib, 7 * gib) == pytest.approx(
        0.4
    )


def test_ceiling_decides_when_set():
    gib = 1 << 30
    total = 128 * gib
    assert buffer_cache.auto_limit_gib_for_weights(total, 69 * gib, 74 * gib) == 0.5
    assert buffer_cache.auto_limit_gib_for_weights(total, 69 * gib, 110 * gib) == 6.0


@pytest.mark.parametrize("batched", [False, True])
def test_finished_storage_is_reused_with_exact_prefix_and_zero_tail(restore, batched):
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.apc_manager import _restore_kv

    buffer_cache.install(0.001)
    from mlx_vlm.models.cache import BatchKVCache

    finished = BatchKVCache([0]) if batched else KVCache()
    finished.keys = mx.full((1, 1, 512, 4), 7.0)
    finished.values = mx.full((1, 1, 512, 4), -7.0)
    mx.eval(finished.state)
    size = finished.keys.nbytes + finished.values.nbytes
    buffer_cache.offer_restore_buffers([finished])
    assert buffer_cache.restore_buffer_bytes() == size
    assert mx.get_cache_memory() <= buffer_cache._STATE["limit"] - size
    assert finished.keys is None and finished.values is None
    source = KVCache()
    source.keys = mx.full((1, 1, 128, 4), 0.5)
    source.values = mx.full((1, 1, 128, 4), -0.5)
    source.offset = 128
    targets = []
    out = _restore_kv(source, 300, targets)
    mx.eval(targets)
    assert out.offset == 128
    assert buffer_cache.restore_buffer_bytes() == 0
    for result, src in zip(out.state, source.state, strict=True):
        assert mx.array_equal(result[..., :128, :], src).item()
        assert mx.all(result[..., 128:, :] == 0).item()


def test_storage_respects_pool_budget_and_trim(restore):
    from mlx_vlm.models.cache import KVCache

    buffer_cache.install(0)
    c = KVCache()
    c.keys = mx.ones((1, 1, 256, 4))
    c.values = mx.ones((1, 1, 256, 4))
    buffer_cache.offer_restore_buffers([c])
    assert buffer_cache.restore_buffer_bytes() == 0
    assert c.keys is not None
    buffer_cache.install(0.001)
    buffer_cache.offer_restore_buffers([c])
    assert buffer_cache.restore_buffer_bytes() > 0
    buffer_cache.clear_restore_buffers()
    assert buffer_cache.restore_buffer_bytes() == 0


def test_rehearsal_restore_preserves_finished_storage(restore):
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.apc_manager import _RESTORE_STORAGE, _restore_kv

    buffer_cache.install(0.001)
    finished = KVCache()
    finished.keys = mx.ones((1, 1, 512, 4))
    finished.values = mx.ones((1, 1, 512, 4))
    mx.eval(finished.state)
    buffer_cache.offer_restore_buffers([finished])
    before = buffer_cache.restore_buffer_bytes()
    source = KVCache()
    source.keys = source.values = mx.zeros((1, 1, 128, 4))
    source.offset = 128
    token = _RESTORE_STORAGE.set(False)
    try:
        targets = []
        _restore_kv(source, 300, targets)
        mx.eval(targets)
    finally:
        _RESTORE_STORAGE.reset(token)
    assert buffer_cache.restore_buffer_bytes() == before


def test_only_exact_rehearsal_lookup_excludes_storage():
    from yunshu_engine.apc_manager import _RESTORE_STORAGE, YunshuAPCManager

    manager = object.__new__(YunshuAPCManager)
    manager.keep_prior_for = (1, 2, 3)
    manager._lookup_exact_cache = lambda *a, **kw: _RESTORE_STORAGE.get()
    assert manager.lookup_exact_cache([1, 2, 3]) is False
    assert manager.lookup_exact_cache([1, 2]) is True
    assert manager.lookup_exact_cache([1, 2, 3, 4]) is True
    assert _RESTORE_STORAGE.get() is True


def test_restore_buffer_is_best_fit_and_bounded(restore):
    buffer_cache.install(0.01)
    big = mx.zeros((1, 1, 4096, 4))
    mid = mx.zeros((1, 1, 512, 4))
    small = mx.zeros((1, 1, 256, 4))
    mx.eval(big, mid, small)
    buffer_cache._RESTORE_BUFFERS[:] = [big, mid, small]
    probe = mx.zeros((1, 1, 10, 4))
    assert (
        buffer_cache.take_restore_buffer(probe, 300) is mid
    )  # best fit, not first fit
    assert buffer_cache.take_restore_buffer(probe, 300) is None  # 4096 > 2 x 300
    assert buffer_cache.take_restore_buffer(probe, 200) is small
    assert buffer_cache.take_restore_buffer(probe, 3000) is big


def _finished(rows):
    from mlx_vlm.models.cache import KVCache

    c = KVCache()
    c.keys = mx.ones((1, 1, rows, 4))
    c.values = mx.ones((1, 1, rows, 4))
    mx.eval(c.state)
    return c


def test_pooled_set_nobody_takes_is_released(restore):
    """After one long turn, short chats (under half its capacity) must not pin it forever."""
    buffer_cache.install(0.01)
    buffer_cache.offer_restore_buffers([_finished(2048)])
    big = buffer_cache.restore_buffer_bytes()
    assert big > 0
    small = _finished(64)
    small_bytes = small.keys.nbytes + small.values.nbytes
    for _ in range(buffer_cache.MAX_IDLE_OFFERS - 1):
        buffer_cache.offer_restore_buffers([_finished(64)])
        assert buffer_cache.restore_buffer_bytes() == big  # still waiting for a taker
    buffer_cache.offer_restore_buffers([small])
    assert buffer_cache.restore_buffer_bytes() == small_bytes
    assert buffer_cache._STATE["limit"] - small_bytes >= 0


def test_pooled_set_that_is_taken_stays(restore):
    buffer_cache.install(0.01)
    buffer_cache.offer_restore_buffers([_finished(512)])
    probe = mx.zeros((1, 1, 10, 4))
    assert buffer_cache.take_restore_buffer(probe, 400) is not None  # the keys
    for _ in range(buffer_cache.MAX_IDLE_OFFERS - 1):
        buffer_cache.offer_restore_buffers([_finished(8)])
    assert buffer_cache.restore_buffer_bytes() > 0  # the take reset the idle count
