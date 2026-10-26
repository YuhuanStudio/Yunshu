"""A checkpoint whose recurrent (GDN) state is inf/nan is never published to the prefix cache."""

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_apc_manager import _cache, _mgr  # noqa: E402


def _poisoned(value, dtype=mx.float16):
    cache = _cache(96)
    cache[0].cache = [mx.ones((1, 4, 8), dtype=dtype), mx.full((1, 2, 3), value, dtype)]
    mx.eval(cache[0].cache)
    return cache


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_non_finite_recurrent_state_is_not_stored(bad):
    m = _mgr()
    ids = list(range(1000, 1096))
    m.begin_request()
    assert m.store_exact_cache(ids, _poisoned(bad)) is False
    assert not m._exact_cache
    assert m.stats.rejects_by_reason.get("non_finite_state") == 1


def test_finite_state_is_still_stored():
    m = _mgr()
    ids = list(range(1000, 1096))
    m.begin_request()
    assert m.store_exact_cache(ids, _cache(96)) is True
    assert len(m._exact_cache) == 1
    assert not m.stats.rejects_by_reason.get("non_finite_state")
