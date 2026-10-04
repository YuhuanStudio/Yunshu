"""A reshaped input reuses standard lane sums without unbounded retention."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "lane_sums_bookkeeping",
    Path(__file__).parents[2] / "scripts/research/lane_sums_bookkeeping.py",
)
sums = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sums)


def array(shape):
    size = 1
    for n in shape:
        size *= n
    return SimpleNamespace(shape=shape, size=size)


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    monkeypatch.setattr(sums.q, "_xs_cache", {})
    return sums.q._xs_cache


def test_four_projections_keep_the_same_exact_sums(cache):
    original, exact = array((1, 16, 5120)), array((80, 16))
    first = array((16, 5120))
    cache[id(first)] = (first, exact)
    assert sums.remember(original, first, 64)
    for _ in range(12):
        view = array((16, 5120))
        assert sums.reuse(original, view, 64)
        assert cache[id(view)] == (view, exact)
        assert sums.remember(original, view, 64)
        assert len(cache) <= 4


@pytest.mark.parametrize("damage", ["identity", "rows", "width", "group", "size"])
def test_mismatched_input_or_geometry_does_not_reuse(cache, damage):
    original, view, exact = array((1, 8, 64)), array((8, 64)), array((1, 16))
    cache[id(view)] = (view, exact)
    assert sums.remember(original, view, 64)
    candidate, next_view, group = original, array((8, 64)), 64
    if damage == "identity":
        cache[sums._input_key(original, 64)] = (array(original.shape), exact)
    elif damage == "rows":
        next_view = array((129, 64))
    elif damage == "width":
        next_view = array((16, 32))
    elif damage == "group":
        group = 32
    else:
        next_view.size += 1
    assert not sums.reuse(candidate, next_view, group)
    assert sums._projection_key(next_view, group) not in cache


def test_group_32_has_its_own_standard_projection_key(cache):
    original, view, exact = array((1, 8, 64)), array((8, 64)), array((2, 16))
    cache[(id(view), 32)] = (view, exact)
    assert sums.remember(original, view, 32)
    next_view = array((8, 64))
    assert sums.reuse(original, next_view, 32)
    assert cache[(id(next_view), 32)][1] is exact
    assert not sums.reuse(original, array((8, 64)), 64)


def test_missing_projection_never_creates_or_recomputes_sums(cache):
    assert not sums.remember(array((1, 8, 64)), array((8, 64)), 64)
    assert cache == {}
