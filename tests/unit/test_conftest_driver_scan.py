"""The autouse fixtures find every loaded copy of fullcmp_run.py without scanning the whole heap."""

import gc
import importlib.util
import sys
import time
import types
from pathlib import Path

from tests.unit import conftest as cf

DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "research" / "fullcmp_run.py"


def _gc_scan():
    return [
        o
        for o in gc.get_objects()
        if isinstance(o, types.ModuleType)
        and str(getattr(o, "__file__", "")).endswith("fullcmp_run.py")
    ]


def test_copy_outside_sys_modules_is_found_and_matches_the_heap_scan():
    if not DRIVER.exists():
        import pytest

        pytest.skip("no fullcmp_run.py in this tree")
    spec = importlib.util.spec_from_file_location("fullcmp_copy_unit", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    assert "fullcmp_copy_unit" not in sys.modules
    got = cf._loaded_driver_modules()
    assert any(m is mod for m in got)
    extra = [m for m in _gc_scan() if id(m) not in {id(g) for g in got}]
    assert not extra, [
        (m.__name__, getattr(m, "__spec__", None), m.__dict__.get("__loader__"))
        for m in extra
    ]


def test_lookup_does_not_walk_the_heap():
    junk = [[i] for i in range(300_000)]  # a big heap makes gc.get_objects() slow
    t0 = time.perf_counter()
    for _ in range(20):
        cf._loaded_driver_modules()
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    _gc_scan()
    slow_once = time.perf_counter() - t0
    assert fast < slow_once * 20 / 5, (fast, slow_once)
    assert junk
