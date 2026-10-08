"""GC policy: freeze, thresholds, and a bounded, refcounted first-token hold."""

import gc
import weakref

import pytest

from yunshu_engine import gc_policy as gp
from yunshu_engine.settings import REGISTRY


class FakeGC:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calls = []
        self.threshold = (700, 10, 10)
        self.count0 = 0
        self.frozen = 0

    def isenabled(self):
        return self.enabled

    def disable(self):
        self.enabled = False
        self.calls.append("disable")

    def enable(self):
        self.enabled = True
        self.calls.append("enable")

    def get_count(self):
        return (self.count0, 0, 0)

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, *t):
        self.threshold = t
        self.calls.append(("threshold", t))

    def collect(self, *a):
        self.calls.append("collect")

    def freeze(self):
        self.frozen = 99
        self.calls.append("freeze")

    def unfreeze(self):
        self.frozen = 0
        self.calls.append("unfreeze")

    def get_freeze_count(self):
        return self.frozen


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def guard(fake, clock=None, **kw):
    return gp.TtftGcGuard(fake, clock=clock or Clock(), watchdog=False, **kw)


def test_setting_is_registered_with_policies():
    s = REGISTRY["YUNSHU_GC_POLICY"]
    assert s.choices == gp.POLICIES and s.stability == "stable"


def test_default_policy_touches_nothing():
    f = FakeGC()
    assert gp.install("default", f) is None
    assert f.calls == [] and f.threshold == (700, 10, 10)


def test_freeze_collects_first_then_freezes_only():
    f = FakeGC()
    assert gp.install("freeze", f) is None
    assert f.calls == ["collect", "freeze"] and f.threshold == (700, 10, 10)


def test_defer_freezes_raises_gen2_and_returns_guard():
    f = FakeGC()
    g = gp.install("defer", f)
    assert isinstance(g, gp.TtftGcGuard)
    assert f.calls[:2] == ["collect", "freeze"]
    assert f.threshold == (700, 10, gp.GEN2_MULTIPLIER)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        gp.install("bogus", FakeGC())


def test_hold_is_refcounted_across_requests():
    f = FakeGC()
    g = guard(f)
    g.begin("a")
    g.begin("b")
    assert not f.enabled and f.calls.count("disable") == 1
    g.first_token("a")
    assert not f.enabled  # b still waits
    g.first_token("b")
    assert f.enabled and g.holds == 1
    g.first_token("b")  # idempotent
    assert f.calls == ["disable", "enable"]


def test_never_overrides_a_collector_the_caller_disabled():
    f = FakeGC(enabled=False)
    g = guard(f)
    g.begin("a")
    g.first_token("a")
    assert not f.enabled and f.calls == []


def test_hold_released_after_time_bound():
    f, c = FakeGC(), Clock()
    g = guard(f, c, max_hold_s=10)
    g.begin("a")
    c.t = 9
    assert not g.check() and not f.enabled
    c.t = 11
    assert g.check() and f.enabled and g.forced_releases == 1
    g.first_token("a")  # late first token must not double-enable
    assert f.calls == ["disable", "enable"]


def test_hold_released_after_allocation_bound():
    f = FakeGC()
    g = guard(f, max_allocs=1000)
    g.begin("a")
    f.count0 = 1001
    assert g.check() and f.enabled


def test_memory_stays_bounded_real_gc():
    """Cyclic garbage made during a hold is reclaimed once the hold ends."""

    class Node:
        pass

    g = gp.TtftGcGuard(max_allocs=10**9, watchdog=False)
    was = gc.isenabled()
    gc.enable()
    try:
        g.begin("r")
        refs = []
        for _ in range(2000):
            a, b = Node(), Node()
            a.o, b.o = b, a
            refs.append(weakref.ref(a))
        del a, b
        assert all(r() is not None for r in refs)  # nothing collected while held
        g.first_token("r")
        assert gc.isenabled()
        gc.collect()
        assert all(r() is None for r in refs)
    finally:
        if not was:
            gc.disable()


def test_real_watchdog_releases_a_stuck_hold():
    import time

    was = gc.isenabled()
    gc.enable()
    g = gp.TtftGcGuard(max_hold_s=0.05, poll_s=0.01)
    try:
        g.begin("never-emits")
        deadline = time.monotonic() + 3
        while not gc.isenabled() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gc.isenabled() and g.forced_releases == 1
    finally:
        if not was:
            gc.disable()
