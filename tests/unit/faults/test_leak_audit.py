"""LeakAudit: loop-owned executor threads are not leaks; real stray threads are."""

import threading

import pytest

from .harness import LeakAudit


def _spawn(name):
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, name=name)
    t.start()
    return t, stop


def test_event_loop_executor_thread_is_not_a_leak():
    audit = LeakAudit()
    t, stop = _spawn("asyncio_0")
    try:
        audit.assert_clean()
    finally:
        stop.set()
        t.join()


def test_stray_non_daemon_thread_is_a_leak():
    audit = LeakAudit()
    t, stop = _spawn("stray-worker")
    try:
        with pytest.raises(AssertionError, match="leaked non-daemon threads"):
            audit.assert_clean()
    finally:
        stop.set()
        t.join()
