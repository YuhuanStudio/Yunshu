"""The finished-request feed ring: a full ring drops the OLDEST row and counts the drop."""

import time

from yunshu_gateway.x_yunshu import _Registry


def _done(reg: _Registry, n: int) -> None:
    for _ in range(n):
        reg.record_done({"t": time.time(), "model": "m"})


def test_full_ring_drops_oldest_and_counts_it():
    reg = _Registry()
    _done(reg, 512)
    assert reg.dropped == 0
    _done(reg, 1)
    assert reg.dropped == 1
    rows, latest = reg.recent_cursor(limit=512)
    assert latest == 513
    assert len(rows) == 512
    # Newest first: the surviving tail is seq 513 down to seq 2; seq 1 is the dropped one.
    assert rows[0]["seq"] == 513
    assert rows[-1]["seq"] == 2


def test_drop_counter_keeps_counting_under_load():
    reg = _Registry()
    _done(reg, 512 + 37)
    assert reg.dropped == 37
    assert reg.recent_cursor(limit=512)[0][-1]["seq"] == 38
