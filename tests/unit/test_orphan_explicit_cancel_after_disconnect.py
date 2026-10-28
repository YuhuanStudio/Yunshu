"""Stop = disconnect, then the cancel endpoint: the parked generation must stop, not decode
through the orphan window and replay to a regenerate."""

import asyncio
import threading

from tests.unit.test_orphan_adoption import FakeEngine, _take
from yunshu_engine import orphan_adoption as oa
from yunshu_engine.request_tracker import RequestTracker


def test_cancel_by_request_id_stops_a_parked_orphan():
    async def main():
        reg = oa.get_registry()
        reg.cancel_all()
        eng = FakeEngine(n=500)
        tracker = RequestTracker()
        gen = tracker.register("chatcmpl-1", "m")
        it = oa.run("kc", eng.start, gen.cancel_event, 60.0, 2)
        await _take(it, 2)
        gen.cancel_event.set()  # the client vanished: a disconnect, not a cancel
        await it.aclose()
        tracker.unregister("chatcmpl-1")  # the HTTP request is over
        assert len(reg.snapshot()) == 1 and not eng.cancels[0].is_set()

        assert tracker.cancel("chatcmpl-1") is True  # the cancel endpoint, afterwards
        assert eng.cancels[0].is_set() and reg.snapshot() == []
        assert tracker.cancel("chatcmpl-1") is False  # nothing left to cancel
        # an identical regenerate starts fresh instead of replaying the stale text
        it2 = oa.run("kc", eng.start, threading.Event(), 60.0, 2)
        await _take(it2, 1)
        assert eng.starts == 2
        await it2.aclose()
        reg.cancel_all()

    asyncio.run(main())
