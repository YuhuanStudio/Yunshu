"""Orphan adoption: a vanished client's generation is kept and adopted by an identical retry."""

import asyncio
import threading
from types import SimpleNamespace

from yunshu_engine import orphan_adoption as oa
from yunshu_engine.request import RequestOutput


class FakeEngine:
    """Generates n tokens, one per `tick` seconds; counts tokens actually produced."""

    def __init__(self, n=20, tick=0.01):
        self.n, self.tick = n, tick
        self.produced = 0
        self.starts = 0
        self.cancels: list[threading.Event] = []

    def start(self, real_cancel):
        self.starts += 1
        self.cancels.append(real_cancel)
        return self._gen(real_cancel)

    async def _gen(self, cancel):
        for i in range(self.n):
            if cancel.is_set():
                yield RequestOutput(
                    request_id="r", finished=True, finish_reason="cancel"
                )
                return
            await asyncio.sleep(self.tick)
            self.produced += 1
            last = i == self.n - 1
            yield RequestOutput(
                request_id="r",
                new_text=f"t{i} ",
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
            )


async def _take(it, k):
    out = []
    async for o in it:
        out.append(o)
        if len(out) >= k:
            break
    return out


def _run(coro):
    return asyncio.run(coro)


def test_retry_adopts_orphan_and_work_is_not_repeated():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=30)
        c1 = threading.Event()
        it = oa.run("k", eng.start, c1, 5.0, 2, reg)
        got = await _take(it, 5)
        c1.set()  # client timeout: disconnect
        await it.aclose()
        assert len(reg.snapshot()) == 1  # parked, not cancelled
        assert not eng.cancels[0].is_set()
        await asyncio.sleep(0.1)  # keeps decoding in the background
        c2 = threading.Event()
        t0 = asyncio.get_running_loop().time()
        outs = [o async for o in oa.run("k", eng.start, c2, 5.0, 2, reg)]
        assert eng.starts == 1
        assert eng.produced == 30
        text = "".join(o.new_text for o in outs)
        assert text == "".join(f"t{i} " for i in range(30))
        assert text.startswith("".join(o.new_text for o in got))
        assert outs[-1].finish_reason == "stop"
        assert reg.adopted_total == 1 and reg.snapshot() == []
        assert asyncio.get_running_loop().time() - t0 < 0.5

    _run(main())


def test_unadopted_orphan_is_cancelled_at_window_end():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=500, tick=0.01)
        c1 = threading.Event()
        it = oa.run("k", eng.start, c1, 0.2, 2, reg)
        await _take(it, 2)
        c1.set()
        await it.aclose()
        await asyncio.sleep(0.5)
        assert eng.cancels[0].is_set()
        assert reg.expired_total == 1 and reg.snapshot() == []
        assert eng.produced < 100

    _run(main())


def test_explicit_cancel_stops_at_once():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=500)
        c1 = threading.Event()
        it = oa.run("k", eng.start, c1, 60.0, 2, reg)
        await _take(it, 2)
        c1.explicit_cancel = True
        c1.set()
        await it.aclose()
        assert eng.cancels[0].is_set() and reg.snapshot() == []

    _run(main())


def test_two_live_identical_requests_are_not_merged():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=10)
        a = oa.run("k", eng.start, threading.Event(), 5.0, 2, reg)
        b = oa.run("k", eng.start, threading.Event(), 5.0, 2, reg)
        ra, rb = await asyncio.gather(_take(a, 99), _take(b, 99))
        assert eng.starts == 2 and len(ra) == len(rb) == 10

    _run(main())


def test_different_key_does_not_adopt():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=50)
        c1 = threading.Event()
        it = oa.run("k1", eng.start, c1, 5.0, 2, reg)
        await _take(it, 2)
        c1.set()
        await it.aclose()
        await _take(oa.run("k2", eng.start, threading.Event(), 5.0, 2, reg), 1)
        assert eng.starts == 2
        reg.cancel_all()

    _run(main())


def test_orphan_limit_evicts_oldest_and_zero_grace_cancels():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=500)
        for k in ("a", "b", "c"):
            c = threading.Event()
            it = oa.run(k, eng.start, c, 60.0, 2, reg)
            await _take(it, 1)
            c.set()
            await it.aclose()
        assert len(reg.snapshot()) == 2 and eng.cancels[0].is_set()
        c = threading.Event()
        it = oa.run("z", eng.start, c, 0.0, 2, reg)
        await _take(it, 1)
        c.set()
        await it.aclose()
        assert eng.cancels[-1].is_set()
        reg.cancel_all()

    _run(main())


def test_finished_orphan_replays_after_original_ended():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=5)
        c1 = threading.Event()
        it = oa.run("k", eng.start, c1, 5.0, 2, reg)
        await _take(it, 1)
        c1.set()
        await it.aclose()
        await asyncio.sleep(0.3)  # generation ended while orphaned
        outs = [o async for o in oa.run("k", eng.start, threading.Event(), 5.0, 2, reg)]
        assert eng.starts == 1 and len(outs) == 5

    _run(main())


def test_request_key_ignores_volatile_and_separates_params():
    k1 = oa.request_key(
        "m",
        [{"role": "user", "content": "hi"}],
        {"temperature": 0.7, "timeout_seconds": 5, "cancel_event": object()},
    )
    k2 = oa.request_key(
        "m", [{"role": "user", "content": "hi"}], {"temperature": 0.7, "priority": 3}
    )
    k3 = oa.request_key("m", [{"role": "user", "content": "hi"}], {"temperature": 0.8})
    k4 = oa.request_key(
        "m",
        [{"role": "user", "content": "hi"}],
        {"temperature": 0.7, "tools": [{"a": 1}]},
    )
    assert k1 == k2 and k1 != k3 and k1 != k4
    assert oa.request_key("m", [], {"x": lambda: 1}) is None


def test_tracker_marks_explicit_cancel():
    from yunshu_engine.request_tracker import RequestTracker

    t = RequestTracker()
    g = t.register("r1", "m")
    assert not oa.explicitly_cancelled(g.cancel_event)
    t.cancel("r1")
    assert oa.explicitly_cancelled(g.cancel_event) and g.cancel_event.is_set()
    _ = SimpleNamespace


def test_adopter_gets_rebased_stats_so_replay_is_no_speed():
    from dataclasses import dataclass

    @dataclass
    class Stats:
        t_submit: float = 1.0
        t_admit: float = 1.0
        t_first: float = 2.0
        t_last: float = 50.0
        generated: int = 0
        prefill_done: int = 0
        prefill_total: int = 5
        finish_reason: str | None = None

    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=12)
        orig = Stats(generated=999)

        def start(real_cancel):
            real_cancel.run_stats = orig
            return eng.start(real_cancel)

        c1 = threading.Event()
        it = oa.run("k", start, c1, 5.0, 2, reg)
        await _take(it, 4)
        c1.set()
        await it.aclose()
        await asyncio.sleep(0.05)
        c2 = threading.Event()
        outs = []
        async for o in oa.run("k", start, c2, 5.0, 2, reg):
            outs.append(o)
        mine = c2.run_stats
        assert mine is not orig
        assert mine.replayed >= 4 and mine.generated == 12 - mine.replayed
        assert mine.t_first > orig.t_last and mine.finish_reason == "stop"

    _run(main())


def test_server_made_seed_is_not_identity():
    from yunshu_gateway.routers.chat import _per_choice_seed

    m = [{"role": "user", "content": "x"}]
    a = oa.request_key("m", m, {"seed": None, "temperature": 0.7})
    b = oa.request_key("m", m, {"seed": _per_choice_seed(None, 0), "temperature": 0.7})
    c = oa.request_key("m", m, {"seed": _per_choice_seed(None, 0), "temperature": 0.7})
    assert a == b == c
    assert a != oa.request_key("m", m, {"seed": 5, "temperature": 0.7})
    assert _per_choice_seed(7, 0) == 7


def test_omitted_defaults_match_spelled_out_defaults():
    m = [{"role": "user", "content": "x"}]
    a = oa.request_key(
        "m", m, {"max_tokens": 10, "temperature": 0.7, "stop": [], "logprobs": False}
    )
    b = oa.request_key("m", m, {"max_tokens": 10, "top_p": 1.0, "seed": None})
    assert a == b
    assert a != oa.request_key("m", m, {"max_tokens": 10, "temperature": 0.0})


def test_deliberate_break_without_disconnect_cancels_at_once():
    async def main():
        reg = oa.OrphanRegistry()
        eng = FakeEngine(n=500)
        c1 = threading.Event()
        it = oa.run("k", eng.start, c1, 60.0, 2, reg)
        await _take(it, 2)
        await it.aclose()  # the caller just stopped reading; no disconnect signalled
        assert eng.cancels[0].is_set() and reg.snapshot() == [] and c1.is_set()

    _run(main())
