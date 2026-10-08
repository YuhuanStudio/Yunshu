"""Job lifecycle of VLMBatchRunner: executor failure, cancel before admit, output bound."""

import threading
from concurrent.futures import ThreadPoolExecutor

from tests.unit.test_vlm_runner_batching import FakeGen, runner  # noqa: F401,F811
from yunshu_engine import vlm_batch_runner as vbr


def _run(rn, n=3, **kw):
    return list(rn.iter_tokens([1, 2, 3], max_tokens=n, prompt_kwargs={}, **kw))


class BrokenExecutor:
    def submit(self, *a, **k):
        raise RuntimeError("cannot schedule new futures after shutdown")


class FailSecondExecutor:
    """Runs the first slice, refuses the resubmit."""

    def __init__(self):
        self.real = ThreadPoolExecutor(max_workers=1)
        self.calls = 0

    def submit(self, fn, *a, **k):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("executor gone")
        return self.real.submit(fn, *a, **k)


def _in_thread(fn):
    res = {}

    def go():
        try:
            res["v"] = fn()
        except BaseException as e:  # noqa: BLE001
            res["e"] = e

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "consumer hung forever"
    return res


def test_r08_submit_failure_fails_consumer_not_hangs(runner):  # noqa: F811
    runner._executor = BrokenExecutor()
    res = _in_thread(lambda: _run(runner))
    assert isinstance(res.get("e"), RuntimeError)
    assert not runner.busy()
    assert runner._driving is False


def test_r08_resubmit_failure_fails_active_jobs(runner):  # noqa: F811
    ex = FailSecondExecutor()
    runner._executor = ex
    res = _in_thread(lambda: _run(runner, n=50))
    assert isinstance(res.get("e"), RuntimeError)
    assert not runner.busy()
    ex.real.shutdown(wait=True)


def test_r10_cancelled_before_admit_never_prefills(runner):  # noqa: F811
    ev = threading.Event()
    ev.set()
    assert _run(runner, cancel_event=ev) == []
    assert FakeGen.instances == [] or all(not g.rows for g in FakeGen.instances)
    assert all(g.next_uid == 0 for g in FakeGen.instances)
    assert not runner.busy()


def test_r09_output_queue_is_bounded(runner, monkeypatch):  # noqa: F811
    monkeypatch.setattr(vbr, "OUT_LIMIT", 8)
    job = vbr._Job(
        ids=[1],
        max_tokens=100,
        greedy=True,
        sampling=None,
        use_draft=False,
        logprobs=False,
        top_logprobs=0,
        processors=[],
        prompt_kwargs={},
        salt=None,
        seed=None,
        cancel_event=None,
        stats=vbr.RunStats(),
    )
    for i in range(1000):
        runner._emit(job, (i, None))
    assert job.out.qsize() <= 8 + 1
    assert job.overflowed
    # Exactly one terminal: an error follows the data, then nothing more.
    runner._emit(job, vbr._DONE)
    items = []
    while not job.out.empty():
        items.append(job.out.get())
    assert sum(isinstance(i, BaseException) for i in items) == 1


def _handoff_job(ack):
    job = vbr._Job(
        ids=[1],
        max_tokens=10,
        greedy=True,
        sampling=None,
        use_draft=True,
        logprobs=False,
        top_logprobs=0,
        processors=[],
        prompt_kwargs={},
        salt=None,
        seed=None,
        cancel_event=None,
        stats=vbr.RunStats(),
        first_delivery=ack,
    )
    job.allow_draft = True
    job.stats.generated = 1
    return job


def test_first_delivery_handoff_publishes_before_wait_and_waits_only_once(runner):  # noqa: F811
    from types import SimpleNamespace

    calls = []
    job = _handoff_job(
        SimpleNamespace(
            wait=lambda bound: calls.append((bound, job.out.get_nowait())) or False
        )
    )
    runner._executor = object()
    runner.draft_kind = "dflash"
    runner._emit(job, (11, None))
    assert calls == [(0.004, (11, None))]
    assert job.first_delivery_waited
    job.stats.generated = 2
    runner._emit(job, (12, None))
    runner._emit(job, vbr._DONE)
    assert len(calls) == 1
    assert job.out.get_nowait() == (12, None)
    assert job.out.get_nowait() is vbr._DONE


def test_non_draft_and_inline_consumers_never_wait_for_delivery(runner):  # noqa: F811
    from types import SimpleNamespace

    def unexpected_wait(bound):
        raise AssertionError("handoff on a path with no competing copy setup")

    for kind, executor, use_draft in (
        ("off", object(), True),
        ("dflash", None, True),
        ("dflash", object(), False),
    ):
        runner.draft_kind = kind
        runner._executor = executor
        job = _handoff_job(SimpleNamespace(wait=unexpected_wait))
        job.use_draft = use_draft
        runner._emit(job, (11, None))
        assert job.out.get_nowait() == (11, None)
        assert not job.first_delivery_waited


def test_failure_detaches_work_before_notifying_consumers(runner, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    job = _handoff_job(None)
    group = SimpleNamespace(jobs={1: job}, gen=SimpleNamespace(close=lambda: None))
    runner._batches["failed"] = group
    runner._driving = True
    emitted = []

    def emit(j, error):
        assert not runner.busy(), "terminal published while failed jobs remain active"
        emitted.append((j, error))

    monkeypatch.setattr(runner, "_emit", emit)
    error = RuntimeError("executor failed")
    runner._fail_all(error)
    assert emitted == [(job, error)]


def test_gc_guard_spans_request_arrival_to_first_emit(runner):  # noqa: F811
    from yunshu_engine.gc_policy import TtftGcGuard

    class Spy(TtftGcGuard):
        def __init__(self):
            super().__init__(watchdog=False)
            self.log = []

        def begin(self, key):
            self.log.append("begin")
            super().begin(key)

        def first_token(self, key):
            self.log.append("first")
            super().first_token(key)

    import gc

    spy = Spy()
    runner.gc_guard = spy
    was = gc.isenabled()
    try:
        assert len(_run(runner, n=3)) >= 1
    finally:
        if was:
            gc.enable()
    assert spy.log[0] == "begin" and "first" in spy.log
    assert gc.isenabled() == was and not spy._held
