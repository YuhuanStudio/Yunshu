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
