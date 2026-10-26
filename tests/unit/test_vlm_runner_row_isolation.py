"""One request's grammar dead end / admission failure must not take other requests down."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.unit.test_vlm_runner_batching import FakeGen, runner  # noqa: F401,F811
from yunshu_engine import vlm_batch_runner as vbr


class DeadConstraint:
    def get_allowed_tokens(self, tokenizer, generated):
        return []


class ProcessingGen(FakeGen):
    """FakeGen that runs each row's logits processors like upstream's decode step."""

    def insert(self, prompts, max_tokens, prompt_kwargs, logits_processors, **kw):
        (uid,) = super().insert(
            prompts, max_tokens, prompt_kwargs, logits_processors, **kw
        )
        self.procs = getattr(self, "procs", {})
        self.procs[uid] = logits_processors[0] or []
        return [uid]

    def next(self):
        logits = mx.zeros((1, 4))
        for uid in list(self.rows):
            for proc in self.procs[uid]:
                proc(mx.array([1]), logits)
        return super().next()


@pytest.fixture
def pgen_runner(monkeypatch):
    import importlib

    ar = importlib.import_module("mlx_vlm.generate.ar")
    FakeGen.instances = []
    monkeypatch.setattr(ar, "BatchGenerator", ProcessingGen)
    return vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), processor=None)


def _two_requests(rn, bad_kwargs):
    ex = ThreadPoolExecutor(max_workers=1)
    rn._executor = ex
    gate = threading.Event()
    queued = threading.Event()
    ex.submit(gate.wait)
    submit = rn._submit

    def submit_together(job):
        submit(job)
        with rn._lock:
            if len(rn._pending) == 2:
                queued.set()

    rn._submit = submit_together
    res = {}

    def consume(name, kw):
        try:
            res[name] = list(
                rn.iter_tokens([1, 2, 3], max_tokens=5, prompt_kwargs={}, **kw)
            )
        except BaseException as exc:  # noqa: BLE001
            res[name] = exc

    threads = [
        threading.Thread(target=consume, args=("bad", bad_kwargs)),
        threading.Thread(target=consume, args=("good", {})),
    ]
    for t in threads:
        t.start()
    assert queued.wait(10)
    gate.set()
    for t in threads:
        t.join(10)
        assert not t.is_alive(), "consumer hung"
    ex.shutdown(wait=True)
    return res


def test_grammar_dead_end_fails_only_that_row(pgen_runner):
    proc = vbr.ConstraintProcessor(DeadConstraint(), tokenizer=None)
    res = _two_requests(pgen_runner, {"logits_processors": [proc]})
    assert isinstance(res["bad"], ValueError)
    assert "no valid next token" in str(res["bad"])
    assert res["good"] == [101, 102, 103, 104, 105]  # uid 1's tokens: 100 * uid + step
    assert not pgen_runner.busy()


def test_prefix_wait_failure_fails_the_job_not_hangs(runner, monkeypatch):  # noqa: F811
    def boom(job):
        raise RuntimeError("checkpoint_ready exploded")

    monkeypatch.setattr(runner, "_prefix_wait", boom)
    res = {}

    def go():
        try:
            res["v"] = list(
                runner.iter_tokens([1, 2, 3], max_tokens=3, prompt_kwargs={})
            )
        except BaseException as exc:  # noqa: BLE001
            res["e"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "consumer hung forever"
    assert isinstance(res.get("e"), RuntimeError)
    assert not runner.busy()


def test_popped_jobs_are_emitted_when_the_slice_fails(runner, monkeypatch):  # noqa: F811
    """A job already popped from _pending is in no group; the failure handler must reach it."""
    job = vbr._Job(
        ids=[1],
        max_tokens=3,
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
    with runner._lock:
        runner._pending.append(job)
    calls = {"n": 0}
    real = runner.inflight

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:  # the `alone` computation, after the job left _pending
            raise RuntimeError("slice failed")
        return real()

    monkeypatch.setattr(runner, "inflight", flaky)
    runner._drive_slice_body(resubmit=False)
    item = job.out.get(timeout=1)
    assert isinstance(item, RuntimeError)
