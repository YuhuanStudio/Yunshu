"""One request's grammar dead end / admission failure must not take other requests down."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.unit.test_vlm_runner_batching import FakeDriver, runner  # noqa: F401,F811
from yunshu_engine import vlm_batch_runner as vbr


class DeadConstraint:
    def get_allowed_tokens(self, tokenizer, generated):
        return []


class ProcessingDriver(FakeDriver):
    """FakeDriver that runs each row's logits processors like the driver's decode step."""

    def step(self):
        logits = mx.zeros((1, 4))
        from yunshu_engine.constraint_eos import GrammarDeadEndError

        for row in self.rows:
            for proc in row[0].processors:
                proc(mx.array([1]), logits)
            guide = row[0].guide
            if guide is not None:  # what RoundDriver._draw does with a guide's mask
                try:
                    guide.mask()
                except GrammarDeadEndError as exc:
                    guide.fault = exc
        return super().step()


@pytest.fixture
def pgen_runner():
    r = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), processor=None)
    r.driver = ProcessingDriver()
    return r


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
    good = res["good"]
    assert isinstance(good, list) and len(good) == 5, good
    assert good == list(range(good[0], good[0] + 5))  # an uninterrupted run
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


def _production_guide_kwargs():
    """The ``guide=`` / ``logits_processors=`` vlm_engine hands the runner for a json_schema."""
    from yunshu_engine.vlm_engine import VLMEngine

    class Dead:
        def get_allowed_tokens(self, tokenizer, generated):
            return []

        def rollback(self):
            pass

    seen = {}

    def iter_tokens(ids, **kw):
        seen.update(kw)
        return iter(())

    eng = VLMEngine.__new__(VLMEngine)
    eng._config = {"vocab_size": 8}
    eng._tokenizer = SimpleNamespace(
        detokenizer=SimpleNamespace(
            reset=lambda: None, finalize=lambda: None, last_segment=""
        )
    )
    eng._reasoning_markers = lambda: (None, None, "<think>", "</think>", False)
    eng._get_eos_ids = lambda: [7]
    eng._tool_guide = lambda *_: None
    eng._generation_suppress_ids = lambda: None
    eng._build_text_constraint = lambda _: Dead()
    eng._batch_runner = SimpleNamespace(iter_tokens=iter_tokens)
    list(
        eng._runner_events_impl(
            [1, 2, 3],
            max_tokens=5,
            temperature=0,
            top_p=1,
            top_k=0,
            min_p=0,
            seed=0,
            stop=None,
            stop_token_ids=None,
            repetition_penalty=1,
            frequency_penalty=0,
            presence_penalty=0,
            logit_bias=None,
            json_schema={"type": "object"},
            enable_thinking=False,
            thinking_budget=None,
            cancel_event=None,
            stats=SimpleNamespace(finish_reason="stop"),
        )
    )
    assert seen["guide"] is not None
    return {"guide": seen["guide"], "logits_processors": seen["logits_processors"]}


def test_production_json_schema_dead_end_fails_only_that_row(pgen_runner):
    res = _two_requests(pgen_runner, _production_guide_kwargs())
    assert isinstance(res["bad"], ValueError)
    assert "no valid next token" in str(res["bad"])
    good = res["good"]
    assert isinstance(good, list) and len(good) == 5, good
    assert good == list(range(good[0], good[0] + 5))  # an uninterrupted run
    assert not pgen_runner.busy()


def test_faulted_row_never_publishes_its_unmasked_first_token(pgen_runner):
    proc = vbr.ConstraintProcessor(DeadConstraint(), tokenizer=None)
    got = []
    with pytest.raises(ValueError, match="no valid next token"):
        for token in pgen_runner.iter_tokens(
            [1, 2, 3], max_tokens=5, prompt_kwargs={}, logits_processors=[proc]
        ):
            got.append(token)
    assert got == []
    assert not pgen_runner.busy()


def _plain_job():
    return vbr._Job(
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


def test_admitted_jobs_are_not_reached_again_by_the_failure_handler(
    runner,  # noqa: F811
    monkeypatch,
):
    """A fails nothing; B's own bookkeeping raises after A was admitted, outside the per-job try."""
    admitted = []

    class Cancel:
        def is_set(self):
            if admitted:
                raise RuntimeError("bookkeeping exploded")
            return False

    first, second = _plain_job(), _plain_job()
    second.cancel_event = Cancel()
    with runner._lock:
        runner._pending.extend([first, second])
    real_admit = runner._admit

    def admit(job, alone):
        real_admit(job, alone)
        admitted.append(job)

    monkeypatch.setattr(runner, "_admit", admit)
    seen = []
    real_emit = runner._emit

    def spy(j, item):
        if isinstance(item, BaseException):
            seen.append(j)
        real_emit(j, item)

    monkeypatch.setattr(runner, "_emit", spy)
    runner._drive_slice_body(resubmit=False)
    assert admitted == [first]
    assert sorted(map(id, seen)) == sorted(map(id, [first, second]))  # each once
