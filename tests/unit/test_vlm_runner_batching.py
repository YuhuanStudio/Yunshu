"""Shared-batch scheduling in VLMBatchRunner, with a fake upstream generator."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from yunshu_engine import vlm_batch_runner as vbr


class FakeGen:
    """Emits token ``100*row + step`` per active row per next(); finishes at max."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.rows = {}
        self.next_uid = 0
        self.max_batch = 0
        self.removed = []
        self.closed = False
        FakeGen.instances.append(self)

    def insert(
        self,
        prompts,
        max_tokens,
        prompt_kwargs,
        logits_processors,
        thinking_budget_criteria=None,
    ):
        assert self.kwargs.get("prefill_batch_size") == 1
        self.budgets = thinking_budget_criteria
        uid = self.next_uid
        self.next_uid += 1
        self.rows[uid] = [0, max_tokens]
        return [uid]

    def remove(self, uid):
        self.removed.append(uid)
        self.rows.pop(uid, None)

    @property
    def has_work(self):
        return bool(self.rows)

    def next(self):
        self.max_batch = max(self.max_batch, len(self.rows))
        out = []
        for uid, state in list(self.rows.items()):
            state[0] += 1
            done = state[0] >= state[1]
            out.append(
                SimpleNamespace(
                    uid=uid,
                    token=100 * uid + state[0],
                    token_logprob=0.0,
                    top_logprobs=None,
                    finish_reason="length" if done else None,
                )
            )
            if done:
                del self.rows[uid]
        return [], out

    def close(self):
        self.closed = True


@pytest.fixture
def runner(monkeypatch):
    import importlib

    ar = importlib.import_module("mlx_vlm.generate.ar")
    FakeGen.instances = []
    monkeypatch.setattr(ar, "BatchGenerator", FakeGen)
    model = SimpleNamespace(language_model=object())
    return vbr.VLMBatchRunner(model, processor=None)


def _collect(runner, n, **kw):
    return list(runner.iter_tokens([1, 2, 3], max_tokens=n, prompt_kwargs={}, **kw))


def test_inline_single_request(runner):
    assert _collect(runner, 3) == [1, 2, 3]
    assert not runner.busy()


def test_concurrent_requests_share_one_generator(runner, monkeypatch):
    ex = ThreadPoolExecutor(max_workers=1)
    runner._executor = ex
    start = threading.Barrier(4)
    results = {}
    gate = threading.Event()
    queued = threading.Event()
    ex.submit(gate.wait)
    submit = runner._submit

    def submit_together(job):
        submit(job)
        with runner._lock:
            if len(runner._pending) == 4:
                queued.set()

    monkeypatch.setattr(runner, "_submit", submit_together)

    def consume(i):
        start.wait()
        results[i] = _collect(runner, 5)

    threads = [threading.Thread(target=consume, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    try:
        assert queued.wait(10), "all four requests must reach admission"
    finally:
        gate.set()
        for t in threads:
            t.join(10)
        ex.shutdown(wait=True)
    assert len(results) == 4
    assert all(len(v) == 5 for v in results.values())
    gens = FakeGen.instances
    # Same greedy settings -> one shared generator that ran rows together.
    assert max(g.max_batch for g in gens) > 1
    assert not runner.busy()


def test_mixed_sampling_shares_one_batch(runner):
    ex = ThreadPoolExecutor(max_workers=1)
    runner._executor = ex
    out = {}
    # Hold the executor until both requests are queued, so they are admitted in the same
    # drive slice whatever the thread timing (no sleeps, no join timeouts).
    gate = threading.Event()
    ex.submit(gate.wait)

    def a():
        out["a"] = _collect(runner, 4)

    def b():
        out["b"] = _collect(runner, 4, temperature=0.7, top_p=0.9)

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start()
    tb.start()
    while True:
        with runner._lock:
            if len(runner._pending) == 2:
                break
        threading.Event().wait(0.001)
    gate.set()
    ta.join()
    tb.join()
    ex.shutdown(wait=True)
    assert len(out["a"]) == 4 and len(out["b"]) == 4
    # One shared generator with the per-row sampler, no per-params groups.
    assert len(FakeGen.instances) == 1
    assert isinstance(FakeGen.instances[0].kwargs["sampler"], vbr.RowSampler)


def test_row_sampler_per_row_params_and_seed():
    import mlx.core as mx

    logprobs = mx.log(mx.softmax(mx.random.normal((3, 50)), axis=-1))
    greedy = mx.argmax(logprobs, axis=-1).tolist()

    def draw(seed):
        s = vbr.RowSampler()
        s.add(11, vbr.RowParams(1.0, 1.0, 0, 0.0, seed))
        vbr._STEP_UIDS = [10, 11, 12]
        try:
            return [s(logprobs).tolist() for _ in range(5)]
        finally:
            vbr._STEP_UIDS = None

    first, again = draw(7), draw(7)
    assert first == again  # seeded row reproducible
    for toks in first:
        assert toks[0] == greedy[0] and toks[2] == greedy[2]  # other rows greedy
    assert len({t[1] for t in first}) > 1  # the sampled row actually samples


def test_abandoned_consumer_frees_its_row(runner):
    it = runner.iter_tokens([1], max_tokens=50, prompt_kwargs={})
    assert next(it) == 1
    it.close()  # consumer stops early (e.g. a stop string)
    runner._drive_slice(resubmit=False)
    assert FakeGen.instances[0].removed == [0]
    assert not runner.busy() or not runner._batches


def test_cancel_event_finishes_with_cancel(runner):
    ev = threading.Event()
    stats = vbr.RunStats()
    it = runner.iter_tokens(
        [1], max_tokens=50, prompt_kwargs={}, cancel_event=ev, stats=stats
    )
    assert next(it) == 1
    ev.set()
    assert list(it) == []
    assert stats.finish_reason == "cancel"


def test_thinking_budget_uses_upstream_criteria_without_draft(runner):
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return {"</think>": [7], "<think>": [6], "\n": [5]}[text]

    runner.processor = SimpleNamespace(tokenizer=Tok())
    runner.drafter = object()
    stats = vbr.RunStats()
    out = list(
        runner.iter_tokens(
            [1], max_tokens=2, prompt_kwargs={}, thinking_budget=4, stats=stats
        )
    )
    gen = FakeGen.instances[0]
    assert out == [1, 2]
    assert gen.kwargs["draft_model"] is None and not stats.used_draft
    (crit,) = gen.budgets
    assert crit is not None and crit.thinking_budget == 4


def test_token_mask_processor_min_tokens_ignore_eos_suppress():
    import mlx.core as mx

    logits = mx.zeros((1, 6))
    p = vbr.TokenMaskProcessor(suppress=[1], eos_ids=[5], min_tokens=2)
    first = p(mx.array([]), logits)
    assert first[0, 1].item() == float("-inf") and first[0, 5].item() == float("-inf")
    p.process_last_token(3, logits)
    second = p.process_last_token(3, logits)
    assert second[0, 5].item() == 0.0  # min_tokens reached: EOS allowed again
    q = vbr.TokenMaskProcessor(eos_ids=[5], ignore_eos=True)
    for _ in range(5):
        out = q.process_last_token(2, logits)
    assert out[0, 5].item() == float("-inf")


def test_ragged_format_set_only_while_runner_steps():
    from yunshu_engine.kernels import ragged_kv

    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    runner.ragged_kv = "int8"
    seen = []

    class Gen:
        _prompt_batch = None
        _generation_batch = None

        def next(self):
            seen.append(ragged_kv._STATE["format"])
            return [], []

    group = vbr._Group(gen=Gen(), spec=False)
    group.jobs = {1: SimpleNamespace(cancel_event=None, abandoned=False)}
    runner._step_group(group)
    assert seen == ["int8"]
    assert ragged_kv._STATE["format"] is None


class FakeDriver:
    """Round driver stand-in: one token per row per step, ``10 + step``."""

    def __init__(self):
        self.rows, self.removed, self.head = [], [], object()

    def add(self, req):
        self.rows.append([req, 0])

    def remove(self, handle):
        self.removed.append(handle)
        self.rows = [r for r in self.rows if r[0].handle is not handle]

    def busy(self):
        return bool(self.rows)

    def step(self):
        from yunshu_engine.round_driver.driver import Event

        out = []
        for row in self.rows:
            row[1] += 1
            done = row[1] >= row[0].max_tokens
            out.append(
                Event(row[0].handle, 10 + row[1], None, "length" if done else None)
            )
        self.rows = [r for r in self.rows if r[1] < r[0].max_tokens]
        return out


def test_text_requests_go_to_the_round_driver():
    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    runner.driver = FakeDriver()
    stats = vbr.RunStats()
    got = list(runner.iter_tokens([1, 2, 3], max_tokens=3, stats=stats))
    assert got == [11, 12, 13]
    assert stats.finish_reason == "length" and stats.generated == 3
    assert stats.used_draft  # greedy, no processors: the driver may draft
    assert not runner.busy()


def test_round_driver_drops_cancelled_rows():
    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    runner.driver = FakeDriver()
    cancel = threading.Event()
    stats = vbr.RunStats()
    it = runner.iter_tokens([1], max_tokens=50, cancel_event=cancel, stats=stats)
    assert next(it) == 11
    cancel.set()
    assert list(it) == []
    assert stats.finish_reason == "cancel"
    assert len(runner.driver.removed) == 1


def test_prefix_sharing_uses_invariant_kernels_in_shared_batch(runner, monkeypatch):
    from yunshu_engine.kernels import batch_invariant

    active = []
    monkeypatch.setattr(batch_invariant, "is_installed", lambda: True)
    monkeypatch.setattr(batch_invariant, "set_active", active.append)
    runner.prefix_invariant = True
    assert _collect(runner, 2) == [1, 2]
    assert True in active
    assert active[-1] is False
