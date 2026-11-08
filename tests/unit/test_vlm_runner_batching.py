"""VLMBatchRunner scheduling with a fake round driver: every request is a driver row.

The stock BatchGenerator runner (shared generators, speculative lane groups, promotion) is gone: the
runner admits requests, applies memory / prefix admission and hands rows to the one round driver.
``FakeDriver`` emits one token per row per step: ``100 * <admission index> + step``."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from yunshu_engine import vlm_batch_runner as vbr


class FakeDriver:
    """Round driver stand-in: one token per row per step, ``100 * n + step`` for the n-th admitted row."""

    def __init__(self):
        self.rows, self.removed, self.requests = [], [], []
        self.head = object()
        self.can_draft = True
        self.max_rows = 0
        self.apc = None  # RoundDriver always has it (None without a prefix cache)

    def add(self, req):
        n = len(self.requests)
        self.requests.append(req)
        self.rows.append([req, 0, n])
        self.max_rows = max(self.max_rows, len(self.rows))

    def remove(self, handle):
        self.removed.append(handle)
        self.rows = [r for r in self.rows if r[0].handle is not handle]

    def busy(self):
        return bool(self.rows)

    def flush_checkpoints(self):
        """The runner publishes captured APC checkpoints after emitting a step's tokens."""

    def planned_checkpoints(self, handle):
        return None

    def step(self):
        from yunshu_engine.round_driver.driver import Event

        out = []
        for row in self.rows:
            row[1] += 1
            done = row[1] >= row[0].max_tokens
            out.append(
                Event(
                    row[0].handle,
                    100 * row[2] + row[1],
                    None,
                    "length" if done else None,
                )
            )
        self.rows = [r for r in self.rows if r[1] < r[0].max_tokens]
        return out


@pytest.fixture
def runner():
    r = vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), processor=None)
    r.driver = FakeDriver()
    return r


def _collect(runner, n, **kw):
    return list(runner.iter_tokens([1, 2, 3], max_tokens=n, prompt_kwargs={}, **kw))


def test_inline_single_request(runner):
    assert _collect(runner, 3) == [1, 2, 3]
    assert not runner.busy()


def test_concurrent_requests_are_rows_of_the_one_driver(runner, monkeypatch):
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
    assert runner.driver.max_rows > 1  # the rows ran together in the one driver
    assert not runner.busy()


def test_sampling_parameters_ride_on_the_request_not_on_a_group(runner):
    ex = ThreadPoolExecutor(max_workers=1)
    runner._executor = ex
    out = {}
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
    sampling = sorted(
        (r.sampling is not None, getattr(r.sampling, "temperature", None))
        for r in runner.driver.requests
    )
    assert sampling == [(False, None), (True, 0.7)]


def test_abandoned_consumer_frees_its_row(runner):
    it = runner.iter_tokens([1], max_tokens=50, prompt_kwargs={})
    assert next(it) == 1
    it.close()  # consumer stops early (e.g. a stop string)
    runner._drive_slice(resubmit=False)
    assert len(runner.driver.removed) == 1
    assert not runner.driver.rows


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
    assert len(runner.driver.removed) == 1


def test_thinking_budget_reaches_the_driver_row_as_the_forcing_criteria(runner):
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return {"</think>": [7], "<think>": [6], "\n": [5]}[text]

    runner.processor = SimpleNamespace(tokenizer=Tok())
    out = list(
        runner.iter_tokens(
            [1], max_tokens=2, prompt_kwargs={}, thinking_budget=4, stats=vbr.RunStats()
        )
    )
    assert out == [1, 2]
    (req,) = runner.driver.requests
    assert req.budget is not None and req.budget.thinking_budget == 4


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


def test_text_requests_go_to_the_round_driver():
    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    runner.driver = FakeDriver()
    stats = vbr.RunStats()
    got = list(runner.iter_tokens([1, 2, 3], max_tokens=3, stats=stats))
    assert got == [1, 2, 3]
    assert stats.finish_reason == "length" and stats.generated == 3
    assert stats.used_draft  # greedy, no processors: the driver may draft
    assert not runner.busy()


def test_a_runner_without_a_driver_refuses_instead_of_serving_on_another_path():
    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    with pytest.raises(RuntimeError, match="no round driver"):
        list(runner.iter_tokens([1, 2, 3], max_tokens=3, stats=vbr.RunStats()))


def test_round_driver_drops_cancelled_rows():
    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    runner.driver = FakeDriver()
    cancel = threading.Event()
    stats = vbr.RunStats()
    it = runner.iter_tokens([1], max_tokens=50, cancel_event=cancel, stats=stats)
    assert next(it) == 1
    cancel.set()
    assert list(it) == []
    assert stats.finish_reason == "cancel"
    assert len(runner.driver.removed) == 1


def test_driver_rows_set_and_extend_the_draft_vocab_context():
    """Regression: the driver never told the reduced draft readout about the prompt's ids, so code
    prompts drafted from the base set only: 8K code decode 3.8 instead of 5 tokens/round."""
    calls = []

    class Vocab:
        def set_context(self, ids):
            calls.append(("set", list(ids)))

        def add_context(self, ids):
            calls.append(("add", list(ids)))

    runner = vbr.VLMBatchRunner(
        SimpleNamespace(language_model=object()), processor=None
    )
    runner.driver = FakeDriver()
    runner.drafter = SimpleNamespace(_draft_vocab=Vocab())
    list(runner.iter_tokens([1, 2, 3], max_tokens=1, stats=vbr.RunStats()))
    assert calls[0] == ("set", [1, 2, 3])


def test_a_lone_finished_row_teaches_the_storage_tiers_the_prefill_speed(runner):
    """The stock runner fed the APC cost model at every finish; the driver path must too."""
    observed = []
    runner.apc_manager = SimpleNamespace(
        disk=SimpleNamespace(observe_prefill=lambda *a: observed.append(a))
    )
    assert _collect(runner, 2) == [1, 2]
    assert len(observed) == 1 and observed[0][0] == 3  # the prompt's uncached tokens


def test_a_row_records_where_its_prefix_hit_came_from(runner):
    rec = SimpleNamespace(cached=True, tier="ssd", ms=12.5, device="Micron", t=3.0)
    runner.apc_manager = SimpleNamespace(take_lookup=lambda *a, **k: rec)
    stats = vbr.RunStats()
    list(runner.iter_tokens([1, 2, 3], max_tokens=2, prompt_kwargs={}, stats=stats))
    assert (stats.cache_tier, stats.cache_reload_ms) == ("ssd", 12.5)
    assert stats.cache_reason == "hit" and stats.cache_device == "Micron"


def test_a_cleanup_step_that_raises_still_publishes_the_terminal(runner):
    runner.apc_manager = SimpleNamespace(
        disk=SimpleNamespace(
            observe_prefill=lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    )
    assert _collect(runner, 2) == [1, 2]
    assert runner.finish_cleanup_failures == {"observe_prefill": 1}
    assert not runner.busy()
