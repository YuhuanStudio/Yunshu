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

    def insert(self, prompts, max_tokens, prompt_kwargs, logits_processors):
        assert self.kwargs.get("prefill_batch_size") == 1
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


def test_concurrent_requests_share_one_generator(runner):
    ex = ThreadPoolExecutor(max_workers=1)
    runner._executor = ex
    start = threading.Barrier(4)
    results = {}

    def consume(i):
        start.wait()
        results[i] = _collect(runner, 5)

    threads = [threading.Thread(target=consume, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    ex.shutdown(wait=True)
    assert all(len(v) == 5 for v in results.values())
    gens = FakeGen.instances
    # Same greedy settings -> one shared generator that ran rows together.
    assert max(g.max_batch for g in gens) > 1
    assert not runner.busy()


def test_mixed_sampling_shares_one_batch(runner):
    ex = ThreadPoolExecutor(max_workers=1)
    runner._executor = ex
    out = {}
    barrier = threading.Barrier(2)

    def a():
        barrier.wait()
        out["a"] = _collect(runner, 4)

    def b():
        barrier.wait()
        out["b"] = _collect(runner, 4, temperature=0.7, top_p=0.9)

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start()
    tb.start()
    ta.join(10)
    tb.join(10)
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
