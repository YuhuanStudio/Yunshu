"""A lone decoding row moves from the shared batch onto the speculative lane."""

from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_batch_runner import VLMBatchRunner, _Group


class _Drafter:  # type name carries "dflash" for _spec_mode
    pass


_Drafter.__name__ = "DFlashDraftModel"


class _Batch:
    def __init__(self, uid=7, n=40, max_tokens=300):
        self.uids = [uid]
        self.max_tokens = [max_tokens]
        self._num_tokens = [n]
        self._next_tokens = mx.array([11], dtype=mx.int32)
        self.prompt_cache = ["kv0", "gdn0"]
        self.sampler = None
        self.stop_criteria = lambda t: False
        self.logits_processors = []
        self.is_speculative = False
        self.filtered = None

    def _eval_pending_state(self):
        pass

    def filter(self, keep):
        self.filtered = list(keep)
        self.prompt_cache.clear()


def _runner(job_overrides=None, pending=(), inflight=1, second_job=False):
    batch = _Batch()
    gen = SimpleNamespace(
        _generation_batch=batch, _prompt_batch=None, _unprocessed_sequences=[]
    )
    job = SimpleNamespace(
        uid=7,
        promotable=True,
        priority=0,
        ids=[1, 2, 3],
        stats=SimpleNamespace(
            t_first=1.0, used_apc=True, used_draft=False, spec_reason="x", spec_mode=""
        ),
        spec_base=None,
        spec_last=None,
        spec_request=None,
        use_draft=False,
        greedy=True,
        keyed=None,
        logprobs=False,
        top_logprobs=0,
        guide=None,
        spec_procs=[],
    )
    for k, v in (job_overrides or {}).items():
        setattr(job, k, v)
    jobs = {7: job}
    if second_job:
        jobs[8] = SimpleNamespace()
    group = _Group(gen=gen, spec=False, jobs=jobs, prefills={})
    built = {}
    runner = SimpleNamespace(
        drafter=_Drafter(),
        draft_kind="dflash",
        draft_block_size=None,
        model=object(),
        _spec=None,
        _aux_spec=None,
        _driver_jobs={},
        _pending=list(pending),
        inflight=lambda: inflight,
        _batches={("k",): group},
        _vocab_owner=None,
        promotions=0,
        built=built,
    )

    def new_generator(**kw):
        built.update(kw)
        return SimpleNamespace(_generation_batch=None)

    runner._new_generator = new_generator
    return runner, group, batch, job


def test_lone_promotable_row_moves_onto_the_lane_with_its_cache_and_budget():
    runner, group, batch, job = _runner()
    assert VLMBatchRunner._maybe_promote(runner) is True
    spec = runner._spec
    assert spec.spec and list(spec.jobs) == [7] and not group.jobs
    gb = spec.gen._generation_batch
    assert gb.is_speculative and gb.uids == [7]
    assert gb.prompt_cache == ["kv0", "gdn0"]  # copied before the old batch is cleared
    assert gb.max_tokens == [260] and gb.hidden is None  # tokens left, no stale hidden
    assert int(gb.first_tokens[0]) == 11
    assert batch.filtered == []
    assert job.stats.spec_reason == "promoted" and runner.promotions == 1


def test_no_promotion_while_anything_else_is_in_flight_or_the_row_is_not_eligible():
    for kwargs in (
        {"inflight": 2},
        {"pending": [object()]},
        {"second_job": True},
        {"job_overrides": {"promotable": False}},
        {"job_overrides": {"stats": SimpleNamespace(t_first=None, used_apc=True)}},
    ):
        runner, *_ = _runner(**kwargs)
        assert VLMBatchRunner._maybe_promote(runner) is False, kwargs
        assert runner._spec is None


def test_sampled_promotion_moves_the_same_sampler_with_the_committed_position_base():
    from yunshu_engine.keyed_sampling import KeyedSampler
    from yunshu_engine.vlm_batch_runner import RowParams

    sampler = KeyedSampler(RowParams(0.7, 0.95, 20, 0.0, 17), 17)
    runner, group, batch, job = _runner(
        job_overrides={"greedy": False, "keyed": sampler}
    )
    assert VLMBatchRunner._maybe_promote(runner)
    promoted = runner._spec.gen._generation_batch
    assert promoted.sampler is sampler
    assert not promoted.greedy_sampling
    assert runner.built["sampler"] is sampler and not runner.built["greedy"]
    assert sampler.position_base == 40
    assert job.keyed is sampler and not group.jobs
    assert promoted.max_tokens == [260]


def test_promotion_transfers_the_pending_tokens_logprob_and_top_logprobs_once():
    runner, _, batch, job = _runner(job_overrides={"logprobs": True, "top_logprobs": 2})
    batch._next_lps = mx.array([-1.25])
    batch._next_top_idx = mx.array([[11, 12]])
    batch._next_top_lp = mx.array([[-1.25, -2.5]])
    assert VLMBatchRunner._maybe_promote(runner)
    assert job.spec_request.logprobs and job.spec_request.top_logprobs == 2
    assert job.spec_request.take(11) == (-1.25, [(11, -1.25), (12, -2.5)])
    assert not job.spec_request.pending
    # The following verify report must not be shifted by the pending first token.
    job.spec_request.record(mx.array([[-2.0, -1.0]]), [1])
    assert job.spec_request.take(1)[0] == -1.0
