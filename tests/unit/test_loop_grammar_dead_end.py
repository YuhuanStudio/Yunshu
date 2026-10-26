"""EngineCore loop: a grammar dead end in one row ends only that request (finish_reason "error"); the
other rows of the same batch keep running (it used to raise out of BatchGenerator.next() and fail
every active request)."""

from __future__ import annotations

import mlx.core as mx

from tests.unit.test_engine_core import _FakeGenResponse, _FakeTokenizer
from yunshu_engine.constraint_eos import ConstrainedDecodingError
from yunshu_engine.request import Request, SamplingParams
from yunshu_engine.scheduler import Scheduler


class _SamplerBatchGen:
    """Calls each row's sampler on every decode step, like BatchGenerator does."""

    def __init__(self):
        self.samplers = {}
        self._uid = 0

    def insert(self, prompts, max_tokens=None, samplers=None, **kw):
        uids = []
        for i, _ in enumerate(prompts):
            self.samplers[self._uid] = samplers[i]
            uids.append(self._uid)
            self._uid += 1
        return uids

    def next(self):
        return [], []

    def next_generated(self):
        out = []
        for uid, sampler in list(self.samplers.items()):
            tok = int(sampler(mx.zeros((1, 8))).item())
            out.append(_FakeGenResponse(uid, tok, "stop" if tok == 3 else None))
            if tok == 3:
                del self.samplers[uid]
        return out


def _dead_end(logits):
    raise ConstrainedDecodingError("constrained decoding dead end")


def _fine(logits):
    return mx.array([1])


def _req(rid, schema):
    return Request(
        request_id=rid,
        prompt="Hello",
        sampling_params=SamplingParams(max_tokens=10, json_schema=schema),
        prompt_token_ids=[0, 1, 2],
        num_prompt_tokens=3,
    )


def test_dead_end_fails_only_its_own_row(monkeypatch):
    sched = Scheduler(None, _FakeTokenizer())
    sched._batch_gen = _SamplerBatchGen()
    made = iter([_dead_end, _fine])
    monkeypatch.setattr(sched, "_make_sampler", lambda sp, ids=None: next(made))
    for rid in ("bad", "good"):
        sched.add_request(_req(rid, {"type": "object"}))
    sched._schedule_waiting()

    outs = sched._process_responses(sched._batch_gen.next_generated())
    by = {o.request_id: o for o in outs}
    assert by["bad"].finished and by["bad"].finish_reason == "error"
    assert "dead end" in by["bad"].error
    assert not by["good"].finished and by["good"].error is None
    assert "bad" not in sched._row_guards


def test_unconstrained_rows_are_not_wrapped(monkeypatch):
    sched = Scheduler(None, _FakeTokenizer())
    sched._batch_gen = _SamplerBatchGen()
    sched.add_request(_req("plain", None))
    sched._schedule_waiting()
    assert not sched._row_guards
