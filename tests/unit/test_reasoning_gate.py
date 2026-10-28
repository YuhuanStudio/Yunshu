"""A structured request binds its grammar AFTER the reasoning ends (vllm 4ac1f8d2a, ollama
1ce2b680, mlx2 01dcd75): free reasoning, no stop or implicit end inside it, then the grammar."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.reasoning_gate import (
    NOT_CLOSED,
    ReasoningGate,
    ThinkGatedGuide,
    ThinkGatedSampler,
)

from .tiny_text_engine import FakeTokenizer, make_engine

OPEN, CLOSE, EOS = 60, 61, 1
PROMPT = "hello:"  # ends with the think-start token
CHOICE = {"type": "choice", "choices": ["ab"]}
ANSWER = [26, 27]  # decode to "a", "b" in FakeTokenizer


class SeqConstraint:
    """Grammar stand-in: exactly ANSWER, then EOS (the tiny tokenizer has no vocabulary)."""

    def __init__(self):
        self.i = 0

    def get_allowed_tokens(self, tokenizer, generated):
        return [ANSWER[self.i]] if self.i < len(ANSWER) else [EOS]

    def advance(self, text):
        self.i += 1

    def checkpoint(self):
        return self.i

    def rollback(self, saved=None):
        self.i = saved or 0


@pytest.fixture(autouse=True)
def _seq_constraint(monkeypatch):
    from yunshu_engine.grammar_constraint import ConstraintFactory

    monkeypatch.setattr(
        ConstraintFactory, "create", staticmethod(lambda *a, **k: SeqConstraint())
    )


class ThinkTok(FakeTokenizer):
    think_start_id = OPEN

    def encode(self, text, add_special_tokens=True, **kw):
        if text == "<think>":
            return [OPEN]
        if text == "</think>":
            return [CLOSE]
        return super().encode(text, add_special_tokens=add_special_tokens)


class Steer:
    """Logits processor: wants EOS for the first ``eos_steps`` tokens, then </think>."""

    def __init__(self, eos_steps, close_at):
        self.eos_steps, self.close_at, self.n = eos_steps, close_at, 0

    def __call__(self, tokens, logits):
        n, self.n = self.n, self.n + 1
        want = EOS if n < self.eos_steps else (CLOSE if n == self.close_at else None)
        if want is None:
            return logits
        return logits + mx.zeros_like(logits).at[..., want].add(1e4)


def _engine():
    eng = make_engine()
    eng._tokenizer = ThinkTok()
    return eng


async def test_text_path_reasoning_is_free_then_grammar_binds():
    out = await _engine()._generate_fast(
        prompt=PROMPT,
        max_tokens=12,
        temperature=0.0,
        json_schema=CHOICE,
        logits_processors=[Steer(eos_steps=3, close_at=5)],
    )
    # EOS wanted for 3 steps is masked (never ends inside the reasoning); the close lands at 5
    assert out.error is None
    # 5 free tokens + the close + the 2-token answer + EOS; the fake tokenizer decodes the
    # close as "j", so the text split is checked on the stream below
    assert out.finish_reason == "stop" and out.completion_tokens == 8


async def test_text_path_unclosed_reasoning_reports_not_enforced():
    from yunshu_engine.request_tracker import current_request_info

    class Info:
        structured_output = None

    info = Info()
    tok = current_request_info.set(info)
    try:
        out = await _engine()._generate_fast(
            prompt=PROMPT, max_tokens=4, temperature=0.0, json_schema=CHOICE
        )
    finally:
        current_request_info.reset(tok)
    assert out.finish_reason == "length" and out.text == ""
    assert info.structured_output["enforced"] is False
    assert info.structured_output["reason"] == NOT_CLOSED


async def test_text_path_without_open_think_is_constrained_from_token_one():
    out = await _engine()._generate_fast(
        prompt="hello", max_tokens=6, temperature=0.0, json_schema=CHOICE
    )
    assert out.text == "ab"


async def test_text_stream_splits_reasoning_from_the_grammar_answer():
    kw = dict(
        prompt=PROMPT,
        max_tokens=12,
        temperature=0.0,
        json_schema=CHOICE,
    )
    text, reasoning = "", ""
    async for o in _engine()._stream_generate_fast(
        logits_processors=[Steer(3, 5)], **kw
    ):
        if o.current_state == "reasoning":
            reasoning += o.new_text or ""
        else:
            text += o.new_text or ""
    assert text == "ab" and len(reasoning) == 6


def test_gate_report_flips_when_the_close_is_generated():
    rep = {}
    gate = ReasoningGate(CLOSE, [EOS], [7], rep)
    assert rep == {"enforced": False, "reason": NOT_CLOSED}
    gate.feed(5)
    assert gate.open
    gate.feed(CLOSE)
    assert not gate.open and rep["enforced"] is True


class _Inner:
    words = 2
    constrained = True

    def __init__(self):
        self.fed = []

    def mask(self):
        return np.array([1, 0], dtype=np.int32)

    def feed(self, t):
        self.fed.append(t)
        return True

    def checkpoint(self):
        return len(self.fed)

    def restore(self, n):
        del self.fed[n:]


def _bit(row, t):
    return bool((int(row[t >> 5]) >> (t & 31)) & 1)


def test_guide_masks_stop_and_implicit_ends_while_reasoning_then_defers_to_inner():
    gate = ReasoningGate(CLOSE, [EOS], [7])
    g = ThinkGatedGuide(_Inner(), gate)
    row = g.mask()
    assert not _bit(row, EOS) and not _bit(row, 7)
    assert _bit(row, CLOSE) and _bit(row, 5)
    g.feed(5)
    assert g.inner.fed == []
    g.feed(CLOSE)
    assert g.inner.fed == []  # the close itself is not a grammar token
    assert list(g.mask()) == [1, 0]
    g.feed(9)
    assert g.inner.fed == [9]


def test_guide_plan_walks_the_close_and_restores():
    gate = ReasoningGate(CLOSE, [EOS], [])
    g = ThinkGatedGuide(_Inner(), gate)
    out = g.plan([5, CLOSE], 3)
    assert not _bit(out[0], EOS) and not _bit(out[1], EOS)  # still reasoning
    assert list(out[2]) == [1, 0]  # after the drafted close: the grammar
    assert gate.open  # plan rewinds


def test_sampler_blocks_stop_and_switches_after_the_close():
    calls = []

    def free(logits):
        calls.append("free")
        return mx.argmax(logits, axis=-1)

    def constrained(logits):
        calls.append("grammar")
        return mx.argmax(logits, axis=-1)

    gate = ReasoningGate(CLOSE, [EOS], [])
    s = ThinkGatedSampler(free, constrained, gate)
    logits = mx.zeros((1, 64)).at[0, EOS].add(5.0)
    assert int(s(logits)[0]) != EOS  # the stop is masked while reasoning
    assert int(s(mx.zeros((1, 64)).at[0, CLOSE].add(5.0))[0]) == CLOSE
    assert not gate.open
    s(logits)
    assert calls == ["free", "free", "grammar"]


def _vlm_guide(monkeypatch, prompt):
    """The guide the VLM batch runner receives for a schema request."""
    from unittest.mock import MagicMock

    from yunshu_engine import constrained_spec, vlm_batch_runner
    from yunshu_engine.vlm_engine import VLMEngine

    seen = {}

    def iter_tokens(ids, **kw):
        seen.update(kw)
        return iter(())

    eng = MagicMock()
    eng._build_text_constraint.return_value = object()
    eng._get_eos_ids.return_value = [EOS]
    eng._config = {"vocab_size": 64}
    eng._tokenizer = MagicMock()
    eng._tool_guide.return_value = None
    eng._tool_call_marker_id.return_value = 7
    eng._reasoning_markers.return_value = (OPEN, CLOSE, "<think>", "</think>", False)
    eng._batch_runner.iter_tokens = iter_tokens
    monkeypatch.setattr(vlm_batch_runner, "ConstraintProcessor", MagicMock())
    monkeypatch.setattr(constrained_spec, "ConstraintGuide", lambda *a, **k: _Inner())
    list(
        VLMEngine._runner_events_impl(
            eng,
            mx.array(prompt),
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=None,
            stop=None,
            stop_token_ids=None,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
            json_schema={"type": "object"},
            enable_thinking=True,
            thinking_budget=None,
            cancel_event=None,
            stats=MagicMock(),
        )
    )
    return seen["guide"]


def test_vlm_runner_gets_a_gated_guide_when_the_prompt_opens_think(monkeypatch):
    guide = _vlm_guide(monkeypatch, [5, 6, OPEN])
    assert isinstance(guide, ThinkGatedGuide)
    row = guide.mask()
    assert not _bit(row, EOS) and not _bit(row, 7)


def test_vlm_runner_guide_is_ungated_when_thinking_is_closed(monkeypatch):
    guide = _vlm_guide(monkeypatch, [5, 6, OPEN, CLOSE])
    assert isinstance(guide, _Inner)


def test_engine_loop_sampler_is_gated_when_the_prompt_opens_think():
    from types import SimpleNamespace

    from yunshu_engine.request import SamplingParams
    from yunshu_engine.scheduler import Scheduler

    me = SimpleNamespace(tokenizer=ThinkTok())
    sp = SamplingParams(temperature=0.0, grammar={"type": "choice", "choices": ["ab"]})
    gated = Scheduler._make_sampler(me, sp, [5, 6, OPEN])
    assert isinstance(gated, ThinkGatedSampler) and gated.gate.open
    ungated = Scheduler._make_sampler(me, sp, [5, 6, OPEN, CLOSE])
    assert not isinstance(ungated, ThinkGatedSampler)
    logits = mx.zeros((1, 64)).at[0, EOS].add(9.0)
    assert int(gated(logits)[0]) != EOS  # a stop inside the reasoning is masked
