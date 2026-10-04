"""A bare structured output (constraint masks from the first token) must not get a
thinking budget: forcing a thinking close into the mask can leave every logit -inf."""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from yunshu_engine import constrained_spec, vlm_batch_runner
from yunshu_engine.vlm_engine import VLMEngine


def _run(monkeypatch, *, schema, enable_thinking, budget):
    seen = {}

    def iter_tokens(ids, **kw):
        seen.update(kw)
        return iter(())

    eng = MagicMock()
    eng._build_text_constraint.return_value = object() if schema else None
    eng._get_eos_ids.return_value = [1]
    eng._config = {"vocab_size": 10}
    eng._tokenizer = MagicMock()
    eng._tool_guide.return_value = None
    eng._reasoning_markers.return_value = (5, 6, "<think>", "</think>", False)
    eng._batch_runner.iter_tokens = iter_tokens
    monkeypatch.setattr(vlm_batch_runner, "ConstraintProcessor", MagicMock())
    monkeypatch.setattr(constrained_spec, "ConstraintGuide", MagicMock())
    gen = VLMEngine._runner_events_impl(
        eng,
        mx.array([[1, 5]])[0],
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
        json_schema=schema,
        enable_thinking=enable_thinking,
        thinking_budget=budget,
        cancel_event=None,
        stats=MagicMock(),
    )
    list(gen)
    return seen


@pytest.mark.parametrize("enable_thinking", [True, None])
@pytest.mark.parametrize("budget", [0, 64])
def test_bare_schema_gets_no_thinking_budget(monkeypatch, enable_thinking, budget):
    seen = _run(
        monkeypatch,
        schema={"type": "object"},
        enable_thinking=enable_thinking,
        budget=budget,
    )
    assert seen["thinking_budget"] is None


def test_unconstrained_still_gets_budget(monkeypatch):
    seen = _run(monkeypatch, schema=None, enable_thinking=True, budget=64)
    assert seen["thinking_budget"] == 64
