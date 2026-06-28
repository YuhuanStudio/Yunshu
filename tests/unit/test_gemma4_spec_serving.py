"""Tests for the #175 Gemma-4 assistant-drafter n=1 serving route gating.

`_gemma4_spec_eligible` decides whether a single-request call can be served by
the validated spec-decode primitive. It must route ONLY when the output would be
identical to normal generation (greedy / pure-temperature), and never when
features the primitive can't honor are requested (top_p/top_k/min_p/penalties/
grammar/logprobs/logit_bias/xtc). It must also be inactive when the drafter
wasn't loaded.
"""

from types import SimpleNamespace

from python.yunshu_engine.batched_engine import BatchedEngine

_DEFAULTS = dict(
    logprobs=False, json_schema=None, logits_processors=None, logit_bias=None,
    top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
    frequency_penalty=0.0, presence_penalty=0.0, xtc_probability=0.0,
)


def _eligible(proposer, **overrides):
    kw = {**_DEFAULTS, **overrides}
    fake = SimpleNamespace(_gemma4_assistant_proposer=proposer)
    return BatchedEngine._gemma4_spec_eligible(fake, **kw)


def test_inactive_without_drafter():
    # No proposer loaded -> never eligible, even for a clean greedy request.
    assert _eligible(None) is False


def test_eligible_clean_greedy():
    assert _eligible(object()) is True


def test_eligible_pure_temperature():
    # temp>0 with default top_p/top_k/min_p is still distribution-correct.
    assert _eligible(object(), top_p=1.0, top_k=0, min_p=0.0) is True


def test_ineligible_features():
    p = object()
    assert _eligible(p, logprobs=True) is False
    assert _eligible(p, json_schema={"type": "object"}) is False
    assert _eligible(p, logits_processors=[lambda t, l: l]) is False
    assert _eligible(p, logit_bias={1: 2.0}) is False
    assert _eligible(p, top_p=0.5) is False
    assert _eligible(p, top_k=20) is False
    assert _eligible(p, min_p=0.1) is False
    assert _eligible(p, repetition_penalty=1.1) is False
    assert _eligible(p, frequency_penalty=0.5) is False
    assert _eligible(p, presence_penalty=0.5) is False
    assert _eligible(p, xtc_probability=0.3) is False


def test_top_p_none_treated_as_unconstrained():
    assert _eligible(object(), top_p=None) is True


def test_ineligible_with_lora_adapter():
    """(self-audit): a LoRA request must NOT route to the assistant-spec
    primitive — that path runs the BASE-weight drafter and applies no adapter, so
    it would silently generate from the base model (violating the "identical to
    normal generation" contract). LoRA requests fall through to _generate_fast,
    which applies the adapter atomically on the executor (the keystone)."""
    p = object()
    assert _eligible(p, lora_adapter="my-adapter") is False
    # No adapter (None / empty) stays eligible.
    assert _eligible(p, lora_adapter=None) is True
    assert _eligible(p, lora_adapter="") is True
