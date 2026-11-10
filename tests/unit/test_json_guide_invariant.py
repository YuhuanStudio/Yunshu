"""Response-format guides (json_schema) keep the batch-invariant arithmetic when greedy without
logprobs / processors, in speculative and spec-off decoding alike (constrained_spec.invariant_guide).

Real-model identity (spec on == spec off, json validity, speed) is a GPU check; these pin routing."""

from types import SimpleNamespace

import pytest


def _make(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


@pytest.fixture
def guides():
    from yunshu_engine import tool_call_grammar as tcg
    from yunshu_engine.constrained_spec import CombinedGuide, ConstraintGuide
    from yunshu_engine.reasoning_gate import ThinkGatedGuide

    text = _make(ConstraintGuide)
    tool = _make(tcg.ToolCallGuide)
    gated = _make(ThinkGatedGuide, inner=text)
    return SimpleNamespace(
        text=text,
        tool=tool,
        gated=gated,
        combined=_make(CombinedGuide, guides=(text, tool)),
        combined_gated=_make(CombinedGuide, guides=(gated, tool)),
    )


def test_response_format_guides_are_invariant(guides):
    from yunshu_engine.constrained_spec import invariant_guide

    for g in (guides.text, guides.gated, guides.combined, guides.combined_gated):
        assert invariant_guide(g)
        assert not invariant_guide(g, logprobs=True)
        assert not invariant_guide(g, processors=[object()])


def test_unknown_guide_member_is_serial(guides):
    from yunshu_engine.constrained_spec import CombinedGuide, invariant_guide

    assert not invariant_guide(_make(CombinedGuide, guides=(guides.text, object())))


def test_spec_off_and_spec_on_choose_the_same_arithmetic(guides):
    # exact_verify and the round driver's guide rows use one predicate (the stock runner's
    # ``_serial_rows`` is gone with the stock runner).
    from yunshu_engine import constrained_spec as cs

    for g in (guides.text, guides.combined):
        assert cs.invariant_request(
            SimpleNamespace(guide=g, logprobs=False, processors=[])
        )
        assert not cs.invariant_request(
            SimpleNamespace(guide=g, logprobs=True, processors=[])
        )
        assert not cs.invariant_request(
            SimpleNamespace(guide=g, logprobs=False, processors=[object()])
        )


def test_exact_verify_keeps_invariant_kernels_for_json_guide(guides, monkeypatch):
    from yunshu_engine import constrained_spec as cs
    from yunshu_engine.kernels import batch_invariant

    monkeypatch.setattr(batch_invariant, "is_installed", lambda: True)
    request = SimpleNamespace(logprobs=False, processors=[], serial_verify_rounds=0)
    with cs.exact_verify(request, guides.text):
        assert cs._SERIAL_VERIFY is False
    assert request.serial_verify_rounds == 0


def test_only_tool_guide_may_use_the_fast_tree(guides):
    from yunshu_engine.constrained_spec import tree_guide

    assert tree_guide(guides.tool)
    assert not tree_guide(guides.text)
    assert not tree_guide(guides.combined)
    assert not tree_guide(guides.gated)
