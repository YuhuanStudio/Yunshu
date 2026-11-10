"""Tool-call guides alone keep the batch-invariant path (constrained_spec.invariant_guide).

The real-model proof is scripts/research/probe_tools_identity.py (DFlash, MTP and drafting off
give identical content and tool calls with the grammar on); these pin the routing decisions.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def tool_guide(monkeypatch):
    from yunshu_engine import tool_call_grammar as tcg

    guide = object.__new__(tcg.ToolCallGuide)
    return guide


def test_invariant_guide_only_for_a_tool_guide_alone(tool_guide):
    from yunshu_engine.constrained_spec import CombinedGuide, invariant_guide

    assert invariant_guide(tool_guide)
    assert not invariant_guide(None)
    assert not invariant_guide(tool_guide, logprobs=True)
    assert not invariant_guide(tool_guide, processors=[object()])
    combined = object.__new__(CombinedGuide)
    combined.guides = (tool_guide, object())
    assert not invariant_guide(combined)


def test_exact_verify_keeps_invariant_kernels_for_a_tool_guide(tool_guide, monkeypatch):
    from yunshu_engine import constrained_spec as cs
    from yunshu_engine.kernels import batch_invariant

    monkeypatch.setattr(batch_invariant, "is_installed", lambda: True)
    request = SimpleNamespace(logprobs=False, processors=[], serial_verify_rounds=0)
    with cs.exact_verify(request, tool_guide):
        assert cs._SERIAL_VERIFY is False
    assert request.serial_verify_rounds == 0


def test_report_only_verify_keeps_invariant_arithmetic(monkeypatch):
    from yunshu_engine import constrained_spec as cs
    from yunshu_engine.kernels import batch_invariant

    monkeypatch.setattr(batch_invariant, "is_installed", lambda: True)
    monkeypatch.setitem(batch_invariant._STATE, "active", True)
    request = cs.SpecRequest(True, 3)
    with cs.exact_verify(request):
        assert batch_invariant._STATE["active"]
        assert not cs._SERIAL_VERIFY
    assert request.serial_verify_rounds == 0
