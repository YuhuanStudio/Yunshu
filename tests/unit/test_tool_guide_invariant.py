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
    assert not invariant_guide(combined)


def test_serial_rows(tool_guide):
    from yunshu_engine.vlm_batch_runner import _serial_rows

    assert not _serial_rows(SimpleNamespace(guide=None, logprobs=False))
    assert _serial_rows(SimpleNamespace(guide=None, logprobs=True))
    assert not _serial_rows(SimpleNamespace(guide=tool_guide, logprobs=False))
    assert _serial_rows(SimpleNamespace(guide=tool_guide, logprobs=True))
    assert _serial_rows(SimpleNamespace(guide=object(), logprobs=False))


def test_exact_verify_keeps_invariant_kernels_for_a_tool_guide(tool_guide, monkeypatch):
    from yunshu_engine import constrained_spec as cs
    from yunshu_engine.kernels import batch_invariant

    monkeypatch.setattr(batch_invariant, "is_installed", lambda: True)
    request = SimpleNamespace(logprobs=False, processors=[], serial_verify_rounds=0)
    with cs.exact_verify(request, tool_guide):
        assert cs._SERIAL_VERIFY is False
    assert request.serial_verify_rounds == 0


def test_takes_guided_needs_the_copy_wrapper(monkeypatch):
    from mlx_vlm.speculative import utils

    from yunshu_engine import dflash_copy

    monkeypatch.setattr(utils, "_dflash_rounds", lambda *a, **k: iter(()))
    assert dflash_copy.takes_guided(None, None, [], {"first_bonus": 1}) is False
