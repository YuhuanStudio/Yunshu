"""Response-format guides draft on the round driver (masked tree), not as plain rows."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")


def _make(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class Automaton:
    """Allows token (state % 3) + 1 at each step; a tiny stand-in for a grammar."""

    words = 1
    constrained = True

    def __init__(self):
        self.state = 0
        self.fed = []

    def mask(self):
        out = np.zeros(1, dtype=np.int32)
        out[0] = 1 << ((self.state % 3) + 1)
        return out

    def checkpoint(self):
        return self.state

    def restore(self, saved):
        self.state = saved

    def feed(self, token):
        self.fed.append(token)
        self.state += 1


def test_plan_tree_masks_follow_each_nodes_path():
    from yunshu_engine.constrained_spec import plan_tree

    g = Automaton()
    # root(pending) -> a(1) -> b(2) ; root -> c(3, forbidden at the root) -> d(1)
    toks = [7, 1, 2, 3, 1]
    parents = [-1, 0, 1, 0, 3]
    masks = plan_tree(g, toks, parents)
    allowed = lambda row, t: bool((masks[row, 0] >> t) & 1)  # noqa: E731
    assert allowed(0, 1) and not allowed(0, 2)  # root: token 1
    assert allowed(1, 2) and not allowed(1, 1)  # after "1": token 2
    assert allowed(2, 3)  # after "1,2": token 3
    assert (
        masks[4, 0] == -1
    )  # below the forbidden node 3: never reached, left all-allowed
    assert g.state == 0  # the live guide is untouched


def test_response_format_rows_draft_and_tool_rows_stay_plain():
    from yunshu_engine import tool_call_grammar as tcg
    from yunshu_engine.constrained_spec import (
        CombinedGuide,
        ConstraintGuide,
        masks_every_row,
    )
    from yunshu_engine.reasoning_gate import ThinkGatedGuide
    from yunshu_engine.round_driver.dflash import _plain_guided

    text = _make(ConstraintGuide)
    in_call = type("InCall", (tcg.ToolCallGuide,), {"constrained": True})
    tool = _make(in_call)
    gated = _make(ThinkGatedGuide, inner=text)
    combined = _make(CombinedGuide, guides=(text, tool))
    for g in (text, gated, combined):
        assert masks_every_row(g)
        assert not _plain_guided(SimpleNamespace(req=SimpleNamespace(guide=g)))
    assert not masks_every_row(tool)
    assert _plain_guided(SimpleNamespace(req=SimpleNamespace(guide=tool)))
    assert not _plain_guided(SimpleNamespace(req=SimpleNamespace(guide=None)))


def test_guides_answer_arms_for_the_driver_emit_loop():
    from yunshu_engine.constrained_spec import CombinedGuide, ConstraintGuide
    from yunshu_engine.reasoning_gate import ThinkGatedGuide

    text = _make(ConstraintGuide)
    assert text.arms(5) is False
    assert _make(ThinkGatedGuide, inner=text).arms(5) is False
    assert _make(CombinedGuide, guides=(text, text)).arms(5) is False


def test_driver_guide_routes_response_format_rows_to_the_driver():
    from yunshu_engine.constrained_spec import ConstraintGuide
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    runner = object.__new__(VLMBatchRunner)
    runner.driver = SimpleNamespace(dflash=object(), can_draft=True)
    guide = _make(ConstraintGuide)

    def job(**kw):
        base = dict(guide=guide, logprobs=False, processors=[])
        base.update(kw)
        return SimpleNamespace(**base)

    assert runner._driver_guide(job())
    assert not runner._driver_guide(job(logprobs=True))
    assert not runner._driver_guide(job(processors=[object()]))
    assert not runner._driver_guide(job(guide=None))
    runner.driver = SimpleNamespace(dflash=None, can_draft=False)
    assert not runner._driver_guide(job())
