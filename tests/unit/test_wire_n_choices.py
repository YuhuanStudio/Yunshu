"""R23: n > 1 on chat and completions: per-choice seeds stay inside the signed 64-bit range
and differ, streaming n > 1 is refused up front, and every choice shares the logical request's
cancel event (usage billing for n > 1 is in test_wire_usage_invariants)."""

from __future__ import annotations

import openai
import pytest

from .wire_clients import Clients
from .wire_harness import Script, install

USER = [{"role": "user", "content": "hi"}]
MAX = 2**63 - 1


def _serve(monkeypatch, **script):
    c, eng = install(monkeypatch, Script(**script))
    return Clients(c), eng


@pytest.mark.parametrize("seed", [MAX, MAX - 1, -(2**63), 0, 7])
def test_per_choice_seeds_are_valid_and_distinct(monkeypatch, seed):
    cl, eng = _serve(monkeypatch, pieces=["a", "b"])
    cl.oa.chat.completions.create(model="m", messages=USER, n=3, seed=seed)
    cl.oa.completions.create(model="m", prompt="hi", n=3, seed=seed)
    seeds = [c["seed"] for c in eng.calls]
    assert len(seeds) == 6
    for group in (seeds[:3], seeds[3:]):
        assert len(set(group)) == 3, group
        assert all(-(2**63) <= s < 2**63 for s in group), group
        assert group[0] == seed  # choice 0 is the stream n=1 would give


def test_streamed_n_choices_are_rejected_not_half_served(monkeypatch):
    cl, eng = _serve(monkeypatch, pieces=["a"])
    for create in (
        lambda: cl.oa.chat.completions.create(
            model="m", messages=USER, n=2, stream=True
        ),
        lambda: cl.oa.completions.create(model="m", prompt="hi", n=2, stream=True),
    ):
        with pytest.raises(openai.BadRequestError):
            create()
    assert not eng.calls


def test_all_choices_share_the_request_cancel_event(monkeypatch):
    cl, eng = _serve(monkeypatch, pieces=["a", "b"])
    cl.oa.chat.completions.create(model="m", messages=USER, n=3)
    cl.oa.completions.create(model="m", prompt="hi", n=3)
    for group in (eng.calls[:3], eng.calls[3:]):
        events = {id(c.get("cancel_event")) for c in group}
        assert len(events) == 1 and group[0].get("cancel_event") is not None
