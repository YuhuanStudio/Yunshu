"""Logprob requests stay on the MTP chain rounds: the tree rounds record no probability reports."""

from yunshu_engine import constrained_spec as cs
from yunshu_engine import mtp_lane


def test_logprobs_need_the_chain(monkeypatch):
    request = cs.SpecRequest(logprobs=True)
    monkeypatch.setattr(cs, "_REQUEST", request)
    assert mtp_lane._request_needs_chain()
    request.logprobs = False
    assert not mtp_lane._request_needs_chain()
    request.processors.append(object())
    assert mtp_lane._request_needs_chain()
    monkeypatch.setattr(cs, "_REQUEST", None)
    assert not mtp_lane._request_needs_chain()
