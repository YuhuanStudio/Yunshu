"""R26: usage detail fields are subsets of the totals, zero output is consistent."""

from __future__ import annotations

import json

import pytest

from yunshu_gateway.streaming import format_openai_usage_chunk


def _usage(**kw):
    raw = format_openai_usage_chunk("id", "m", **kw)
    return json.loads(raw.removeprefix("data: "))["usage"]


def test_totals_and_subsets():
    u = _usage(
        prompt_tokens=10, completion_tokens=7, reasoning_tokens=3, cached_tokens=4
    )
    assert u["total_tokens"] == 17
    assert u["completion_tokens_details"]["reasoning_tokens"] == 3
    assert u["prompt_tokens_details"]["cached_tokens"] == 4


def test_zero_output():
    u = _usage(prompt_tokens=5, completion_tokens=0)
    assert u["completion_tokens"] == 0 and u["total_tokens"] == 5
    assert u["completion_tokens_details"]["reasoning_tokens"] == 0


@pytest.mark.parametrize("reasoning,cached", [(9, 0), (0, 99), (9, 99)])
def test_details_never_exceed_their_totals(reasoning, cached):
    u = _usage(
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=reasoning,
        cached_tokens=cached,
    )
    assert u["completion_tokens_details"]["reasoning_tokens"] <= 5
    assert u["prompt_tokens_details"]["cached_tokens"] <= 10
