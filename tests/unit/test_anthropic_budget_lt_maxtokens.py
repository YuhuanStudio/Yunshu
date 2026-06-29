"""(deferred-LOW from the Anthropic-protocol hunt): Anthropic mandates
max_tokens > thinking.budget_tokens (the thinking budget must leave room for the visible
answer) and returns 400 otherwise. Yunshu accepted budget_tokens >= max_tokens leniently;
now it validates per the contract.
"""

from __future__ import annotations

import pytest

from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest


def _req(max_tokens, budget):
    return AnthropicMessagesRequest(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
        thinking={"type": "enabled", "budget_tokens": budget},
    )


def test_budget_ge_max_tokens_rejected():
    with pytest.raises(ValueError):
        _req(max_tokens=100, budget=100)  # equal
    with pytest.raises(ValueError):
        _req(max_tokens=100, budget=200)  # greater


def test_budget_lt_max_tokens_accepted():
    r = _req(max_tokens=1000, budget=200)
    assert r.thinking["budget_tokens"] == 200


def test_thinking_disabled_unaffected():
    r = AnthropicMessagesRequest(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        thinking={"type": "disabled"},
    )
    assert r.max_tokens == 50
