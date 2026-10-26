"""The context clamp must keep the Anthropic invariant budget_tokens < max_tokens."""

from yunshu_gateway.routers.anthropic import _clamped_thinking_budget
from yunshu_gateway.token_budget import plan


def test_context_clamp_leaves_an_answer_token():
    # 16000-token window, 10000-token prompt: 6000 left; asked for 32000 out / 16000 thinking
    tb = plan(16_000, 10_000, 32_000, 16_000)
    assert tb is not None and tb.clamped
    assert tb.max_tokens_granted == 6000
    budget = _clamped_thinking_budget(tb.thinking_budget_granted, tb.max_tokens_granted)
    assert budget is not None and budget < tb.max_tokens_granted


def test_clamped_budget_helper():
    assert _clamped_thinking_budget(6000, 6000) == 5999
    assert _clamped_thinking_budget(1024, 6000) == 1024
    assert _clamped_thinking_budget(None, 6000) is None
    assert _clamped_thinking_budget(1, 1) == 1
