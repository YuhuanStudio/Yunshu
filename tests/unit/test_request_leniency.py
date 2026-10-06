"""Requests the official SDKs / popular clients send that the gateway used to answer 400 (or truncate)."""

import pytest

from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest
from yunshu_gateway.routers.chat import ChatCompletionRequest
from yunshu_gateway.routers.completions import CompletionRequest
from yunshu_gateway.routers.responses import ResponsesRequest

U = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "model,kw",
    [
        (ChatCompletionRequest, dict(messages=U)),
        (CompletionRequest, dict(prompt="x")),
        (ResponsesRequest, dict(input="x")),
        (AnthropicMessagesRequest, dict(messages=U, max_tokens=10)),
    ],
)
def test_explicit_null_means_unset(model, kw):
    nulls = {k: None for k in ("temperature", "top_p", "top_k", "stream", "max_tokens", "n", "seed", "min_p") if k in model.model_fields}
    if model is AnthropicMessagesRequest:
        nulls.pop("max_tokens", None)
    r = model(model="m", **kw, **nulls)
    assert r.stream is False if "stream" in model.model_fields else True
    assert r.temperature == model.model_fields["temperature"].default


def test_negative_top_k_is_disabled():
    for model, kw in [(ChatCompletionRequest, dict(messages=U)), (AnthropicMessagesRequest, dict(messages=U, max_tokens=5)), (ResponsesRequest, dict(input="x"))]:
        assert model(model="m", top_k=-1, **kw).top_k == 0


def test_omitted_max_tokens_follows_the_setting(monkeypatch):
    from yunshu_engine import settings

    assert ChatCompletionRequest(model="m", messages=U).max_tokens == settings.get("YUNSHU_DEFAULT_MAX_TOKENS") >= 8192
    assert ResponsesRequest(model="m", input="x").max_output_tokens == settings.get("YUNSHU_DEFAULT_MAX_TOKENS")
    assert ChatCompletionRequest(model="m", messages=U, max_tokens=200000).max_tokens == 200000
    assert AnthropicMessagesRequest(model="m", messages=U, max_tokens=200000).max_tokens == 200000


def test_allowed_tools_restricts_tools_chat_and_responses():
    t = lambda n: {"type": "function", "function": {"name": n, "parameters": {}}}  # noqa: E731
    r = ChatCompletionRequest(
        model="m",
        messages=U,
        tools=[t("a"), t("b")],
        tool_choice={"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [t("a")]}},
    )
    assert r.tool_choice == "required" and [x.function.name for x in r.tools] == ["a"]
    q = ResponsesRequest(
        model="m",
        input="x",
        tools=[{"type": "function", "name": "a", "parameters": {}}, {"type": "function", "name": "b", "parameters": {}}],
        tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "b"}]},
    )
    assert q.tool_choice == "auto" and [x.name for x in q.tools] == ["b"]
