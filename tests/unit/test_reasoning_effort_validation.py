"""reasoning_effort accepts the OpenAI / Anthropic vocabulary and rejects anything else with a 400.

The spec is the floor (API row 385): OpenAI ReasoningEffort and Anthropic output_config.effort
both list low/medium/high/xhigh, OpenAI adds none/minimal and Anthropic adds max. Values outside
that set are a validation error that names the accepted values, on every request model.
"""

import pytest
from pydantic import ValidationError

from yunshu_engine.output_limits import ACCEPTED_EFFORTS, check_effort
from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest
from yunshu_gateway.routers.chat import ChatCompletionRequest
from yunshu_gateway.routers.completions import CompletionRequest
from yunshu_gateway.routers.responses import ResponsesRequest

MSG = [{"role": "user", "content": "hi"}]


def _chat(effort):
    return ChatCompletionRequest(model="m", messages=MSG, reasoning_effort=effort)


def _responses(effort):
    return ResponsesRequest(model="m", input="hi", reasoning_effort=effort)


def _completion(effort):
    return CompletionRequest(model="m", prompt="hi", reasoning_effort=effort)


def _anthropic(effort):
    return AnthropicMessagesRequest(
        model="m", max_tokens=8, messages=MSG, reasoning_effort=effort
    )


BUILDERS = [_chat, _responses, _completion, _anthropic]


def test_accepted_vocabulary_is_the_spec_union():
    assert ACCEPTED_EFFORTS == (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )


@pytest.mark.parametrize("effort", ACCEPTED_EFFORTS)
@pytest.mark.parametrize("build", BUILDERS)
def test_spec_values_are_accepted(build, effort):
    build(effort)


@pytest.mark.parametrize("build", BUILDERS)
def test_unknown_value_is_rejected_and_lists_accepted_values(build):
    with pytest.raises(ValidationError) as exc:
        build("ultra")
    text = str(exc.value)
    assert "reasoning_effort" in text
    for level in ACCEPTED_EFFORTS:
        assert level in text


def test_check_effort_normalizes_case_and_passes_none():
    assert check_effort(None) is None
    assert check_effort(" High ") == "high"
    with pytest.raises(ValueError, match="accepted values"):
        check_effort("extreme")
