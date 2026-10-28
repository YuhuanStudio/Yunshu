"""A malformed Anthropic tool_choice is a 400, never silently run as auto (mlxcel 465299e9)."""

import pytest
from pydantic import ValidationError

from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest


def _req(tool_choice):
    return AnthropicMessagesRequest(
        model="m",
        max_tokens=5,
        messages=[{"role": "user", "content": "x"}],
        tool_choice=tool_choice,
    )


@pytest.mark.parametrize(
    "tc, msg",
    [
        ({"type": "bogus"}, "tool_choice.type must be"),
        ({"type": "tool"}, "tool_choice.name is required"),
    ],
)
def test_malformed_tool_choice_is_rejected(tc, msg):
    with pytest.raises(ValidationError, match=msg):
        _req(tc)


@pytest.mark.parametrize(
    "tc", [{"type": "auto"}, {"type": "any"}, {"type": "none"}, "auto"]
)
def test_valid_tool_choice_passes(tc):
    _req(tc)
