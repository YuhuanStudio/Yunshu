"""OpenAI accepts whitespace-only user/system content ("\\n", " "); only a missing or
empty-string user prompt (nothing at all to respond to) is a malformed request."""

import pytest
from pydantic import ValidationError

from yunshu_gateway.routers.chat import ChatCompletionRequest


@pytest.mark.parametrize("ws", ["\n", " ", "  \t\n"])
def test_whitespace_user_and_system_content_accepted(ws):
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "system", "content": ws}, {"role": "user", "content": ws}],
    )
    assert req.messages[1].content == ws


def test_empty_user_content_still_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": ""}])
