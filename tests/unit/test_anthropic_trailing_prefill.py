"""Anthropic Messages prefill: a trailing assistant text turn reaches the template as an open turn."""

from tests.unit.test_assistant_prefill import _engine, _RecordingTokenizer
from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    _convert_anthropic_messages,
)


def test_trailing_assistant_text_survives_conversion_and_continues():
    msgs, _ = _convert_anthropic_messages(
        [
            AnthropicMessage(role="user", content="What is 2+2?"),
            AnthropicMessage(role="assistant", content="The answer is"),
        ]
    )
    assert msgs[-1] == {"role": "assistant", "content": "The answer is"}
    tok = _RecordingTokenizer()
    out = _engine(tok)._apply_chat_template(msgs)
    assert tok.last_kwargs.get("continue_final_message") is True
    assert out.endswith("The answer is")
