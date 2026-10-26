"""/tokenize and /apply-template accept OpenAI-style tool-call history.

OpenAI sends ``function.arguments`` as a JSON *string*; chat templates iterate it as a mapping, so
the preview routes answered 400 "Can only get item pairs from a mapping" for any conversation that
had a tool call, while the generation paths (which normalise the string) rendered it."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent))
from test_control_token_injection import _tok  # noqa: E402

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
    "{% for tc in m.tool_calls or [] %}<call {{ tc.function.name }}"
    "{% for k, v in tc.function.arguments|items %} {{ k }}={{ v }}{% endfor %}>{% endfor %}"
    "<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
MESSAGES = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "noargs", "arguments": ""},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]


@pytest.fixture
def tok(monkeypatch):
    from yunshu_gateway.routers import models, tokenize

    tok = _tok()
    tok.chat_template = TEMPLATE
    monkeypatch.setattr(models, "_check_permission", lambda *a, **k: None)
    monkeypatch.setattr(models, "_check_model_access", lambda *a, **k: None)
    monkeypatch.setattr(tokenize, "_resolve_tokenizer", lambda m: tok)
    return tok


def test_apply_template_renders_string_tool_call_arguments(tok):
    from yunshu_gateway.routers import tokenize

    out = asyncio.run(
        tokenize.apply_template(
            tokenize.ApplyTemplateRequest(messages=MESSAGES), object()
        )
    )
    assert "<call get_weather city=Paris>" in out["prompt"]
    assert "<call noargs>" in out["prompt"]


def test_tokenize_counts_string_tool_call_arguments(tok):
    from yunshu_gateway.routers import tokenize

    req = tokenize.TokenizeRequest(messages=MESSAGES, add_special_tokens=False)
    out = asyncio.run(tokenize.tokenize(req, object()))
    rendered = tok.apply_chat_template(
        [
            {
                **m,
                "tool_calls": [
                    {**tc, "function": {**tc["function"], "arguments": a}}
                    for tc, a in zip(
                        m["tool_calls"], ({"city": "Paris"}, {}), strict=True
                    )
                ],
            }
            if m.get("tool_calls")
            else m
            for m in MESSAGES
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert out["tokens"] == tok.encode(rendered, add_special_tokens=False)


def test_the_callers_messages_are_not_mutated(tok):
    import copy

    from yunshu_gateway.routers import tokenize

    keep = copy.deepcopy(MESSAGES)
    asyncio.run(
        tokenize.apply_template(
            tokenize.ApplyTemplateRequest(messages=MESSAGES), object()
        )
    )
    assert keep == MESSAGES
