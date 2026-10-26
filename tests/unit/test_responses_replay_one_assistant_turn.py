"""Replaying a Responses tool loop must rebuild the ONE assistant turn the model generated.

A response with parallel tool calls is output as [reasoning, message(text=""), function_call A,
function_call B]. Replayed item by item (stateless `input`, or a previous_response_id chain)
it became a blank assistant turn plus one assistant turn per call, so the rendered history
differed from what was generated and KV-cached."""

from __future__ import annotations

import jinja2
import pytest

from yunshu_gateway.routers import responses as resp_mod

# Qwen-style rendering: one <|im_start|>assistant block per assistant message.
_TEMPLATE = jinja2.Template(
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}"
    "{{ m.content }}"
    "{% for tc in m.tool_calls or [] %}<tool_call>{{ tc.function.name }}</tool_call>{% endfor %}"
    "<|im_end|>\n{% endfor %}"
)

GENERATED = [
    {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "plan"}],
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": ""}],
    },
    {"type": "function_call", "call_id": "a", "name": "read", "arguments": "{}"},
    {"type": "function_call", "call_id": "b", "name": "ls", "arguments": "{}"},
]
EXPECTED = (
    "<|im_start|>user\ngo<|im_end|>\n"
    "<|im_start|>assistant\n<think>plan</think>"
    "<tool_call>read</tool_call><tool_call>ls</tool_call><|im_end|>\n"
    "<|im_start|>tool\nr1<|im_end|>\n<|im_start|>tool\nr2<|im_end|>\n"
)


def _outputs():
    return [
        {"type": "function_call_output", "call_id": "a", "output": "r1"},
        {"type": "function_call_output", "call_id": "b", "output": "r2"},
    ]


def test_stateless_replay_is_one_assistant_turn():
    req = resp_mod.ResponsesRequest(
        model="m",
        input=[{"role": "user", "content": "go"}, *GENERATED, *_outputs()],
    )
    msgs = resp_mod._convert_to_messages(req)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool"]
    assert [tc["id"] for tc in msgs[1]["tool_calls"]] == ["a", "b"]
    assert _TEMPLATE.render(messages=msgs) == EXPECTED


def test_chain_replay_is_one_assistant_turn(monkeypatch):
    store = {
        "r1": {
            "previous_response_id": None,
            "_input_messages": [{"role": "user", "content": "go"}],
            "output": GENERATED,
        }
    }
    monkeypatch.setattr(resp_mod, "_get_stored_response", lambda i: store.get(i))
    monkeypatch.setattr(resp_mod, "_owns_stored", lambda request, p: True)
    req = resp_mod.ResponsesRequest(
        model="m", input=_outputs(), previous_response_id="r1"
    )
    msgs = resp_mod._apply_previous_response_chain(
        req, resp_mod._convert_to_messages(req), None
    )
    assert _TEMPLATE.render(messages=msgs) == EXPECTED


def test_stateless_and_chain_replay_render_identically(monkeypatch):
    # (same expectation as above from both entry points: they agree with each other)
    test_stateless_replay_is_one_assistant_turn()
    test_chain_replay_is_one_assistant_turn(monkeypatch)


@pytest.mark.parametrize(
    "items, roles",
    [
        # text + call stay one turn; the text is not lost
        (
            [
                {"role": "assistant", "content": "checking"},
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "n",
                    "arguments": "{}",
                },
            ],
            ["assistant"],
        ),
        # two real assistant texts with a user between stay separate
        (
            [
                {"role": "assistant", "content": "x"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "y"},
            ],
            ["assistant", "user", "assistant"],
        ),
    ],
)
def test_ordinary_turns_unchanged(items, roles):
    msgs = resp_mod._convert_to_messages(
        resp_mod.ResponsesRequest(model="m", input=items)
    )
    assert [m["role"] for m in msgs] == roles
    if roles == ["assistant"]:
        assert msgs[0]["content"] == "checking" and len(msgs[0]["tool_calls"]) == 1
