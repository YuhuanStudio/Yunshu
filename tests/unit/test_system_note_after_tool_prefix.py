"""omlx 5a0a15a7: a system note after a tool result must not rewrite the prompt's front."""

import json
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from yunshu_engine.message_adapter import adapt_messages

TEMPLATE = Path(__file__).parent.parent / "fixtures/glmspark_templates/qwen38.jinja"


def _render(messages):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    return env.from_string(TEMPLATE.read_text()).render(
        messages=messages, add_generation_prompt=True, enable_thinking=False
    )


def _tool_turn(i):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "Read", "arguments": {"p": str(i)}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"},
    ]


def _conv(turns, note_each=True):
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug."},
    ]
    for t in range(turns):
        msgs += _tool_turn(t)
        if note_each:
            msgs.append({"role": "system", "content": f"hook note {t}"})
    return msgs


@pytest.mark.parametrize("model", ["Qwen3.8-27B", "unknown-model"])
def test_consecutive_agent_turns_are_prefix_stable(model):
    prompts = [_render(adapt_messages(_conv(t), model)) for t in (1, 2, 3)]
    for a, b in zip(prompts, prompts[1:]):
        head = a[: a.rindex("hook note")]
        assert b.startswith(head)
    assert "hook note 0" in prompts[0] and "hook note 2" in prompts[2]
    assert prompts[2].count("You are an agent.") == 1
    assert prompts[2].index("You are an agent.") < prompts[2].index("hook note 0")


def test_note_before_assistant_stays_in_place_and_multiple_merge():
    msgs = _conv(1, note_each=False) + [
        {"role": "system", "content": "a"},
        {"role": "developer", "content": [{"type": "text", "text": "b"}]},
        {"role": "assistant", "content": "done"},
    ]
    out = adapt_messages(msgs, "Qwen3.8-27B")
    assert [m["role"] for m in out] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert out[4]["content"] == "[System note]\na\n\nb\n[/System note]"
    _render(out)  # the system-first template accepts it


def test_note_between_tool_results_is_still_hoisted():
    call = {"type": "function", "function": {"name": "R", "arguments": {}}}
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [dict(call, id="a"), dict(call, id="b")],
        },
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "system", "content": "tip"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
    ]
    out = adapt_messages(msgs, "Qwen3.8-27B")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool"]
    assert out[0]["content"] == "tip"


def test_token_counter_after_tool_is_dropped_not_noted():
    msgs = _conv(1, note_each=False) + [
        {"role": "system", "content": "<total_tokens>9 tokens left</total_tokens>"}
    ]
    out = adapt_messages(msgs, "Qwen3.8-27B")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
