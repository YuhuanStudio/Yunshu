"""CPU regressions derived from TensorFold 0620's history and argument defects."""

import json

import pytest

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.tool_arguments import coerce_tool_arguments, tool_schemas
from yunshu_engine.vlm_engine import VLMEngine
from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    _convert_anthropic_messages,
    _enforce_anthropic_tool_choice,
)
from yunshu_gateway.routers.chat import ChatMessage, _extract_messages


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_openai_echo_preserves_interleaved_reasoning(field):
    messages = _extract_messages(
        [
            ChatMessage.model_validate(
                {"role": "assistant", "content": None, field: "check weather"}
            )
        ]
    )
    assert messages == [
        {"role": "assistant", "content": "", "reasoning_content": "check weather"}
    ]


@pytest.mark.parametrize("args", [None, "", "{}"])
def test_openai_empty_arguments_accepted(args):
    msg = ChatMessage.model_validate(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "ping", "arguments": args}}
            ],
        }
    )
    assert _extract_messages([msg])[0]["tool_calls"][0]["function"]["arguments"] == "{}"


@pytest.mark.parametrize("args", [None, "", "null"])
@pytest.mark.parametrize("path", ["text", "vlm"])
def test_empty_history_arguments_render_no_synthetic_value(args, path):
    calls = [{"id": "c1", "function": {"name": "ping", "arguments": args}}]
    if path == "text":
        calls = BatchedEngine._normalize_messages_for_chat_template(
            [{"role": "assistant", "tool_calls": calls}]
        )[0]["tool_calls"]
    else:
        calls = VLMEngine._normalize_vlm_tool_calls(calls)
    assert calls[0]["function"]["arguments"] == {}


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_union_numeric_arguments(keyword, wire):
    schema = {
        "type": "object",
        "properties": {"days": {keyword: [{"type": "integer"}, {"type": "null"}]}},
    }
    tool = (
        {"function": {"name": "weather", "parameters": schema}}
        if wire == "openai"
        else {"name": "weather", "input_schema": schema}
    )
    assert json.loads(
        coerce_tool_arguments("weather", '{"days":"3"}', tool_schemas([tool]))
    ) == {"days": 3}


def test_anthropic_null_input_is_no_arguments():
    messages, _ = _convert_anthropic_messages(
        [
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "check weather"},
                    {"type": "tool_use", "id": "c1", "name": "ping", "input": None},
                ],
            )
        ]
    )
    assert messages[0]["reasoning_content"] == "check weather"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{}"


@pytest.mark.parametrize("choice", ["none", {"type": "none"}])
def test_anthropic_none_enforcement(choice):
    assert (
        _enforce_anthropic_tool_choice([{"name": "ping", "arguments": "{}"}], choice)
        == []
    )


@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_missing_reasoning_restored_with_renumbered_ids_and_isolated_prefix(wire):
    from yunshu_gateway.middleware.tool_reasoning import ToolReasoningCache

    cache = ToolReasoningCache(entries=1)
    body = {"model": "Qwen3.8", "messages": [{"role": "user", "content": "weather?"}]}
    if wire == "openai":
        turn = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "orig",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Taipei"}'},
                }
            ],
        }
        renamed = json.loads(json.dumps(turn).replace("orig", "call_0_0"))
        renamed["content"] = ""
        explicit = {**renamed, "reasoning_content": ""}
    else:
        turn = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "orig",
                    "name": "weather",
                    "input": {"city": "Taipei"},
                }
            ],
        }
        renamed = json.loads(json.dumps(turn).replace("orig", "call_0_0"))
        explicit = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": ""}, *renamed["content"]],
        }
    cache.remember(body, turn, "check the city", "caller-a")
    history = {**body, "messages": [*body["messages"], renamed]}
    restored = cache.restore(history, "caller-a")["messages"][-1]
    if wire == "openai":
        assert restored["reasoning_content"] == "check the city"
    else:
        assert restored["content"][0] == {
            "type": "thinking",
            "thinking": "check the city",
        }
    assert cache.restore(history, "caller-b") == history
    changed = {**history, "model": "different-model"}
    assert cache.restore(changed, "caller-a") == changed
    changed = {
        **history,
        "messages": [{"role": "user", "content": "different session"}, renamed],
    }
    assert cache.restore(changed, "caller-a") == changed
    assert (
        cache.restore({**body, "messages": [*body["messages"], explicit]}, "caller-a")[
            "messages"
        ][-1]
        == explicit
    )
    # A collision with different hidden reasoning is deliberately not restored.
    cache.remember(body, turn, "different reasoning", "caller-a")
    assert cache.restore(history, "caller-a") == history


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wire", ["openai", "anthropic"])
@pytest.mark.asyncio
async def test_wire_reasoning_recovered_after_complete_tool_reply(stream, wire):
    import httpx

    from yunshu_gateway.middleware.tool_reasoning import ToolReasoningMiddleware

    received = []
    turn = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "check city",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "weather", "arguments": "{}"},
            }
        ],
    }
    anth_turn = [
        {"type": "thinking", "thinking": "check city"},
        {"type": "tool_use", "id": "c1", "name": "weather", "input": {}},
    ]

    async def app(scope, receive, send):
        event = await receive()
        received.append(json.loads(event["body"]))
        if wire == "openai":
            result = {"choices": [{"message": turn}]}
            events = [
                {
                    "choices": [
                        {"index": 0, "delta": {"reasoning_content": "check city"}}
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, **turn["tool_calls"][0]}]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        else:
            result = {"stop_reason": "tool_use", "content": anth_turn}
            events = [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": anth_turn[0],
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": anth_turn[1],
                },
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
                {"type": "message_stop"},
            ]
        raw = (
            (
                "".join("data: " + json.dumps(e) + "\n\n" for e in events)
                + ("data: [DONE]\n\n" if wire == "openai" else "")
            ).encode()
            if stream
            else json.dumps(result).encode()
        )
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for i in range(0, len(raw), 7):
            await send(
                {
                    "type": "http.response.body",
                    "body": raw[i : i + 7],
                    "more_body": i + 7 < len(raw),
                }
            )

    middleware = ToolReasoningMiddleware(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=middleware), base_url="http://test"
    ) as client:
        body = {
            "model": "m",
            "stream": stream,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
            "messages": [{"role": "user", "content": "weather?"}],
        }
        path = "/v1/chat/completions" if wire == "openai" else "/v1/messages"
        await client.post(path, json=body)
        history_turn = (
            {k: v for k, v in turn.items() if k != "reasoning_content"}
            if wire == "openai"
            else {"role": "assistant", "content": anth_turn[1:]}
        )
        await client.post(
            path, json={**body, "messages": [*body["messages"], history_turn]}
        )
    last = received[-1]["messages"][-1]
    assert (
        last.get("reasoning_content")
        if wire == "openai"
        else last["content"][0].get("thinking")
    ) == "check city"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["string", "null"]},
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"enum": ["123", "456"]},
    ],
)
@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_glm_union_numeric_looking_strings_preserved(schema, wire):
    from types import SimpleNamespace

    from yunshu_engine.tool_format import formats_for_tokenizer, parse_tool_output

    params = {"type": "object", "properties": {"code": schema}}
    tool = (
        {"function": {"name": "lookup", "parameters": params}}
        if wire == "openai"
        else {"name": "lookup", "input_schema": params}
    )
    formats = formats_for_tokenizer(
        SimpleNamespace(chat_template="<tool_call><arg_key><arg_value>")
    )
    calls, _ = parse_tool_output(
        "<tool_call>lookup<arg_key>code</arg_key><arg_value>123</arg_value></tool_call>",
        formats,
        [tool],
    )
    assert json.loads(calls[0]["arguments"]) == {"code": "123"}


@pytest.mark.parametrize("family", ["qwen35", "qwen38", "glm53"])
@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_real_templates_keep_reasoning_and_empty_tool_arguments(family, wire):
    from pathlib import Path

    from jinja2 import Environment

    source = (
        Path(__file__).parents[1]
        / "fixtures"
        / "glmspark_templates"
        / f"{family}.jinja"
    ).read_text()
    if wire == "openai":
        messages = _extract_messages(
            [
                ChatMessage(role="user", content="weather?"),
                ChatMessage(
                    role="assistant",
                    content=None,
                    reasoning_content="check city",
                    tool_calls=[
                        {"id": "c1", "function": {"name": "ping", "arguments": ""}}
                    ],
                ),
                ChatMessage(role="tool", tool_call_id="c1", content=None),
            ]
        )
    else:
        messages, _ = _convert_anthropic_messages(
            [
                AnthropicMessage(role="user", content="weather?"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {"type": "thinking", "thinking": "check city"},
                        {"type": "tool_use", "id": "c1", "name": "ping", "input": None},
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {"type": "tool_result", "tool_use_id": "c1", "content": None}
                    ],
                ),
            ]
        )
    messages = BatchedEngine._normalize_messages_for_chat_template(messages)
    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)

    def reject(message):
        raise ValueError(message)

    rendered = env.from_string(source).render(
        messages=messages, tools=[], add_generation_prompt=True, raise_exception=reject
    )
    assert "check city" in rendered
    assert "None" not in rendered
    assert "<tool_call>" in rendered and "ping" in rendered
    assert "value" not in rendered


@pytest.mark.parametrize("wire", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "finish,terminal,recover",
    [
        ("stop", 99, True),
        ("stop", 98, False),
        ("length", 99, False),
        ("cancel", 99, False),
    ],
)
def test_glm_think_call_only_rescued_on_observation_eos(
    wire, finish, terminal, recover
):
    from types import SimpleNamespace

    from yunshu_engine.tool_format import formats_for_tokenizer, parse_tool_output
    from yunshu_gateway.streaming import extract_thinking

    engine = VLMEngine.__new__(VLMEngine)
    engine._tokenizer = SimpleNamespace(
        chat_template="<arg_key><arg_value><tool_call>", encode=lambda text, **kw: [99]
    )
    call = (
        "<tool_call>ping<arg_key>code</arg_key><arg_value>123</arg_value></tool_call>"
    )
    tool = {
        "name": "ping",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}},
    }
    if wire == "openai":
        tool = {
            "type": "function",
            "function": {"name": tool["name"], "parameters": tool["input_schema"]},
        }
    events = [("check city ", 1, "reasoning", None, 1, None)]
    events += [(c, i + 2, "reasoning", None, i + 2, None) for i, c in enumerate(call)]
    events.append(("", terminal, "normal", finish, len(call) + 1, None))
    engine._runner_events_impl = lambda *a, **kw: iter(events)
    output = list(engine._runner_events([1], tool_recovery_tools=[tool]))
    reasoning = "".join(e[0] for e in output if e[2] == "reasoning")
    visible = "".join(e[0] for e in output if e[2] != "reasoning")
    split = extract_thinking("<think>" + reasoning + "</think>" + visible, "glm53")
    calls, _ = parse_tool_output(
        split[1], formats_for_tokenizer(engine._tokenizer), [tool]
    )
    assert bool(calls) == recover
    assert (call in reasoning) == (not recover)
    assert output[-1][3] == finish
    assert [e[1] for e in output if e[1] is not None] == [e[1] for e in events]


@pytest.mark.parametrize(
    "schema,value,expected",
    [
        ({"type": ["string", "null"]}, "null", None),
        ({"enum": [1, 2, 3]}, "3", 3),
        ({"const": True}, "true", True),
        ({"type": "integer"}, '"3"', 3),
        ({"type": "boolean"}, '"true"', True),
        ({"anyOf": [{"type": "integer"}, {}]}, "3", "3"),
    ],
)
@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_full_schema_argument_values(schema, value, expected, wire):
    params = {"type": "object", "properties": {"v": schema}}
    tool = (
        {"function": {"name": "f", "parameters": params}}
        if wire == "openai"
        else {"name": "f", "input_schema": params}
    )
    assert (
        json.loads(
            coerce_tool_arguments(
                "f",
                json.dumps({"v": value}),
                tool_schemas([tool]),
                raw_text_values=True,
            )
        )["v"]
        == expected
    )


@pytest.mark.parametrize(
    "tail",
    [
        "<tool_call>unknown<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
        "<tool_call>ping<arg_key>x</arg_key><arg_value>1",
        "<tool_call>ping<arg_key>x</arg_key><arg_value>1</arg_value></tool_call> example",
    ],
)
def test_thinkcall_unknown_incomplete_or_prose_stays_reasoning(tail):
    from types import SimpleNamespace

    from yunshu_engine.tool_format import formats_for_tokenizer
    from yunshu_engine.tool_thinking import recover_tool_events

    formats = formats_for_tokenizer(
        SimpleNamespace(chat_template="<arg_key><arg_value><tool_call>")
    )
    events = [
        (tail, 1, "reasoning", None, 1, None),
        ("", 99, "normal", "stop", 1, None),
    ]
    out = list(
        recover_tool_events(
            events,
            observation_id=99,
            formats=formats,
            tools=[{"name": "ping", "input_schema": {}}],
        )
    )
    assert "".join(e[0] for e in out if e[2] == "normal") == ""
    assert "".join(e[0] for e in out if e[2] == "reasoning") == tail


def test_reasoning_cache_byte_budget_and_eviction():
    from yunshu_gateway.middleware.tool_reasoning import ToolReasoningCache

    cache = ToolReasoningCache(entries=1, max_bytes=4)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    turn = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "ping", "arguments": "{}"}}],
    }
    cache.remember(body, turn, "🙂", "a")
    assert cache.size == 4
    cache.remember(body, turn, "12345", "b")
    assert cache.size == 4 and len(cache.data) == 1
    cache.remember(body, turn, "abc", "b")
    assert cache.size == 3 and len(cache.data) == 1
    history = {**body, "messages": [*body["messages"], turn]}
    assert cache.restore(history, "a") == history
    assert cache.restore(history, "b")["messages"][-1]["reasoning_content"] == "abc"


def test_null_reasoning_field_is_missing_not_explicit_empty():
    from yunshu_gateway.middleware.tool_reasoning import ToolReasoningCache

    cache = ToolReasoningCache()
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "ping", "arguments": "{}"}}],
    }
    cache.remember(body, turn, "check", "a")
    history = {
        **body,
        "messages": [*body["messages"], {**turn, "reasoning_content": None}],
    }
    assert cache.restore(history, "a")["messages"][-1]["reasoning_content"] == "check"


@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_valid_json_nullable_string_null_remains_a_string(wire):
    from yunshu_engine.tool_format import fallback_formats, parse_tool_output

    schema = {"type": "object", "properties": {"v": {"type": ["string", "null"]}}}
    tool = (
        {"function": {"name": "f", "parameters": schema}}
        if wire == "openai"
        else {"name": "f", "input_schema": schema}
    )
    calls, _ = parse_tool_output(
        '<tool_call>{"name":"f","arguments":{"v":"null"}}</tool_call>',
        fallback_formats(),
        [tool],
    )
    assert json.loads(calls[0]["arguments"])["v"] == "null"


@pytest.mark.parametrize("family", ["glm", "qwen"])
@pytest.mark.parametrize("wire", ["openai", "anthropic"])
def test_native_xml_unquoted_nullable_null_is_json_null(family, wire):
    from types import SimpleNamespace

    from yunshu_engine.tool_format import formats_for_tokenizer, parse_tool_output

    schema = {"type": "object", "properties": {"v": {"type": ["string", "null"]}}}
    tool = (
        {"function": {"name": "f", "parameters": schema}}
        if wire == "openai"
        else {"name": "f", "input_schema": schema}
    )
    body = (
        "f<arg_key>v</arg_key><arg_value>null</arg_value>"
        if family == "glm"
        else "<function=f><parameter=v>null</parameter></function>"
    )
    formats = formats_for_tokenizer(
        SimpleNamespace(chat_template="<tool_call>\n" + body)
    )
    calls, _ = parse_tool_output("<tool_call>" + body + "</tool_call>", formats, [tool])
    assert json.loads(calls[0]["arguments"])["v"] is None


def test_lm_tool_parser_fallback_accepts_a_tokenizer_not_template_text():
    from types import SimpleNamespace

    from yunshu_engine.tool_format import formats_for_tokenizer

    tokenizer = SimpleNamespace(chat_template="{{ messages }}", get_vocab=lambda: {})
    assert [f.name for f in formats_for_tokenizer(tokenizer)] == [
        "yunshu_json",
        "json_message",
    ]
