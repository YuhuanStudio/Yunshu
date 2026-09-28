"""Native tool rendering for VLMs + schema-typed tool-call arguments."""

import json
from types import SimpleNamespace

import pytest

from yunshu_engine.tool_arguments import (
    coerce_tool_arguments,
    coerce_tool_calls,
    tool_schemas,
)
from yunshu_engine.tool_call_streamer import ToolCallStreamer

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city", "days"],
        },
    },
}
RICH = {
    "type": "function",
    "function": {
        "name": "f",
        "parameters": {
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array", "items": {"type": "integer"}},
                "o": {
                    "type": "object",
                    "properties": {"k": {"type": "integer"}},
                },
                "s": {"type": "string"},
                "u": {},
                "opt": {"type": ["integer", "null"]},
            },
        },
    },
}


def _coerce(args, tools=(RICH,)):
    return json.loads(
        coerce_tool_arguments("f", json.dumps(args), tool_schemas(list(tools)))
    )


def test_coerces_declared_types():
    out = _coerce(
        {
            "i": "3",
            "n": "2.5",
            "b": "True",
            "a": "[1, 2]",
            "o": '{"k": "4"}',
            "s": "07",
            "u": "5",
            "opt": "null",
        }
    )
    assert out == {
        "i": 3,
        "n": 2.5,
        "b": True,
        "a": [1, 2],
        "o": {"k": 4},
        "s": "07",
        "u": "5",
        "opt": None,
    }


def test_nested_values_and_items_typed():
    out = _coerce({"a": ["1", "2"], "o": {"k": "9"}})
    assert out == {"a": [1, 2], "o": {"k": 9}}


def test_bad_values_left_unchanged():
    out = _coerce({"i": "three", "n": "x", "b": "yes", "a": "[1,", "o": "nope"})
    assert out == {"i": "three", "n": "x", "b": "yes", "a": "[1,", "o": "nope"}


def test_never_raises_and_passthrough():
    schemas = tool_schemas([RICH])
    assert coerce_tool_arguments("f", "not json", schemas) == "not json"
    assert coerce_tool_arguments("f", "[1]", schemas) == "[1]"
    assert coerce_tool_arguments("unknown", '{"i": "3"}', schemas) == '{"i": "3"}'
    already = '{"i": 3}'
    assert coerce_tool_arguments("f", already, schemas) is already


def test_coerce_tool_calls_accepts_pydantic_like_tools():
    tool = SimpleNamespace(model_dump=lambda: WEATHER)
    calls = coerce_tool_calls(
        [{"name": "get_weather", "arguments": '{"city": "Oslo", "days": "3"}'}],
        [tool],
    )
    assert json.loads(calls[0]["arguments"]) == {"city": "Oslo", "days": 3}


QWEN_XML = (
    "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n"
    "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)


@pytest.mark.parametrize("step", [1, 3, 7])
def test_streamer_types_qwen_xml_call(step):
    st = ToolCallStreamer(model_name="Qwen3.8-27B", tools=[WEATHER])
    outs = []
    for i in range(0, len(QWEN_XML), step):
        outs += st.process_token(QWEN_XML[i : i + step])
    outs += st.flush()
    calls = [o.tool_call for o in outs if o.tool_call is not None]
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Oslo", "days": 3}
    # XML arguments are never streamed as raw text deltas.
    assert not any(o.tool_call_args_delta for o in outs)
    text = "".join(o.text for o in outs)
    assert "<function=" not in text and "<parameter" not in text


def test_streamer_without_tools_keeps_strings():
    st = ToolCallStreamer(model_name="Qwen3.8-27B")
    outs = st.process_token(QWEN_XML) + st.flush()
    (call,) = [o.tool_call for o in outs if o.tool_call is not None]
    assert json.loads(call.arguments)["days"] == "3"


# ── Gateway: native tools vs injected prompt on the VLM path ──


def _chat_req(tool_choice):
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "weather in Oslo for 3 days?"}],
        tools=[WEATHER],
        tool_choice=tool_choice,
    )


class _Engine:
    def __init__(self, native):
        self.native = native

    def supports_native_tools(self):
        return self.native


@pytest.mark.parametrize("choice", [None, "auto"])
def test_vlm_tool_plan_native_for_auto(choice):
    from yunshu_gateway.routers.chat import _vlm_tool_plan

    msgs = [{"role": "user", "content": "hi"}]
    out, native = _vlm_tool_plan(_chat_req(choice), _Engine(True), msgs)
    assert out is msgs
    assert native and native[0]["function"]["name"] == "get_weather"


@pytest.mark.parametrize(
    "choice",
    ["none", "required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_vlm_tool_plan_injects_for_forced_choice(choice):
    from yunshu_gateway.routers.chat import _vlm_tool_plan

    msgs = [{"role": "user", "content": "hi"}]
    out, native = _vlm_tool_plan(_chat_req(choice), _Engine(True), msgs)
    assert native is None
    assert out is not msgs and out[0]["role"] == "system"


def test_vlm_tool_plan_injects_without_template_support():
    from yunshu_gateway.routers.chat import _vlm_tool_plan

    msgs = [{"role": "user", "content": "hi"}]
    out, native = _vlm_tool_plan(_chat_req("auto"), _Engine(False), msgs)
    assert native is None and out[0]["role"] == "system"
    out, native = _vlm_tool_plan(_chat_req("auto"), object(), msgs)
    assert native is None


# ── VLMEngine: tools travel in the explicit template extras ──


class _CapturingTokenizer:
    chat_template = "{% if tools %}{{ tools }}{% endif %}{{ reasoning_effort }}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "PROMPT:" + json.dumps(kwargs.get("tools"), sort_keys=True)

    def encode(self, text, add_special_tokens=True):
        return [len(text)]


def _vlm_engine():
    from yunshu_engine.vlm_engine import VLMEngine, _VLMTextPromptCache

    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = _CapturingTokenizer()
    eng._processor = SimpleNamespace(chat_template=None)
    eng._text_prompt_cache = _VLMTextPromptCache()
    eng._model_path = "/models/Qwen3.8-27B"
    return eng


def test_request_template_extra_collects_tools_and_effort():
    eng = _vlm_engine()
    kwargs = {"tools": [WEATHER], "reasoning_effort": "low"}
    extra = eng._request_template_extra(kwargs)
    assert extra == {"tools": [WEATHER], "reasoning_effort": "low"}
    assert "tools" not in kwargs and "reasoning_effort" not in kwargs
    assert eng.supports_native_tools()


def test_tools_reach_chat_template_and_cache_key():
    eng = _vlm_engine()
    msgs = [{"role": "user", "content": "hi"}]
    extra = eng._request_template_extra({"tools": [WEATHER]})
    with_tools = eng._tokenize_with_cache(msgs, template_extra=extra)
    without = eng._tokenize_with_cache(msgs)
    assert eng._tokenizer.calls[0]["tools"] == [WEATHER]
    assert "tools" not in eng._tokenizer.calls[1]
    assert with_tools.tolist() != without.tolist()


def test_streamed_xml_call_starts_with_id_and_name():
    from yunshu_engine.tool_call_streamer import ToolCallStreamer

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            },
        }
    ]
    s = ToolCallStreamer(model_name="Qwen3.8-27B", tools=tools)
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTaipei\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    outs = []
    for i in range(0, len(text), 3):
        outs += s.process_token(text[i : i + 3])
    outs += s.flush()
    kinds = [
        ("start" if o.tool_call_start else "call")
        for o in outs
        if o.tool_call_start or o.tool_call
    ]
    assert kinds == ["start", "call"]
    start = next(o.tool_call_start for o in outs if o.tool_call_start)
    call = next(o.tool_call for o in outs if o.tool_call)
    assert start.name == "get_weather" and start.id == call.id
