"""Tool-call formats: detection from the chat template, parsing, streaming.

Model outputs below are recorded from real checkpoints where available
(Gemma-4 E4B: docs/research/runs/2026-09-28-families/gemma4-e4b-matrix.jsonl)
or copied from each family's documented format.
"""

import json
from types import SimpleNamespace

import pytest

from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.tool_format import (
    DEEPSEEK,
    INJECTED_JSON,
    JSON_MESSAGE,
    fallback_formats,
    formats_for_tokenizer,
    native_format,
    parse_tool_output,
    tool_formats,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "metric": {"type": "boolean"},
            },
        },
    },
}
TIME = {
    **WEATHER,
    "function": {**WEATHER["function"], "name": "get_time"},
}
TOOLS = [WEATHER, TIME]


def _tok(template):
    return SimpleNamespace(chat_template=template)


# Minimal chat templates carrying each family's markers (what the registry reads).
TEMPLATES = {
    "gemma4": "{{ '<|tool_call>call:' }}{{ '<tool_call|>' }}",
    "qwen3_coder": "{{ '<tool_call>\\n<function=' }}",
    "glm47": "{{ '<tool_call>' }}{{ '<arg_key>' }}",
    "mistral": "{{ '[TOOL_CALLS]' }}",
    "pythonic": "{{ '<|tool_call_start|>' }}{{ '<|tool_call_end|>' }}",
    "deepseek": "{{ '<｜tool▁calls▁begin｜>' }}",
}


def _formats(name):
    return formats_for_tokenizer(_tok(TEMPLATES[name]))


GEMMA = (
    '<|tool_call>call:get_weather{city:<|"|>Taipei<|"|>,days:3,metric:true}<tool_call|>'
)
QWEN = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nTaipei\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)
GLM = (
    "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Taipei</arg_value>\n"
    "<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>"
)
MISTRAL_V11 = '[TOOL_CALLS]get_weather[ARGS]{"city": "Taipei", "days": 3}'
MISTRAL_V3 = (
    '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Taipei", "days": 3}}]'
)
PYTHONIC = '<|tool_call_start|>[get_weather(city="Taipei", days=3)]<|tool_call_end|>'
INJECTED = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Taipei", "days": "3"}}'
    "</tool_call>"
)
DS = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
    '```json\n{"city": "Taipei", "days": 3}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
)

CASES = [
    ("gemma4", GEMMA, {"city": "Taipei", "days": 3, "metric": True}),
    ("qwen3_coder", QWEN, {"city": "Taipei", "days": 3}),
    ("glm47", GLM, {"city": "Taipei", "days": 3}),
    ("mistral", MISTRAL_V11, {"city": "Taipei", "days": 3}),
    ("mistral", MISTRAL_V3, {"city": "Taipei", "days": 3}),
    ("pythonic", PYTHONIC, {"city": "Taipei", "days": 3}),
    ("deepseek", DS, {"city": "Taipei", "days": 3}),
]


# ── detection ──


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_native_format_from_template(name):
    fmt = native_format(_tok(TEMPLATES[name]))
    assert fmt is not None and fmt.name == name


def test_json_tools_template_and_no_template_use_injected_form():
    # Qwen2.5 / Qwen3-Omni templates ask for <tool_call>{json}</tool_call>,
    # the same shape as Yunshu's injected prompt.
    tok = _tok("{{ '<tool_call>' }}{{ tool_call.name }}")
    assert native_format(tok) is None
    assert formats_for_tokenizer(tok) == fallback_formats()
    assert formats_for_tokenizer(_tok(None)) == fallback_formats()
    assert fallback_formats() == (INJECTED_JSON, JSON_MESSAGE)


def test_native_first_then_fallbacks():
    fmts = _formats("gemma4")
    assert [f.name for f in fmts] == ["gemma4", "yunshu_json", "json_message"]


def test_tool_formats_reads_engine_tokenizer_and_caches():
    engine = SimpleNamespace(
        _processor=SimpleNamespace(tokenizer=_tok(TEMPLATES["gemma4"]))
    )
    first = tool_formats(engine)
    assert first[0].name == "gemma4"
    assert tool_formats(engine) is first
    text_engine = SimpleNamespace(_tokenizer=_tok(TEMPLATES["qwen3_coder"]))
    assert tool_formats(text_engine)[0].name == "qwen3_coder"
    assert tool_formats(None) == fallback_formats()


def test_deepseek_detection_is_yunshu_owned():
    assert native_format(_tok(TEMPLATES["deepseek"])) is DEEPSEEK


# ── non-streaming ──


@pytest.mark.parametrize(("name", "text", "args"), CASES)
def test_parse_each_family(name, text, args):
    calls, rest = parse_tool_output(
        "Let me check. " + text + "\nDone.", _formats(name), TOOLS
    )
    assert [c["name"] for c in calls] == ["get_weather"]
    assert json.loads(calls[0]["arguments"]) == args
    assert rest == "Let me check. \nDone." or rest == "Let me check. Done."


def test_injected_form_types_string_values_with_schema():
    calls, rest = parse_tool_output(INJECTED, fallback_formats(), TOOLS)
    assert json.loads(calls[0]["arguments"]) == {"city": "Taipei", "days": 3}
    assert rest == ""


def test_injected_form_parsed_for_native_model_forced_choice():
    # Forced tool_choice injects the JSON form even for Qwen: the qwen3_coder
    # parser rejects the JSON body and yunshu_json reads it.
    calls, _ = parse_tool_output(INJECTED, _formats("qwen3_coder"), TOOLS)
    assert calls and calls[0]["name"] == "get_weather"


def test_pydantic_and_anthropic_tools_type_arguments():
    pyd = SimpleNamespace(model_dump=lambda: WEATHER)
    anth = {"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]}
    for tools in ([pyd], [anth]):
        calls, _ = parse_tool_output(QWEN, _formats("qwen3_coder"), tools)
        assert json.loads(calls[0]["arguments"])["days"] == 3


def test_multiple_calls_in_order():
    text = QWEN + "\n" + QWEN.replace("Taipei", "Oslo")
    calls, rest = parse_tool_output(text, _formats("qwen3_coder"), TOOLS)
    assert [json.loads(c["arguments"])["city"] for c in calls] == ["Taipei", "Oslo"]
    assert rest == ""


def test_json_message_whole_output():
    calls, rest = parse_tool_output(
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Taipei"}}',
        fallback_formats(),
    )
    assert calls[0]["name"] == "get_weather" and rest == ""


def test_plain_json_answer_is_not_a_call():
    for text in (
        '{"answer": 52}',
        'here is some data {"name": "Alice", "arguments": 1} end',
        "just text",
    ):
        calls, rest = parse_tool_output(text, fallback_formats(), TOOLS)
        assert calls == [] and rest == text


def test_unparseable_block_stays_visible():
    text = "<tool_call>not a call</tool_call> tail"
    calls, rest = parse_tool_output(text, fallback_formats(), TOOLS)
    assert calls == [] and rest == text


def test_truncated_call_parsed_to_end():
    calls, _ = parse_tool_output(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "X"}}',
        fallback_formats(),
    )
    assert calls and calls[0]["name"] == "get_weather"


# ── streaming ──


def _stream(text, formats, step=1, splits=None, **kw):
    st = ToolCallStreamer(formats, tools=TOOLS, **kw)
    chunks = (
        [text[a:b] for a, b in zip([0, *splits], [*splits, len(text)], strict=True)]
        if splits
        else [text[i : i + step] for i in range(0, len(text), step)]
    )
    outs = []
    for c in chunks:
        outs += st.process_token(c)
    outs += st.flush()
    return outs


def _summary(outs):
    text = "".join(o.text for o in outs)
    calls = [
        (o.tool_call.name, json.loads(o.tool_call.arguments))
        for o in outs
        if o.tool_call
    ]
    return text, calls


@pytest.mark.parametrize("step", [1, 2, 3, 5, 7, 64])
@pytest.mark.parametrize(("name", "call", "args"), CASES)
def test_stream_each_family(name, call, args, step):
    text = "Let me check. " + call + ("\n" if name == "mistral" else "") + "Done."
    outs = _stream(text, _formats(name), step=step)
    content, calls = _summary(outs)
    assert calls == [("get_weather", args)]
    assert content.replace("\n", "") == "Let me check. Done."
    # the start (id + name) precedes the complete call, same id
    kinds = [
        ("start", o.tool_call_start.id)
        if o.tool_call_start
        else ("call", o.tool_call.id)
        for o in outs
        if o.tool_call_start or o.tool_call
    ]
    assert kinds[0][0] == "start" and kinds[1] == ("call", kinds[0][1])


def test_stream_marker_split_at_every_position():
    fmts = _formats("gemma4")
    text = "Hi " + GEMMA
    for cut in range(1, len(text)):
        content, calls = _summary(_stream(text, fmts, splits=[cut]))
        assert content == "Hi ", cut
        assert calls == [("get_weather", {"city": "Taipei", "days": 3, "metric": True})]


def test_stream_text_is_not_held_back_needlessly():
    st = ToolCallStreamer(_formats("qwen3_coder"))
    assert [o.text for o in st.process_token("hello ")] == ["hello "]
    # a possible marker prefix is held until it resolves
    assert st.process_token("<tool") == []
    assert [o.text for o in st.process_token("box>")] == ["<toolbox>"]


def test_stream_forced_choice_and_parallel_cap():
    two = QWEN + QWEN.replace("get_weather", "get_time")
    _, calls = _summary(
        _stream(two, _formats("qwen3_coder"), forced_tool_name="get_time")
    )
    assert [c[0] for c in calls] == ["get_time"]
    _, calls = _summary(_stream(two, _formats("qwen3_coder"), allow_parallel=False))
    assert [c[0] for c in calls] == ["get_weather"]


def test_stream_json_message_held_then_parsed():
    fmts = fallback_formats()
    content, calls = _summary(
        _stream('{"name": "get_weather", "arguments": {"city": "Oslo"}}', fmts, step=4)
    )
    assert calls == [("get_weather", {"city": "Oslo"})] and content == ""
    content, calls = _summary(_stream('{"answer": 52}', fmts, step=4))
    assert calls == [] and content == '{"answer": 52}'


def test_stream_unparseable_block_released_as_text():
    content, calls = _summary(
        _stream("a <tool_call>nope</tool_call> b", fallback_formats(), step=2)
    )
    assert calls == [] and content == "a <tool_call>nope</tool_call> b"


def test_stream_ids_unique_per_call():
    outs = _stream(QWEN + QWEN, _formats("qwen3_coder"), step=5)
    ids = [o.tool_call.id for o in outs if o.tool_call]
    assert len(ids) == 2 and len(set(ids)) == 2
