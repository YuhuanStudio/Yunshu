"""Wire shapes inspired by vLLM tests/tool_parsers (Apache-2.0).

Independent fixture values and test logic; no vLLM implementation copied.
Sources: test_deepseekv31/v32/v4, test_hermes, test_llama3_json,
test_pythonic, test_glm4_moe/glm47_moe, test_kimi_k2_tool_parser.py.
Harmony: OpenAI harmony message format; Mistral: native TOOL_CALLS JSON.
"""

import json
from types import SimpleNamespace

import pytest

from yunshu_engine import tool_format as tf
from yunshu_engine.tool_call_grammar import ToolSpec
from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.tool_family_grammar import build_native_grammar

ARGS = {"city": "Taipei", "days": 3}
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"],
    "additionalProperties": False,
}
TOOLS = [{"name": "weather", "parameters": SCHEMA}]
JSON = json.dumps({"name": "weather", "arguments": ARGS})
DSBODY = '<｜DSML｜invoke name="weather"><｜DSML｜parameter name="city" string="true">Taipei</｜DSML｜parameter><｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter></｜DSML｜invoke>'
CASES = [
    (
        tf.DEEPSEEK,
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>weather<｜tool▁sep｜>{"city":"Taipei","days":3}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    ),
    (tf.DSML, tf.DSML.start + DSBODY + tf.DSML.end),
    (tf.DSML_V4, tf.DSML_V4.start + DSBODY + tf.DSML_V4.end),
    (tf.HERMES, "<tool_call>" + JSON + "</tool_call>"),
    (tf.LLAMA_JSON, "<|python_tag|>" + JSON),
    (tf.PYTHONIC, '[weather(city="Taipei", days=3)]'),
    (tf.MISTRAL, "[TOOL_CALLS][" + JSON + "]"),
    (tf.GLM, '<tool_call>weather\n{"city":"Taipei","days":3}</tool_call>'),
    (
        tf.GLM,
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Taipei</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
    ),
    (
        tf.KIMI,
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>{"city":"Taipei","days":3}<|tool_call_end|><|tool_calls_section_end|>',
    ),
    (
        tf.HARMONY,
        '<|start|>assistant to=weather<|channel|>commentary<|message|>{"city":"Taipei","days":3}<|ghissue|>',
    ),
]


@pytest.mark.parametrize("fmt,text", CASES)
def test_fixture_nonstream_and_all_boundaries(fmt, text):
    calls, content = tf.parse_tool_output(text, (fmt,), TOOLS)
    assert content == ""
    assert calls[0]["name"] == "weather"
    assert json.loads(calls[0]["arguments"]) == ARGS
    for cut in range(len(text) + 1):
        stream = ToolCallStreamer((fmt,), tools=TOOLS)
        out = (
            stream.process_token(text[:cut])
            + stream.process_token(text[cut:])
            + stream.flush()
        )
        assert "".join(o.text for o in out) == ""
        starts = [o.tool_call_start for o in out if o.tool_call_start]
        complete = [o.tool_call for o in out if o.tool_call]
        assert len(starts) == len(complete) == 1
        assert starts[0].id == complete[0].id
        assert json.loads(complete[0].arguments) == ARGS


@pytest.mark.parametrize(
    "text",
    [
        '[weather(city=__import__("os"))]',
        "[weather(**{})]",
        "[weather(1)]",
        '[weather(city="a",city="b")]',
    ],
)
def test_pythonic_never_executes(text):
    with pytest.raises((ValueError, SyntaxError)):
        tf.PYTHONIC.parse(text, TOOLS)


@pytest.mark.parametrize(
    "fmt",
    [
        tf.DEEPSEEK,
        tf.DSML,
        tf.DSML_V4,
        tf.HERMES,
        tf.LLAMA_JSON,
        tf.PYTHONIC,
        tf.MISTRAL,
        tf.GLM,
        tf.KIMI,
        tf.HARMONY,
    ],
)
def test_native_grammar_compiles_and_accepts_own_wire(fmt):
    from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

    class ByteTokenizer:
        tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
        eos_token_id = 256
        bos_token_id = None

        def __call__(self, text):
            return list(text if isinstance(text, bytes) else text.encode())

    tok = LLTokenizer(TokenizerWrapper(ByteTokenizer()))
    # No special tokens: test the multi-token byte marker path.
    hf = SimpleNamespace(convert_tokens_to_ids=lambda _: None)
    lark = build_native_grammar(
        [ToolSpec("weather", SCHEMA)], fmt, hf, only="weather", parallel=True
    )
    grammar = LLMatcher.grammar_from_lark(lark)
    assert not LLMatcher.validate_grammar(grammar)
    matcher = LLMatcher(tok, grammar)
    assert not matcher.get_error()

    wires = {
        "deepseek": CASES[0][1],
        "deepseek_v32": CASES[1][1],
        "deepseek_v4": CASES[2][1],
        "hermes": CASES[3][1],
        "llama3_json": JSON,
        "llama3_pythonic": CASES[5][1],
        "mistral": CASES[6][1],
        "glm47": CASES[8][1],
        "kimi_k2": CASES[9][1],
        "harmony": CASES[10][1],
    }
    for byte in wires[fmt.name].encode():
        assert matcher.consume_token(byte), (fmt.name, matcher.get_error())
    assert matcher.is_accepting()


def test_harmony_generation_header_suffix_all_boundaries():
    tok = SimpleNamespace(chat_template="<|ghissue|>")
    formats = tf.formats_for_tokenizer(tok)
    for head in ("to=weather<|channel|>analysis", "<|channel|>commentary to=weather"):
        text = head + '<|message|>{"city":"Taipei","days":3}<|ghissue|>'
        for cut in range(len(text) + 1):
            s = ToolCallStreamer(formats, tools=TOOLS)
            out = s.process_token(text[:cut]) + s.process_token(text[cut:]) + s.flush()
            assert not "".join(o.text for o in out)
            calls = [o.tool_call for o in out if o.tool_call]
            assert len(calls) == 1
            assert json.loads(calls[0].arguments) == ARGS


def test_fallback_bracket_prose_is_not_held_until_eos():
    s = ToolCallStreamer()
    out = s.process_token("[a normal note]")
    assert "".join(o.text for o in out) == "[a normal note]"


def test_whole_call_filters_undeclared_name():
    s = ToolCallStreamer((tf.PYTHONIC,), tools=TOOLS)
    out = s.process_token("[unregistered()]") + s.flush()
    assert not any(o.tool_call for o in out)


@pytest.mark.parametrize("fmt", [tf.DSML, tf.DSML_V4, tf.GLM])
def test_native_raw_string_schema_bounds_and_enum(fmt):
    from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

    from .test_structural_tag import ByteTokenizer

    tok = LLTokenizer(TokenizerWrapper(ByteTokenizer()))
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string", "enum": ["Taipei"]}},
        "required": ["city"],
    }
    lark = build_native_grammar(
        [ToolSpec("weather", schema)],
        fmt,
        SimpleNamespace(convert_tokens_to_ids=lambda _: None),
        only="weather",
        parallel=False,
    )
    grammar = LLMatcher.grammar_from_lark(lark)
    if fmt is tf.GLM:
        template = "<tool_call>weather<arg_key>city</arg_key><arg_value>{}</arg_value></tool_call>"
    else:
        template = (
            fmt.start
            + '<｜DSML｜invoke name="weather"><｜DSML｜parameter name="city" string="true">{}</｜DSML｜parameter></｜DSML｜invoke>'
            + fmt.end
        )
    m = LLMatcher(tok, grammar)
    assert all(m.consume_token(b) for b in template.format("Taipei").encode())
    assert m.is_accepting()
    m = LLMatcher(tok, grammar)
    assert not all(m.consume_token(b) for b in template.format("Tokyo").encode())


def test_pythonic_nested_json_literals_and_python_literals():
    out = tf.PYTHONIC.parse(
        '[weather(data={"ok":true,"nil":null}, enabled=False)]', None
    )
    assert json.loads(out[0]["arguments"]) == {
        "data": {"ok": True, "nil": None},
        "enabled": False,
    }


@pytest.mark.parametrize(
    "fmt", [tf.DSML, tf.DSML_V4, tf.GLM, tf.KIMI, tf.MISTRAL, tf.HARMONY, tf.PYTHONIC]
)
def test_compile_forced_named_native_and_rollback(monkeypatch, fmt):
    from llguidance import LLTokenizer, TokenizerWrapper

    from yunshu_engine import tool_call_grammar as tcg

    from .test_structural_tag import ByteTokenizer

    tok = LLTokenizer(TokenizerWrapper(ByteTokenizer()))
    monkeypatch.setattr(tcg, "llg_tokenizer", lambda *_: tok)
    monkeypatch.setattr(tf, "native_format", lambda _: fmt)
    hf = SimpleNamespace(chat_template=fmt.start, convert_tokens_to_ids=lambda _: None)
    grammar = tcg.compile_tool_grammar(
        TOOLS,
        hf,
        257,
        tool_choice={"type": "function", "function": {"name": "weather"}},
        parallel=False,
    )
    assert grammar is not None and grammar.forced
    guide = grammar.guide()
    cp = guide.checkpoint()
    guide.feed(ord(" "))
    guide.restore(cp)
    assert guide.consumed == 0
    # Forced native grammars have no single-token trigger assumption.
    assert grammar.start_id == -1


def test_malformed_dsml_parameters_are_not_silently_empty():
    with pytest.raises(ValueError):
        tf.DSML.parse(
            '<｜DSML｜invoke name="weather"><｜DSML｜parameter name="city">lost</｜DSML｜parameter></｜DSML｜invoke>',
            None,
        )


def test_multiline_mistral_json_and_trailing_prose():
    body = json.dumps([{"name": "weather", "arguments": ARGS}], indent=2)
    text = "[TOOL_CALLS]" + body + "done"
    s = ToolCallStreamer((tf.MISTRAL,), tools=TOOLS)
    out = []
    for ch in text:
        out.extend(s.process_token(ch))
    out.extend(s.flush())
    assert "".join(o.text for o in out) == "done"
    calls = [o.tool_call for o in out if o.tool_call]
    assert len(calls) == 1 and json.loads(calls[0].arguments) == ARGS


def test_json_end_marker_inside_argument_is_data():
    args = {"city": 'literal </tool_call> "quoted"', "days": 3}
    text = (
        "<tool_call>"
        + json.dumps({"name": "weather", "arguments": args})
        + "</tool_call>"
    )
    s = ToolCallStreamer((tf.HERMES,), tools=TOOLS)
    out = []
    for ch in text:
        out.extend(s.process_token(ch))
    out.extend(s.flush())
    assert not "".join(o.text for o in out)
    assert json.loads(next(o.tool_call.arguments for o in out if o.tool_call)) == args
