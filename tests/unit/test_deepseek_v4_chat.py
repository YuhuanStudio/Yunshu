"""Official V4 prompt shape and agent/tool history, without loading weights."""

import copy
import json
from types import SimpleNamespace

from yunshu_engine.deepseek_v4_chat import TEMPLATE_MARKERS, install, render
from yunshu_engine.tool_format import formats_for_tokenizer, parse_tool_output


def test_official_chat_and_thinking_primers():
    messages = [
        dict(role="system", content="Be concise."),
        dict(role="user", content="Hello"),
    ]
    assert (
        render(messages, enable_thinking=False, add_generation_prompt=True)
        == "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Hello<｜Assistant｜></think>"
    )
    assert render(messages, enable_thinking=True, add_generation_prompt=True).endswith(
        "<｜Assistant｜><think>"
    )
    assert not render(messages, enable_thinking=False).endswith(
        "<｜Assistant｜></think>"
    )


def test_tool_history_preserves_reasoning_and_typed_arguments():
    messages = [
        dict(role="system", content="Use tools."),
        dict(role="user", content="Weather?"),
        dict(
            role="assistant",
            content="",
            reasoning_content="Need local weather.",
            tool_calls=[
                dict(
                    id="c1",
                    type="function",
                    function=dict(
                        name="weather", arguments={"city": "台北", "days": 2}
                    ),
                )
            ],
        ),
        dict(role="tool", tool_call_id="c1", content="Sunny"),
    ]
    before = copy.deepcopy(messages)
    tools = [
        dict(
            type="function",
            function=dict(name="weather", parameters=dict(type="object")),
        )
    ]
    text = render(
        messages, tools=tools, enable_thinking=True, add_generation_prompt=True
    )
    assert "Need local weather.</think>" in text
    assert (
        '<｜DSML｜parameter name="city" string="true">台北</｜DSML｜parameter>' in text
    )
    assert '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>' in text
    assert "<tool_result>Sunny</tool_result>" in text
    assert messages == before


def test_native_dsml_parser_and_span_removal():
    tokenizer = SimpleNamespace(chat_template=TEMPLATE_MARKERS)
    formats = formats_for_tokenizer(tokenizer)
    assert formats[0].name == "deepseek_v4"
    text = (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="weather">\n'
        '<｜DSML｜parameter name="city" string="true">台北</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
    calls, clean = parse_tool_output(text, formats)
    assert clean == ""
    assert calls[0]["name"] == "weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "台北", "days": 2}


def test_install_does_not_replace_published_template():
    tokenizer = SimpleNamespace(chat_template="published", _chat_template=None)
    assert not install(tokenizer)
    assert tokenizer.chat_template == "published"
    tokenizer = SimpleNamespace(chat_template=None, _chat_template=None)
    criteria = object()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(stopping_criteria=criteria))
    assert install(tokenizer, processor)
    assert tokenizer._chat_template is render
    assert processor.tokenizer is tokenizer
    assert tokenizer.stopping_criteria is criteria


def test_vlm_formatting_keeps_interleaved_reasoning():
    from yunshu_engine.vlm_engine import VLMEngine

    class Tokenizer:
        chat_template = TEMPLATE_MARKERS

        def apply_chat_template(self, messages, **kwargs):
            return render(messages, **kwargs)

    engine = VLMEngine("DeepSeek-V4-Flash-0731")
    engine._config = {"model_type": "deepseek_v4"}
    engine._tokenizer = Tokenizer()
    tools = [
        dict(
            type="function",
            function=dict(name="weather", parameters=dict(type="object")),
        )
    ]
    messages = [
        dict(role="developer", content="Use tools carefully."),
        dict(role="user", content="Weather?"),
        dict(
            role="assistant",
            content="",
            reasoning_content="Need local weather.",
            tool_calls=[
                dict(
                    id="c1",
                    type="function",
                    function=dict(name="weather", arguments='{"city":"台北"}'),
                )
            ],
        ),
        dict(role="tool", tool_call_id="c1", content="Sunny"),
    ]
    text = engine._format_prompt(
        messages, enable_thinking=True, template_extra={"tools": tools}
    )
    assert "Need local weather.</think>" in text
    assert "<tool_result>Sunny</tool_result>" in text
    assert "Use tools carefully." in text
