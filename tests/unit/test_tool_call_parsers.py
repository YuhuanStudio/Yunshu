"""Tests for C15: Tool Call Parsers — multi-model format support."""

import json

from yunshu_engine.tool_call_parsers import (
    ToolCallFormat,
    ToolCallParser,
    ToolCallResult,
    detect_tool_call_format,
    parse_deepseek_tool_calls,
    parse_generic_tool_calls,
    parse_glm_tool_calls,
    parse_llama_tool_calls,
    parse_mistral_tool_calls,
    parse_qwen_tool_calls,
)


class TestDetectFormat:
    def test_qwen(self):
        assert detect_tool_call_format("Qwen2.5-7B-Instruct") == ToolCallFormat.QWEN

    def test_qwq(self):
        assert detect_tool_call_format("QwQ-32B") == ToolCallFormat.QWEN

    def test_deepseek(self):
        assert detect_tool_call_format("deepseek-v3") == ToolCallFormat.DEEPSEEK

    def test_glm(self):
        assert detect_tool_call_format("glm-4") == ToolCallFormat.GLM

    def test_chatglm(self):
        assert detect_tool_call_format("chatglm3-6b") == ToolCallFormat.GLM

    def test_llama(self):
        assert detect_tool_call_format("llama-3.1-8b") == ToolCallFormat.LLAMA

    def test_mistral(self):
        assert detect_tool_call_format("mistral-7b") == ToolCallFormat.MISTRAL

    def test_mixtral(self):
        assert detect_tool_call_format("mixtral-8x7b") == ToolCallFormat.MISTRAL

    def test_unknown(self):
        assert detect_tool_call_format("unknown-model") == ToolCallFormat.GENERIC


class TestParseQwenToolCalls:
    def test_single_call(self):
        text = '<tool_call\n>{"name": "get_weather", "arguments": {"city": "Taipei"}}\n</tool_call\n>'
        calls = parse_qwen_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        args = json.loads(calls[0].arguments)
        assert args["city"] == "Taipei"

    def test_multiple_calls(self):
        text = (
            '<tool_call\n>{"name": "get_weather", "arguments": {"city": "Taipei"}}\n</tool_call\n>'
            '<tool_call\n>{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call\n>'
        )
        calls = parse_qwen_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "get_weather"
        assert calls[1].name == "get_time"

    def test_no_calls(self):
        text = "Just regular text without any tool calls."
        calls = parse_qwen_tool_calls(text)
        assert len(calls) == 0

    def test_empty_tag_before_real_call_no_duplicate(self):
        """an empty <tool_call></tool_call> before a real call made the
        unbounded text.find('{') grab the NEXT call's JSON → a duplicate. The
        brace search is now bounded to each tag's closing marker."""
        text = '<tool_call></tool_call><tool_call>{"name": "g", "arguments": {}}</tool_call>'
        calls = parse_qwen_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "g"

    def test_malformed_json(self):
        text = '<tool_call\n>{"name": "test", "arguments": {broken}\n</tool_call\n>'
        calls = parse_qwen_tool_calls(text)
        # Unbalanced braces cannot be extracted — returns empty
        assert len(calls) == 0

    def test_malformed_json_with_name_fallback(self):
        """If the JSON object is brace-balanced but invalid, fallback extracts name."""
        text = '<tool_call\n>{"name": "test", "arguments": {broken}}\n</tool_call\n>'
        calls = parse_qwen_tool_calls(text)
        # Brace-balanced but JSON is invalid → fallback regex extracts name
        assert len(calls) == 1
        assert calls[0].name == "test"


class TestParseDeepseekToolCalls:
    def test_v3_format(self):
        # Real DeepSeek-V3 wire format: name between markers, args follow sep with
        # no fence (verified vs reference/llama.cpp/tests/test-chat.cpp:3286).
        text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
            '<｜tool▁sep｜>{"city": "Taipei"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        )
        calls = parse_deepseek_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "Taipei"}

    def test_v3_multiple_calls(self):
        text = (
            '<｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"tz": "UTC"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Tokyo"}<｜tool▁call▁end｜>'
        )
        calls = parse_deepseek_tool_calls(text)
        assert [c.name for c in calls] == ["get_time", "get_weather"]

    def test_r1_format_with_fence(self):
        # Older DeepSeek-R1: literal "function" before sep, name after, ```json fence.
        text = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Taipei"}\n```<｜tool▁call▁end｜>'
        )
        calls = parse_deepseek_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "Taipei"}

    def test_no_calls(self):
        calls = parse_deepseek_tool_calls("normal text")
        assert len(calls) == 0


class TestParseGlmToolCalls:
    def test_glm_format(self):
        text = '<|tool_call_block_begin|>get_weather\n```json\n{"city": "Taipei"}\n```'
        calls = parse_glm_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_glm_47_flash_arg_kv(self):
        # GLM-4.7-Flash: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        text = (
            "<tool_call>special_function"
            "<arg_key>arg1</arg_key><arg_value>1</arg_value>"
            '<arg_key>city</arg_key><arg_value>"Taipei"</arg_value></tool_call>'
        )
        calls = parse_glm_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "special_function"
        assert json.loads(calls[0].arguments) == {"arg1": 1, "city": "Taipei"}

    def test_glm_46_arg_kv_newlines(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>"
        )
        calls = parse_glm_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"arg1": 1}

    def test_no_calls(self):
        calls = parse_glm_tool_calls("normal text")
        assert len(calls) == 0


class TestParseLlamaToolCalls:
    def test_llama_format(self):
        text = '[TOOL_CALL] get_weather {"city": "Taipei"} [/TOOL_CALL]'
        calls = parse_llama_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_no_calls(self):
        calls = parse_llama_tool_calls("normal text")
        assert len(calls) == 0


class TestParseMistralToolCalls:
    def test_mistral_format(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Taipei"}}] [/TOOL_CALLS]'
        calls = parse_mistral_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_multiple(self):
        text = '[TOOL_CALLS] [{"name": "f1", "arguments": {}}, {"name": "f2", "arguments": {}}] [/TOOL_CALLS]'
        calls = parse_mistral_tool_calls(text)
        assert len(calls) == 2

    def test_no_calls(self):
        calls = parse_mistral_tool_calls("normal text")
        assert len(calls) == 0


class TestParseGenericToolCalls:
    def test_generic_json(self):
        text = '{"name": "get_weather", "arguments": {"city": "Taipei"}}'
        calls = parse_generic_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_no_calls(self):
        calls = parse_generic_tool_calls("normal text without json")
        assert len(calls) == 0


class TestToolCallResult:
    def test_to_dict(self):
        r = ToolCallResult(id="call_0", name="test", arguments='{"a": 1}')
        d = r.to_dict()
        assert d["id"] == "call_0"
        assert d["name"] == "test"
        assert d["arguments"] == '{"a": 1}'


class TestToolCallParser:
    def test_auto_detect_qwen(self):
        parser = ToolCallParser(model_name="Qwen2.5-7B")
        assert parser.format == ToolCallFormat.QWEN
        assert parser.format_name == "QWEN"

    def test_explicit_format(self):
        parser = ToolCallParser(format=ToolCallFormat.LLAMA)
        assert parser.format == ToolCallFormat.LLAMA

    def test_parse_qwen(self):
        parser = ToolCallParser(model_name="qwen")
        text = '<tool_call\n>{"name": "search", "arguments": {"q": "test"}}\n</tool_call\n>'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_parse_empty(self):
        parser = ToolCallParser()
        calls = parser.parse("")
        assert calls == []

    def test_stats(self):
        parser = ToolCallParser(model_name="qwen")
        text = '<tool_call\n>{"name": "search", "arguments": {}}\n</tool_call\n>'
        parser.parse(text)
        parser.parse("no calls here")
        stats = parser.get_stats()
        assert stats["parse_count"] == 2
        assert stats["call_count"] == 1
        assert stats["format"] == "QWEN"
        assert stats["avg_calls_per_parse"] == 0.5

    def test_register_custom_parser(self):
        def custom_parser(text):
            if "CUSTOM:" in text:
                return [ToolCallResult(id="c0", name="custom", arguments="{}")]
            return []

        ToolCallParser.register_parser(ToolCallFormat.GENERIC, custom_parser)
        parser = ToolCallParser(format=ToolCallFormat.GENERIC)
        calls = parser.parse("CUSTOM: something")
        assert len(calls) == 1
        assert calls[0].name == "custom"

        # Restore default
        ToolCallParser.register_parser(ToolCallFormat.GENERIC, parse_generic_tool_calls)
