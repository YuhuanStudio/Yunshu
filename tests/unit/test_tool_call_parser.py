"""Tests for tool call parser factory."""
import json
import pytest


class TestHermesToolCallParser:
    def test_basic_hermes(self):
        from yunshu_engine.tool_call_parser import HermesToolCallParser
        parser = HermesToolCallParser()
        calls = parser.parse('<tool_call/>{"name": "get_weather", "arguments": {"city": "Taipei"}}</tool_call/>')
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        args = json.loads(calls[0].arguments)
        assert args["city"] == "Taipei"

    def test_multiple_hermes(self):
        from yunshu_engine.tool_call_parser import HermesToolCallParser
        parser = HermesToolCallParser()
        text = (
            '<tool_call/>{"name": "f1", "arguments": {} }</tool_call/>'
            '<tool_call/>{"name": "f2", "arguments": {} }</tool_call/>'
        )
        calls = parser.parse(text)
        assert len(calls) == 2

    def test_no_tool_call(self):
        from yunshu_engine.tool_call_parser import HermesToolCallParser
        parser = HermesToolCallParser()
        calls = parser.parse("just normal text")
        assert len(calls) == 0


class TestQwenXMLToolCallParser:
    def test_qwen_xml(self):
        from yunshu_engine.tool_call_parser import QwenXMLToolCallParser
        parser = QwenXMLToolCallParser()
        calls = parser.parse('<function=get_weather>{"city": "Taipei"}</function>')
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_qwen_xml_params(self):
        from yunshu_engine.tool_call_parser import QwenXMLToolCallParser
        parser = QwenXMLToolCallParser()
        calls = parser.parse('<function=search><parameter=query>AI</parameter></function>')
        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["query"] == "AI"


class TestDeepSeekToolCallParser:
    def test_deepseek_markers(self):
        from yunshu_engine.tool_call_parser import DeepSeekToolCallParser
        parser = DeepSeekToolCallParser()
        calls = parser.parse('✿FUNCTION✿ {"name": "calc", "arguments": {"x": 1}} ✿')
        assert len(calls) == 1
        assert calls[0].name == "calc"


class TestMistralToolCallParser:
    def test_mistral_format(self):
        from yunshu_engine.tool_call_parser import MistralToolCallParser
        parser = MistralToolCallParser()
        calls = parser.parse('{"function": {"name": "search", "arguments": {"q": "test"}}}')
        assert len(calls) == 1
        assert calls[0].name == "search"


class TestChatMLToolCallParser:
    def test_chatml_format(self):
        from yunshu_engine.tool_call_parser import ChatMLToolCallParser
        parser = ChatMLToolCallParser()
        calls = parser.parse('[TOOL_CALLS] [{"name": "f1", "arguments": {}}]')
        assert len(calls) == 1
        assert calls[0].name == "f1"


class TestAnthropicToolCallParser:
    def test_anthropic_xml(self):
        from yunshu_engine.tool_call_parser import AnthropicToolCallParser
        parser = AnthropicToolCallParser()
        calls = parser.parse(
            '<tool_use><name>get_weather</name><input>{"city": "Tokyo"}</input></tool_use>'
        )
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        args = json.loads(calls[0].arguments)
        assert args["city"] == "Tokyo"


class TestGeminiToolCallParser:
    def test_gemini_format(self):
        from yunshu_engine.tool_call_parser import GeminiToolCallParser
        parser = GeminiToolCallParser()
        calls = parser.parse('"functionCall": {"name": "search", "args": {"q": "test"}}')
        assert len(calls) == 1
        assert calls[0].name == "search"


class TestParseToolCallsFactory:
    def test_qwen_model_detection(self):
        from yunshu_engine.tool_call_parser import parse_tool_calls
        text = '<function=calc>{"x": 1}</function>'
        calls = parse_tool_calls(text, model_name="Qwen3-72B")
        assert len(calls) == 1
        assert calls[0].name == "calc"

    def test_deepseek_model_detection(self):
        from yunshu_engine.tool_call_parser import parse_tool_calls
        text = '✿FUNCTION✿ {"name": "calc", "arguments": {"x": 1}} ✿'
        calls = parse_tool_calls(text, model_name="deepseek-v3")
        assert len(calls) == 1
        assert calls[0].name == "calc"

    def test_fallback_tries_all(self):
        from yunshu_engine.tool_call_parser import parse_tool_calls
        text = '<tool_call/>{"name": "test", "arguments": {}}</tool_call/>'
        calls = parse_tool_calls(text, model_name="unknown-model")
        assert len(calls) == 1
        assert calls[0].name == "test"

    def test_no_tool_calls(self):
        from yunshu_engine.tool_call_parser import parse_tool_calls
        text = "Just a regular response without tools"
        calls = parse_tool_calls(text, model_name="Qwen3")
        assert len(calls) == 0


class TestCleanToolMarkup:
    def test_clean_hermes(self):
        from yunshu_engine.tool_call_parser import clean_tool_markup
        text = 'before <tool_call/>{"name": "f"}</tool_call/> after'
        result = clean_tool_markup(text)
        assert "tool_call" not in result
        assert "before" in result
        assert "after" in result

    def test_clean_function_xml(self):
        from yunshu_engine.tool_call_parser import clean_tool_markup
        text = 'before <function=f>{"x":1}</function> after'
        result = clean_tool_markup(text)
        assert "function" not in result

    def test_clean_deepseek(self):
        from yunshu_engine.tool_call_parser import clean_tool_markup
        text = 'before ✿FUNCTION✿ {"name":"f"} ✿ after'
        result = clean_tool_markup(text)
        assert "FUNCTION" not in result
