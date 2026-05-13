"""Tests for Output Parser Factory — model-specific output extraction."""
import pytest

from yunshu_engine.output_parser import (
    DeepSeekOutputParser,
    GemmaOutputParser,
    GLMOutputParser,
    GenericOutputParser,
    HarmonyOutputParser,
    OutputParser,
    ParsedOutput,
    QwenOutputParser,
    get_output_parser,
    parse_output,
    register_output_parser,
)


class TestParsedOutput:
    def test_defaults(self):
        po = ParsedOutput(content="hello")
        assert po.content == "hello"
        assert po.reasoning is None
        assert po.tool_call_text is None


class TestDeepSeekOutputParser:
    def test_strips_thinking(self):
        parser = DeepSeekOutputParser()
        result = parser.parse("<think/>I reasoned</think/>The answer is 42")
        assert result.reasoning == "I reasoned"
        assert result.content == "The answer is 42"

    def test_strips_function_markers(self):
        parser = DeepSeekOutputParser()
        result = parser.parse("Some text ✿FUNCTION✿ {\"name\":\"foo\"} ✿ end")
        assert "✿FUNCTION✿" not in result.content
        assert result.tool_call_text is not None
        assert "FUNCTION" in result.tool_call_text

    def test_no_markers(self):
        parser = DeepSeekOutputParser()
        result = parser.parse("Just text")
        assert result.content == "Just text"
        assert result.reasoning is None

    def test_family_name(self):
        assert DeepSeekOutputParser().family_name() == "deepseek"


class TestQwenOutputParser:
    def test_strips_thinking(self):
        parser = QwenOutputParser()
        result = parser.parse("<think/>Thinking...</think/>Answer")
        assert result.reasoning == "Thinking..."
        assert result.content == "Answer"

    def test_strips_tool_xml(self):
        parser = QwenOutputParser()
        result = parser.parse('<tool_call/>{"name":"f"}</tool_call/>Done')
        assert "<tool_call" not in result.content
        assert result.tool_call_text is not None

    def test_family_name(self):
        assert QwenOutputParser().family_name() == "qwen"


class TestGemmaOutputParser:
    def test_strips_start_think(self):
        parser = GemmaOutputParser()
        result = parser.parse("<start_think/>Reasoning</end_think/>Content")
        assert result.reasoning == "Reasoning"
        assert result.content == "Content"

    def test_no_markers(self):
        parser = GemmaOutputParser()
        result = parser.parse("Plain content")
        assert result.content == "Plain content"

    def test_family_name(self):
        assert GemmaOutputParser().family_name() == "gemma"


class TestHarmonyOutputParser:
    def test_strips_reasoning_markers(self):
        parser = HarmonyOutputParser()
        result = parser.parse("[REASONING]I thought[/REASONING]Answer")
        assert result.reasoning == "I thought"
        assert result.content == "Answer"

    def test_case_insensitive(self):
        parser = HarmonyOutputParser()
        result = parser.parse("[reasoning]test[/reasoning]Done")
        assert result.reasoning == "test"

    def test_family_name(self):
        assert HarmonyOutputParser().family_name() == "harmony"


class TestGLMOutputParser:
    def test_strips_observation(self):
        parser = GLMOutputParser()
        result = parser.parse("Before<observation/>data</observation/>After")
        assert "observation" not in result.content
        assert "Before" in result.content
        assert "After" in result.content

    def test_family_name(self):
        assert GLMOutputParser().family_name() == "glm"


class TestGenericOutputParser:
    def test_extracts_think(self):
        parser = GenericOutputParser()
        result = parser.parse("<think/>Hmm</think/>Ok")
        assert result.reasoning == "Hmm"

    def test_passthrough(self):
        parser = GenericOutputParser()
        result = parser.parse("No markers")
        assert result.content == "No markers"


class TestGetOutputParser:
    def test_deepseek(self):
        assert isinstance(get_output_parser("deepseek-v3"), DeepSeekOutputParser)

    def test_qwen(self):
        assert isinstance(get_output_parser("qwen-2.5"), QwenOutputParser)

    def test_gemma(self):
        assert isinstance(get_output_parser("gemma-4"), GemmaOutputParser)

    def test_harmony(self):
        assert isinstance(get_output_parser("harmony-2"), HarmonyOutputParser)

    def test_glm(self):
        assert isinstance(get_output_parser("glm-4"), GLMOutputParser)

    def test_unknown_generic(self):
        assert isinstance(get_output_parser("llama-3"), GenericOutputParser)

    def test_none_generic(self):
        assert isinstance(get_output_parser(None), GenericOutputParser)


class TestParseOutput:
    def test_convenience(self):
        result = parse_output("<think/>Hmm</think/>Answer", "deepseek-v3")
        assert result.reasoning == "Hmm"
        assert result.content == "Answer"


class TestRegisterOutputParser:
    def test_register_custom(self):
        class CustomParser(GenericOutputParser):
            def family_name(self):
                return "custom"

        register_output_parser("custom", CustomParser)
        from yunshu_engine.output_parser import _REGISTRY
        assert "custom" in _REGISTRY
        _REGISTRY.pop("custom", None)
