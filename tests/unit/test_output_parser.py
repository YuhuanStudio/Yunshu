"""Tests for Output Parser Factory — model-specific output extraction."""

from yunshu_engine.output_parser import (
    CohereOutputParser,
    DeepSeekOutputParser,
    GemmaOutputParser,
    GenericOutputParser,
    GLMOutputParser,
    HarmonyOutputParser,
    InternVLOutputParser,
    LLamaOutputParser,
    MistralOutputParser,
    ParsedOutput,
    PhiOutputParser,
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
        result = parser.parse('Some text ✿FUNCTION✿ {"name":"foo"} ✿ end')
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


class TestMistralOutputParser:
    def test_strips_think(self):
        parser = MistralOutputParser()
        result = parser.parse("[THINK]I reasoned[/THINK]The answer is 42")
        assert result.reasoning == "I reasoned"
        assert result.content == "The answer is 42"

    def test_strips_tool_calls(self):
        parser = MistralOutputParser()
        result = parser.parse('Some text [TOOL_CALLS]\n{"name": "foo"}')
        assert "[TOOL_CALLS]" not in result.content
        assert result.tool_call_text is not None

    def test_no_markers(self):
        parser = MistralOutputParser()
        result = parser.parse("Just text")
        assert result.content == "Just text"
        assert result.reasoning is None

    def test_family_name(self):
        assert MistralOutputParser().family_name() == "mistral"


class TestPhiOutputParser:
    def test_strips_thinking(self):
        parser = PhiOutputParser()
        result = parser.parse("<think/>I reasoned</think/>The answer is 42")
        assert result.reasoning == "I reasoned"
        assert result.content == "The answer is 42"

    def test_strips_tool_markers(self):
        parser = PhiOutputParser()
        result = parser.parse('<|tool_calls|>[{"name": "f"}]<|/tool_calls|>Done')
        assert result.tool_call_text is not None
        assert "<|tool_calls|>" not in result.content

    def test_no_markers(self):
        parser = PhiOutputParser()
        result = parser.parse("Just text")
        assert result.content == "Just text"
        assert result.reasoning is None

    def test_family_name(self):
        assert PhiOutputParser().family_name() == "phi"


class TestCohereOutputParser:
    def test_strips_thinking(self):
        parser = CohereOutputParser()
        result = parser.parse(
            "<|START_THINKING|>I reasoned<|END_THINKING|>The answer is 42"
        )
        assert result.reasoning == "I reasoned"
        assert result.content == "The answer is 42"

    def test_strips_action_markers(self):
        parser = CohereOutputParser()
        result = parser.parse('Text <|START_ACTION|>{"name":"foo"}<|END_ACTION|> more')
        assert "<|START_ACTION|>" not in result.content
        assert result.tool_call_text is not None

    def test_no_markers(self):
        parser = CohereOutputParser()
        result = parser.parse("Just text")
        assert result.content == "Just text"
        assert result.reasoning is None

    def test_family_name(self):
        assert CohereOutputParser().family_name() == "cohere"


class TestLLamaOutputParser:
    def test_strips_thinking(self):
        parser = LLamaOutputParser()
        result = parser.parse("<think/>Hmm</think/>Answer")
        assert result.reasoning == "Hmm"
        assert result.content == "Answer"

    def test_strips_python_tag(self):
        parser = LLamaOutputParser()
        result = parser.parse('Some text <|python_tag|>{"name": "foo"}')
        assert "<|python_tag|>" not in result.content
        assert result.tool_call_text is not None

    def test_no_markers(self):
        parser = LLamaOutputParser()
        result = parser.parse("Just text")
        assert result.content == "Just text"

    def test_closing_python_tag_not_overcaptured(self):
        """the closing tag regex was malformed (missing the second |),
        so the close never matched and the capture ran to end-of-string,
        over-capturing trailing text after a real </python_tag|>."""
        parser = LLamaOutputParser()
        result = parser.parse(
            'pre <|python_tag|>{"name": "foo"}<|/python_tag|> POST-TEXT'
        )
        # The tool body is captured but the trailing visible text is NOT swallowed.
        assert result.tool_call_text is not None
        assert "POST-TEXT" not in (result.tool_call_text or "")

    def test_family_name(self):
        assert LLamaOutputParser().family_name() == "llama"


class TestInternVLOutputParser:
    def test_strips_thinking(self):
        parser = InternVLOutputParser()
        result = parser.parse("<think/>Reasoning</think/>Content")
        assert result.reasoning == "Reasoning"
        assert result.content == "Content"

    def test_strips_img_context(self):
        parser = InternVLOutputParser()
        result = parser.parse("Before<IMG_CONTEXT>abc</IMG_CONTEXT>After")
        assert "<IMG_CONTEXT>" not in result.content
        assert "Before" in result.content
        assert "After" in result.content

    def test_no_markers(self):
        parser = InternVLOutputParser()
        result = parser.parse("Plain content")
        assert result.content == "Plain content"

    def test_family_name(self):
        assert InternVLOutputParser().family_name() == "internvl"


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

    def test_mistral(self):
        assert isinstance(get_output_parser("mistral-7b"), MistralOutputParser)

    def test_codestral(self):
        assert isinstance(get_output_parser("codestral-22b"), MistralOutputParser)

    def test_phi(self):
        assert isinstance(get_output_parser("phi-3.5-mini"), PhiOutputParser)

    def test_cohere(self):
        assert isinstance(get_output_parser("command-r-plus"), CohereOutputParser)

    def test_llama(self):
        assert isinstance(get_output_parser("llama-3.1-70b"), LLamaOutputParser)

    def test_internvl(self):
        assert isinstance(get_output_parser("internvl-2.5"), InternVLOutputParser)

    def test_unknown_generic(self):
        assert isinstance(get_output_parser("falcon-180b"), GenericOutputParser)

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
