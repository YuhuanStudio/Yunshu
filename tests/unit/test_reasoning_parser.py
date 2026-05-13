"""Tests for reasoning parser factory."""
import pytest


class TestQwenReasoningParser:
    def test_basic_think_tags(self):
        from yunshu_engine.reasoning_parser import QwenReasoningParser
        parser = QwenReasoningParser()
        out = parser.parse("<think/>step 1: analyze\nstep 2: conclude</think/>The answer is 42")
        assert out.reasoning == "step 1: analyze\nstep 2: conclude"
        assert out.content == "The answer is 42"

    def test_no_thinking(self):
        from yunshu_engine.reasoning_parser import QwenReasoningParser
        parser = QwenReasoningParser()
        out = parser.parse("Just a direct answer")
        assert out.reasoning is None
        assert out.content == "Just a direct answer"

    def test_empty_thinking(self):
        from yunshu_engine.reasoning_parser import QwenReasoningParser
        parser = QwenReasoningParser()
        out = parser.parse("<think/></think/>Direct answer")
        assert out.reasoning == ""
        assert out.content == "Direct answer"

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import QwenReasoningParser
        assert QwenReasoningParser().family_name() == "qwen"


class TestDeepSeekReasoningParser:
    def test_whitespace_handling(self):
        from yunshu_engine.reasoning_parser import DeepSeekReasoningParser
        parser = DeepSeekReasoningParser()
        out = parser.parse("<think/>  some reasoning  </think/>  the answer  ")
        assert out.reasoning == "some reasoning"
        assert out.content == "the answer"

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import DeepSeekReasoningParser
        assert DeepSeekReasoningParser().family_name() == "deepseek"


class TestGemmaReasoningParser:
    def test_gemma_tags(self):
        from yunshu_engine.reasoning_parser import GemmaReasoningParser
        parser = GemmaReasoningParser()
        out = parser.parse("<start_think/>reasoning here</end_think/>final answer")
        assert out.reasoning == "reasoning here"
        assert out.content == "final answer"

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import GemmaReasoningParser
        assert GemmaReasoningParser().family_name() == "gemma"


class TestHarmonyReasoningParser:
    def test_harmony_markers(self):
        from yunshu_engine.reasoning_parser import HarmonyReasoningParser
        parser = HarmonyReasoningParser()
        out = parser.parse("[REASONING]chain of thought[/REASONING]The result is 5")
        assert out.reasoning == "chain of thought"
        assert out.content == "The result is 5"

    def test_case_insensitive(self):
        from yunshu_engine.reasoning_parser import HarmonyReasoningParser
        parser = HarmonyReasoningParser()
        out = parser.parse("[reasoning]thought[/REASONING]answer")
        assert out.reasoning == "thought"

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import HarmonyReasoningParser
        assert HarmonyReasoningParser().family_name() == "harmony"


class TestGetReasoningParser:
    def test_qwen_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser
        parser = get_reasoning_parser("Qwen3-72B-Instruct")
        assert parser.family_name() == "qwen"

    def test_deepseek_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser
        parser = get_reasoning_parser("deepseek-r1-671b")
        assert parser.family_name() == "deepseek"

    def test_gemma_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser
        parser = get_reasoning_parser("gemma-4-27b")
        assert parser.family_name() == "gemma"

    def test_harmony_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser
        parser = get_reasoning_parser("harmony-4o")
        assert parser.family_name() == "harmony"

    def test_unknown_falls_back(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser
        parser = get_reasoning_parser("unknown-model-xyz")
        assert parser.family_name() == "generic"

    def test_none_falls_back(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser
        parser = get_reasoning_parser(None)
        assert parser.family_name() == "generic"


class TestExtractThinkingWithModelName:
    def test_extract_thinking_with_model_name(self):
        from yunshu_gateway.streaming import extract_thinking
        text = "<think/>step by step</think/>answer is 42"
        thinking, content = extract_thinking(text, model_name="Qwen3-72B")
        assert thinking == "step by step"
        assert content == "answer is 42"

    def test_extract_thinking_gemma_without_model(self):
        from yunshu_gateway.streaming import extract_thinking
        text = "<start_think/>reasoning</end_think/>answer"
        thinking, content = extract_thinking(text)
        assert thinking == "reasoning"
        assert content == "answer"

    def test_extract_thinking_harmony_without_model(self):
        from yunshu_gateway.streaming import extract_thinking
        text = "[REASONING]thought process[/REASONING]final output"
        thinking, content = extract_thinking(text)
        assert thinking == "thought process"
        assert content == "final output"

    def test_extract_thinking_normal(self):
        from yunshu_gateway.streaming import extract_thinking
        text = "<think/>reasoning</think/>answer"
        thinking, content = extract_thinking(text)
        assert thinking == "reasoning"
        assert content == "answer"
