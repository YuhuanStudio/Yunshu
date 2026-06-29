"""Tests for reasoning parser factory."""


class TestQwenReasoningParser:
    def test_basic_think_tags(self):
        from yunshu_engine.reasoning_parser import QwenReasoningParser

        parser = QwenReasoningParser()
        out = parser.parse(
            "<think/>step 1: analyze\nstep 2: conclude</think/>The answer is 42"
        )
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

    def test_stray_unmatched_close_tag_stripped_from_content(self):
        # closing-only mode (templated <think>); a SECOND unmatched </think>
        # emitted inside the answer leaked into visible content verbatim.
        from yunshu_engine.reasoning_parser import QwenReasoningParser

        out = QwenReasoningParser().parse("r1</think>mid</think>tail")
        assert out.reasoning == "r1"
        assert out.content == "midtail"  # the stray </think> is gone
        assert "</think>" not in out.content

    def test_closing_only_unchanged(self):
        from yunshu_engine.reasoning_parser import QwenReasoningParser

        out = QwenReasoningParser().parse("thinking</think>the answer")
        assert out.reasoning == "thinking"
        assert out.content == "the answer"


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

    def test_glm_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("glm-4-9b")
        assert parser.family_name() == "glm"

    def test_mistral_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("mistral-7b-instruct")
        assert parser.family_name() == "mistral"

    def test_codestral_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("codestral-22b")
        assert parser.family_name() == "mistral"

    def test_phi_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("phi-3.5-mini")
        assert parser.family_name() == "phi"

    def test_cohere_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("command-r-plus")
        assert parser.family_name() == "cohere"

    def test_llama_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("llama-3.1-70b")
        assert parser.family_name() == "llama"

    def test_internvl_detection(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("internvl-2.5-8b")
        assert parser.family_name() == "internvl"

    def test_unknown_falls_back(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser("unknown-model-xyz")
        assert parser.family_name() == "generic"

    def test_none_falls_back(self):
        from yunshu_engine.reasoning_parser import get_reasoning_parser

        parser = get_reasoning_parser(None)
        assert parser.family_name() == "generic"


class TestMistralReasoningParser:
    def test_basic_think_tags(self):
        from yunshu_engine.reasoning_parser import MistralReasoningParser

        parser = MistralReasoningParser()
        out = parser.parse(
            "[THINK]step 1: analyze\nstep 2: conclude[/THINK]The answer is 42"
        )
        assert out.reasoning == "step 1: analyze\nstep 2: conclude"
        assert out.content == "The answer is 42"

    def test_no_thinking(self):
        from yunshu_engine.reasoning_parser import MistralReasoningParser

        parser = MistralReasoningParser()
        out = parser.parse("Just a direct answer")
        assert out.reasoning is None
        assert out.content == "Just a direct answer"

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import MistralReasoningParser

        assert MistralReasoningParser().family_name() == "mistral"


class TestPhiReasoningParser:
    def test_basic_think_tags(self):
        from yunshu_engine.reasoning_parser import PhiReasoningParser

        parser = PhiReasoningParser()
        out = parser.parse("<think/>reasoning here</think/>answer")
        assert out.reasoning == "reasoning here"
        assert out.content == "answer"

    def test_no_thinking(self):
        from yunshu_engine.reasoning_parser import PhiReasoningParser

        parser = PhiReasoningParser()
        out = parser.parse("Just a direct answer")
        assert out.reasoning is None

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import PhiReasoningParser

        assert PhiReasoningParser().family_name() == "phi"


class TestCohereReasoningParser:
    def test_basic_thinking_tags(self):
        from yunshu_engine.reasoning_parser import CohereReasoningParser

        parser = CohereReasoningParser()
        out = parser.parse(
            "<|START_THINKING|>I thought about this<|END_THINKING|>The answer is 42"
        )
        assert out.reasoning == "I thought about this"
        assert out.content == "The answer is 42"

    def test_no_thinking(self):
        from yunshu_engine.reasoning_parser import CohereReasoningParser

        parser = CohereReasoningParser()
        out = parser.parse("Just a direct answer")
        assert out.reasoning is None

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import CohereReasoningParser

        assert CohereReasoningParser().family_name() == "cohere"


class TestLLamaReasoningParser:
    def test_basic_think_tags(self):
        from yunshu_engine.reasoning_parser import LLamaReasoningParser

        parser = LLamaReasoningParser()
        out = parser.parse("<think/>reasoning here</think/>answer")
        assert out.reasoning == "reasoning here"
        assert out.content == "answer"

    def test_no_thinking(self):
        from yunshu_engine.reasoning_parser import LLamaReasoningParser

        parser = LLamaReasoningParser()
        out = parser.parse("Just a direct answer")
        assert out.reasoning is None

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import LLamaReasoningParser

        assert LLamaReasoningParser().family_name() == "llama"


class TestInternVLReasoningParser:
    def test_basic_think_tags(self):
        from yunshu_engine.reasoning_parser import InternVLReasoningParser

        parser = InternVLReasoningParser()
        out = parser.parse("<think/>reasoning here</think/>answer")
        assert out.reasoning == "reasoning here"
        assert out.content == "answer"

    def test_no_thinking(self):
        from yunshu_engine.reasoning_parser import InternVLReasoningParser

        parser = InternVLReasoningParser()
        out = parser.parse("Just a direct answer")
        assert out.reasoning is None

    def test_family_name(self):
        from yunshu_engine.reasoning_parser import InternVLReasoningParser

        assert InternVLReasoningParser().family_name() == "internvl"


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


def test_closing_only_think_tag():
    """Qwen3/DeepSeek templates inject the opening <think>, so model output often has
    only </think>. The text before it must be parsed as reasoning, not leaked to content."""
    from yunshu_engine.reasoning_parser import get_reasoning_parser

    p = get_reasoning_parser("Qwen3.5-9B")
    out = p.parse("Step 1. Step 2.</think>Final answer.")
    assert out.reasoning and "Step 1" in out.reasoning
    assert out.content == "Final answer."
    assert "</think>" not in out.content
    assert out.reasoning_tokens > 0
