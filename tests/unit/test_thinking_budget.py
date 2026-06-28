"""Tests for thinking budget processor."""

from yunshu_engine.thinking_budget import (
    ThinkingBudgetConfig,
    ThinkingBudgetProcessor,
    detect_needs_think_prefix,
    parse_thinking_budget,
    resolve_think_close_pattern,
)


class TestThinkingBudgetProcessor:
    def test_no_budget_when_disabled(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(enabled=False))
        result = proc.process_token("reasoning")
        assert result['force_stop'] is False
        assert result['budget_exceeded'] is False

    def test_counts_thinking_tokens(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=10))
        for i in range(5):
            result = proc.process_token("reasoning")
            assert result['force_stop'] is False
            assert result['thinking_tokens_used'] == i + 1

    def test_budget_exceeded_forces_stop(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=3))
        proc.process_token("reasoning")
        proc.process_token("reasoning")
        result = proc.process_token("reasoning")
        assert result['force_stop'] is True
        assert result['budget_exceeded'] is True

    def test_normal_state_does_not_count(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=3))
        proc.process_token("reasoning")
        proc.process_token("reasoning")
        result = proc.process_token("normal")
        assert result['force_stop'] is False
        assert result['thinking_tokens_used'] == 2

    def test_budget_remaining(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=10))
        proc.process_token("reasoning")
        assert proc.budget_remaining == 9

    def test_reset(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=3))
        proc.process_token("reasoning")
        proc.reset()
        assert proc.thinking_tokens_used == 0
        assert proc.is_budget_exceeded is False

    def test_transition_from_normal_to_reasoning_accumulates_count(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=5))
        proc.process_token("reasoning")
        proc.process_token("reasoning")
        proc.process_token("normal")
        assert proc.thinking_tokens_used == 2
        # Re-entering reasoning continues accumulating total count
        # (budget applies across ALL thinking segments, not per-segment)
        result = proc.process_token("reasoning")
        assert result['thinking_tokens_used'] == 3

    def test_get_stats(self):
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))
        stats = proc.get_stats()
        assert stats['enabled'] is True
        assert stats['max_thinking_tokens'] == 100
        assert stats['budget_remaining'] == 100


class TestParseThinkingBudget:
    def test_explicit_budget(self):
        config = parse_thinking_budget({'thinking_budget': 4096})
        assert config is not None
        assert config.max_thinking_tokens == 4096
        assert config.enabled is True

    def test_reasoning_effort_low(self):
        config = parse_thinking_budget({'reasoning_effort': 'low'})
        assert config is not None
        assert config.max_thinking_tokens == 2048

    def test_reasoning_effort_medium(self):
        config = parse_thinking_budget({'reasoning_effort': 'medium'})
        assert config is not None
        assert config.max_thinking_tokens == 8192

    def test_reasoning_effort_high(self):
        config = parse_thinking_budget({'reasoning_effort': 'high'})
        assert config is not None
        assert config.max_thinking_tokens == 32768

    def test_no_budget_params(self):
        config = parse_thinking_budget({})
        assert config is None

    def test_budget_takes_priority_over_effort(self):
        config = parse_thinking_budget({
            'thinking_budget': 100,
            'reasoning_effort': 'high',
        })
        assert config.max_thinking_tokens == 100


class TestDetectNeedsThinkPrefix:
    def test_no_tokenizer_attrs_returns_false(self):
        class MockTok:
            pass
        assert detect_needs_think_prefix([1, 2, 3], MockTok()) is False

    def test_think_start_in_last_tokens(self):
        class MockTok:
            think_start_id = 42
            def convert_tokens_to_ids(self, t):
                return 42
        # think_start (42) is in the last 3 tokens, no think_end after
        assert detect_needs_think_prefix([1, 2, 42], MockTok()) is True

    def test_think_start_not_in_last_tokens(self):
        class MockTok:
            think_start_id = 42
        # think_start (42) is not in the last 3 tokens
        assert detect_needs_think_prefix([42, 1, 2, 3], MockTok()) is False

    def test_disabled_thinking_pattern(self):
        """<think/> followed immediately by </think/> = thinking disabled."""
        class MockTok:
            think_start_id = 42
            think_end_id = 99
            def convert_tokens_to_ids(self, t):
                return 42
        # think_start (42) then think_end (99) = disabled
        assert detect_needs_think_prefix([1, 42, 99], MockTok()) is False

    def test_empty_prompt(self):
        class MockTok:
            think_start_id = 42
        assert detect_needs_think_prefix([], MockTok()) is False

    def test_w1031_real_tokenizer_fallback_uses_bracketed_marker(self):
        # real HF tokenizers have NO think_start_id attr → the function falls
        # back to convert_tokens_to_ids("<think>"). The old slash typo "<think/>" mapped to
        # None on a real tokenizer (→ returns False), so the streaming pre-seed never fired
        # and the WHOLE chain-of-thought leaked into delta.content. The fallback must use
        # the bracketed marker. Every other test in this class hands the mock a
        # think_start_id attribute, which short-circuits and masks this exact path.
        THINK = 248068

        class RealishTok:           # no think_start_id attribute (like a real HF tokenizer)
            unk_token_id = None     # Qwen3.5 has no unk → must not be confused with a hit
            def convert_tokens_to_ids(self, t):
                return THINK if t == "<think>" else None   # "<think/>" → None

        # an OPEN <think> in the prompt tail → thinking enabled
        assert detect_needs_think_prefix([1, 2, THINK], RealishTok()) is True
        # no <think> in the tail → not thinking
        assert detect_needs_think_prefix([THINK, 1, 2, 3], RealishTok()) is False

    def test_w1031_no_slash_markers_in_code(self):
        import inspect

        from yunshu_engine import thinking_budget as tb
        code = "\n".join(ln.split("#", 1)[0] for ln in inspect.getsource(tb).splitlines())
        # the self-closing slash form must not appear in any CODE string (comments stripped)
        assert '"<think/>"' not in code and '"</think/>"' not in code
        assert "'<think/>'" not in code and "'</think/>'" not in code


class TestResolveThinkClosePattern:
    def test_no_chat_template(self):
        class MockTok:
            pass
        result = resolve_think_close_pattern(MockTok())
        assert result == (None, None)

    def test_template_with_newlines(self):
        class MockTok:
            chat_template = "stuff\\n</think/>\\n\\n more"
            think_end = "</think/>"
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]
        leading, trailing = resolve_think_close_pattern(MockTok())
        # Should detect \\n before and \\n\\n after </think/>
        assert leading is not None
        assert trailing is not None

    def test_template_no_whitespace(self):
        class MockTok:
            chat_template = "stuff</think/> more"
            think_end = "</think/>"
            def encode(self, text, add_special_tokens=False):
                return [1, 2]
        leading, trailing = resolve_think_close_pattern(MockTok())
        assert leading is None
        assert trailing is None  # " more" has no newline pattern
