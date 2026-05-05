"""Token counter service tests."""

from yunshu_control.token_counter import (
    count_tokens,
    count_message_tokens,
    estimate_cost,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_none_text(self):
        assert count_tokens(None) == 0

    def test_heuristic_short(self):
        count = count_tokens("hello world")
        assert count > 0
        assert count >= 2  # at least 2 words

    def test_heuristic_long(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = count_tokens(text)
        assert count > 50

    def test_with_mock_tokenizer(self):
        class MockTokenizer:
            def encode(self, text):
                return text.split()
        count = count_tokens("hello world test", MockTokenizer())
        assert count == 3

    def test_tokenizer_error_fallback(self):
        class BrokenTokenizer:
            def encode(self, text):
                raise RuntimeError("broken")
        count = count_tokens("hello world", BrokenTokenizer())
        assert count > 0


class TestCountMessageTokens:
    def test_single_message(self):
        messages = [{"role": "user", "content": "hello"}]
        count = count_message_tokens(messages)
        assert count > 0

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        count = count_message_tokens(messages)
        assert count > 10

    def test_content_list(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
        ]
        count = count_message_tokens(messages)
        assert count >= 85  # includes image overhead


class TestEstimateCost:
    def test_zero_cost(self):
        result = estimate_cost(0, 0)
        assert result["total_cost_usd"] == 0.0

    def test_with_tokens(self):
        result = estimate_cost(100, 50)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert "total_cost_usd" in result
