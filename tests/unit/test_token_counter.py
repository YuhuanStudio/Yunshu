"""Token counter service tests."""

from yunshu_control.token_counter import (
    count_message_tokens,
    count_tokens,
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


class TestCountMessageToolCalls:
    def test_assistant_tool_calls_counted(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Taipei"},
                        },
                    }
                ],
            }
        ]
        # tool_calls path (dict arguments are json-dumped) adds tokens beyond base
        with_tc = count_message_tokens(messages)
        without_tc = count_message_tokens([{"role": "assistant", "content": ""}])
        assert with_tc > without_tc

    def test_tool_role_message(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": "sunny",
            }
        ]
        count = count_message_tokens(messages)
        assert count > 4  # tool_call_id + name + content overhead

    def test_string_arguments_counted(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "f", "arguments": "{\"a\": 1}"}}
                ],
            }
        ]
        assert count_message_tokens(messages) > 0

    def test_content_list_of_strings(self):
        messages = [{"role": "user", "content": ["hello", "world"]}]
        assert count_message_tokens(messages) > 0


class TestEstimateCost:
    def test_zero_cost(self):
        result = estimate_cost(0, 0)
        assert result["total_cost_usd"] == 0.0

    def test_with_tokens(self):
        result = estimate_cost(100, 50)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert "total_cost_usd" in result

    def test_exact_match_pricing(self):
        # exact key match (gpt-4o-mini)
        result = estimate_cost(1_000_000, 1_000_000, "gpt-4o-mini")
        assert result["prompt_cost_usd"] == 0.15
        assert result["completion_cost_usd"] == 0.60
        assert result["total_cost_usd"] == 0.75

    def test_substring_match_pricing(self):
        # "qwen" is a substring of the model id -> open-source pricing
        result = estimate_cost(1_000_000, 1_000_000, "Qwen2.5-0.5B-Instruct-4bit")
        assert result["prompt_cost_usd"] == 0.05
        assert result["completion_cost_usd"] == 0.10

    def test_anthropic_substring(self):
        result = estimate_cost(1_000_000, 0, "claude-opus-4-1")
        assert result["prompt_cost_usd"] == 15.00

    def test_unknown_model_falls_back_to_default(self):
        result = estimate_cost(1_000_000, 1_000_000, "totally-unknown-model")
        assert result["prompt_cost_usd"] == 0.05
        assert result["completion_cost_usd"] == 0.10

    def test_empty_model_id_uses_default(self):
        result = estimate_cost(1_000_000, 0, "")
        assert result["prompt_cost_usd"] == 0.05
