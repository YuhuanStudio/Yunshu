"""Tests for enhanced Anthropic API features.

Covers:
- Image block support in messages (base64)
- Cache control hints extraction
- Streaming tool-use deltas (input_json_delta)
- stop_sequences parameter handling
- Content block extraction (_extract_text_from_content)
- System as list[dict] support
"""


from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    _extract_cache_control_hints,
    _extract_text_from_content,
    _has_image_blocks,
    _try_parse_tool_call_delta,
)

# ── Image block support ──


class TestImageBlockSupport:
    def test_image_block_extracted(self):
        """Image blocks should produce a placeholder description."""
        content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo=",
                },
            },
        ]
        text = _extract_text_from_content(content)
        assert "What is in this image?" in text
        assert "[Image: image/png]" in text

    def test_image_block_jpeg(self):
        """JPEG image blocks should show correct media type."""
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "/9j/4AAQ",
                },
            },
        ]
        text = _extract_text_from_content(content)
        assert "[Image: image/jpeg]" in text

    def test_image_block_gif(self):
        """GIF image blocks should show correct media type."""
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/gif",
                    "data": "R0lGODlh",
                },
            },
        ]
        text = _extract_text_from_content(content)
        assert "[Image: image/gif]" in text

    def test_has_image_blocks_true(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        ]
        assert _has_image_blocks(content) is True

    def test_has_image_blocks_false_string(self):
        assert _has_image_blocks("just text") is False

    def test_has_image_blocks_false_list(self):
        content = [
            {"type": "text", "text": "hello"},
        ]
        assert _has_image_blocks(content) is False

    def test_has_image_blocks_none(self):
        assert _has_image_blocks(None) is False

    def test_image_in_message_schema(self):
        """AnthropicMessage should accept image content blocks."""
        msg = AnthropicMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                },
            ],
        )
        assert isinstance(msg.content, list)
        assert msg.content[1]["type"] == "image"
        assert msg.content[1]["source"]["media_type"] == "image/png"


# ── Cache control hints ──


class TestCacheControlHints:
    def test_cache_control_in_text_block(self):
        """Cache control hints in text blocks should be extracted."""
        system = [
            {"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Additional context."},
        ]
        hints, char_offsets = _extract_cache_control_hints(system)
        assert len(hints) == 1
        assert hints[0]["type"] == "ephemeral"
        assert char_offsets == [28]

    def test_cache_control_none_system(self):
        """None system should return empty hints and offsets."""
        hints, char_offsets = _extract_cache_control_hints(None)
        assert hints == []
        assert char_offsets == []

    def test_cache_control_string_system(self):
        """String system should return empty hints and offsets."""
        hints, char_offsets = _extract_cache_control_hints("You are helpful.")
        assert hints == []
        assert char_offsets == []

    def test_cache_control_multiple_hints(self):
        system = [
            {"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "C"},
        ]
        hints, char_offsets = _extract_cache_control_hints(system)
        assert len(hints) == 2
        # "A" = 1 char, then "\n" separator, then "B" = 1 char → offset 3
        assert char_offsets == [1, 3]

    def test_cache_control_not_stripped_from_text(self):
        """Cache control should be a routing hint, not affect text extraction."""
        content = [
            {"type": "text", "text": "Cacheable content", "cache_control": {"type": "ephemeral"}},
        ]
        text = _extract_text_from_content(content)
        assert text == "Cacheable content"


# ── Streaming tool-use deltas ──


class TestToolUseDeltas:
    def test_parse_json_tool_call(self):
        """Should detect JSON tool call with name and arguments."""
        text = 'Some text {"name": "get_weather", "arguments": {"city": "SF"}} more text'
        calls = _try_parse_tool_call_delta(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert '"city": "SF"' in calls[0]["arguments"]

    def test_parse_xml_tool_call(self):
        """Should detect XML-wrapped tool call."""
        text = '<tool_call/>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call/>'
        calls = _try_parse_tool_call_delta(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "calculate"

    def test_no_tool_call_returns_none(self):
        """Plain text should return None."""
        text = "Just a normal response without any tool calls."
        calls = _try_parse_tool_call_delta(text)
        assert calls is None

    def test_multiple_json_tool_calls(self):
        """Should detect multiple JSON tool calls in text."""
        text = (
            '{"name": "get_weather", "arguments": {"city": "SF"}} '
            '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        )
        calls = _try_parse_tool_call_delta(text)
        assert calls is not None
        assert len(calls) == 2
        assert calls[0]["name"] == "get_weather"
        assert calls[1]["name"] == "get_weather"

    def test_input_json_delta_format(self):
        """Verify the format matches Anthropic's input_json_delta spec."""
        text = '{"name": "search_web", "arguments": {"query": "test"}}'
        calls = _try_parse_tool_call_delta(text)
        assert calls is not None
        # The 'arguments' value should be a JSON string for input_json_delta
        call = calls[0]
        assert "name" in call
        assert "arguments" in call
        # arguments should be a string (the raw JSON)
        assert isinstance(call["arguments"], str)


# ── stop_sequences ──


class TestStopSequences:
    def test_stop_sequences_in_request(self):
        """Request schema should accept stop_sequences."""
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hello")],
            stop_sequences=["END", "---", "STOP"],
        )
        assert len(req.stop_sequences) == 3
        assert "END" in req.stop_sequences

    def test_stop_sequences_default_none(self):
        """Default stop_sequences should be None."""
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        assert req.stop_sequences is None

    def test_stop_sequences_empty_list(self):
        """Empty list of stop sequences should be accepted."""
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hello")],
            stop_sequences=[],
        )
        assert req.stop_sequences == []


# ── _extract_text_from_content ──


class TestExtractTextFromContent:
    def test_string_passthrough(self):
        assert _extract_text_from_content("hello") == "hello"

    def test_none_returns_empty(self):
        assert _extract_text_from_content(None) == ""

    def test_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert _extract_text_from_content(content) == "Hello\nWorld"

    def test_tool_use_block(self):
        content = [
            {"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}},
        ]
        text = _extract_text_from_content(content)
        assert "Tool use: get_weather" in text
        assert "SF" in text

    def test_tool_result_block_string(self):
        content = [
            {"type": "tool_result", "tool_use_id": "toolu_123", "content": "72F sunny"},
        ]
        text = _extract_text_from_content(content)
        assert "Tool result toolu_123" in text
        assert "72F sunny" in text

    def test_tool_result_block_nested(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_456",
                "content": [{"type": "text", "text": "Result: success"}],
            },
        ]
        text = _extract_text_from_content(content)
        assert "Result: success" in text

    def test_thinking_block(self):
        content = [
            {"type": "thinking", "thinking": "Let me reason about this..."},
        ]
        text = _extract_text_from_content(content)
        assert "Let me reason about this..." in text

    def test_mixed_blocks(self):
        content = [
            {"type": "text", "text": "Check weather"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
            {"type": "text", "text": "in this map"},
        ]
        text = _extract_text_from_content(content)
        assert "Check weather" in text
        assert "[Image: image/png]" in text
        assert "in this map" in text

    def test_unknown_block_type(self):
        content = [{"type": "custom", "data": "stuff"}]
        text = _extract_text_from_content(content)
        assert len(text) > 0  # Should not crash


# ── System as list[dict] ──


class TestSystemListDict:
    def test_system_accepts_list_of_dicts(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system=[
                {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be concise."},
            ],
        )
        assert isinstance(req.system, list)
        assert len(req.system) == 2

    def test_system_accepts_string(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="You are helpful.",
        )
        assert isinstance(req.system, str)

    def test_system_text_extraction_from_list(self):
        system = [
            {"type": "text", "text": "Part one.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Part two."},
        ]
        text = _extract_text_from_content(system)
        assert "Part one." in text
        assert "Part two." in text
