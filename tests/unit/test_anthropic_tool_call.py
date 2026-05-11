"""Behavior tests for Anthropic tool call detection — tests real logic, not dict construction."""

from yunshu_gateway.routers.anthropic import (
    _try_parse_tool_call_delta,
    _extract_text_from_content,
)


class TestToolCallDetection:
    """Test _try_parse_tool_call_delta with real nested JSON."""

    def test_no_tool_call(self):
        result = _try_parse_tool_call_delta("Hello world")
        assert result is None

    def test_flat_json_tool_call(self):
        text = '{"name": "get_weather", "arguments": {"city": "SF"}}'
        result = _try_parse_tool_call_delta(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"

    def test_nested_json_tool_call(self):
        """M16 fix: nested JSON should work, not just flat."""
        text = '{"name": "search", "arguments": {"query": {"bool": {"must": [{"match": {"title": "test"}}]}}}}'
        result = _try_parse_tool_call_delta(text)
        assert result is not None
        assert result[0]["name"] == "search"
        import json
        args = json.loads(result[0]["arguments"])
        assert "query" in args
        assert "bool" in args["query"]

    def test_xml_wrapped_tool_call(self):
        text = '<tool_call/>{"name": "fn", "arguments": {"a": 1}}</tool_call/>'
        result = _try_parse_tool_call_delta(text)
        assert result is not None
        assert result[0]["name"] == "fn"

    def test_partial_tool_call_returns_none(self):
        text = '{"name": "get_weather", "arguments": {"city'
        result = _try_parse_tool_call_delta(text)
        assert result is None

    def test_multiple_nesting_levels(self):
        text = '{"name": "create", "arguments": {"data": {"nested": {"deep": {"value": 42}}}}}'
        result = _try_parse_tool_call_delta(text)
        assert result is not None
        assert result[0]["name"] == "create"


class TestTextExtraction:
    """Test _extract_text_from_content with various content formats."""

    def test_string_content(self):
        assert _extract_text_from_content("hello") == "hello"

    def test_list_with_text_blocks(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert _extract_text_from_content(content) == "hello\nworld"

    def test_list_with_image_blocks(self):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image", "source": {"media_type": "image/png", "data": "abc"}},
        ]
        result = _extract_text_from_content(content)
        assert "describe this" in result
        assert "[Image: image/png]" in result

    def test_list_with_cache_control(self):
        content = [
            {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
        ]
        assert _extract_text_from_content(content) == "cached"

    def test_empty_list(self):
        assert _extract_text_from_content([]) == ""

    def test_none_content(self):
        assert _extract_text_from_content(None) == ""
