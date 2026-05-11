"""Tests for streaming module — ThinkingParser, SSE formatters, keepalive."""

import json

from yunshu_gateway.streaming import (
    ThinkingParser,
    extract_thinking,
    extract_tool_calls,
    clean_tool_call_markup,
    format_anthropic_chunk,
    format_openai_chunk,
    format_openai_done,
    format_openai_non_stream,
    _KEEPALIVE_SENTINEL,
)


# ── ThinkingParser ──


class TestThinkingParser:
    def test_plain_text_passes_through(self):
        p = ThinkingParser()
        result = p.process_chunk("Hello world")
        assert result["visible"] == "Hello world"
        assert result["thinking"] == ""
        assert not result["in_thinking"]

    def test_thinking_tag_splits_content(self):
        p = ThinkingParser()
        result = p.process_chunk("<think/>reasoning here</think/>visible output")
        assert result["thinking"] == "reasoning here"
        assert result["visible"] == "visible output"
        assert not result["in_thinking"]

    def test_thinking_across_chunks(self):
        p = ThinkingParser()
        r1 = p.process_chunk("<think/>first ")
        assert r1["thinking"] == "first "
        assert r1["visible"] == ""
        assert r1["in_thinking"]

        r2 = p.process_chunk("second</think/>done")
        assert r2["thinking"] == "second"
        assert r2["visible"] == "done"
        assert not r2["in_thinking"]

    def test_tag_split_across_chunks(self):
        """When tag is split across chunks, partial tag is retained in buffer."""
        p = ThinkingParser()
        r1 = p.process_chunk("<thi")
        # Partial tag is retained, not emitted
        assert r1["visible"] == ""
        assert p.buffer == "<thi"

        r2 = p.process_chunk("nk/>inside")
        assert r2["thinking"] == "inside"
        assert r2["in_thinking"]

    def test_end_tag_split(self):
        p = ThinkingParser()
        p.process_chunk("<think/>inside</thi")
        # Buffer retains partial end tag
        r = p.process_chunk("nk/>outside")
        assert r["in_thinking"] is False
        assert "outside" in r["visible"]

    def test_finalize_preserves_thinking_state(self):
        p = ThinkingParser()
        p.process_chunk("<think/>still thinking")
        assert p.in_thinking
        # Buffer is fully drained by process_chunk, finalize reports state
        result = p.finalize()
        assert result["in_thinking"]  # still in thinking mode

    def test_finalize_visible_done(self):
        p = ThinkingParser()
        p.process_chunk("some text")
        # Buffer is fully drained, finalize reports no active thinking
        result = p.finalize()
        assert not result["in_thinking"]

    def test_empty_input(self):
        p = ThinkingParser()
        result = p.process_chunk("")
        assert result["visible"] == ""
        assert result["thinking"] == ""

    def test_multiple_think_blocks(self):
        p = ThinkingParser()
        r = p.process_chunk("<think/>first</think/>middle<think/>second</think/>end")
        assert "middle" in r["visible"]
        assert "end" in r["visible"]


# ── OpenAI SSE Formatters ──


class TestOpenAIFormat:
    def test_chunk_format(self):
        chunk = format_openai_chunk(
            completion_id="chatcmpl-123",
            model="test-model",
            delta_content="Hello",
        )
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        data = json.loads(chunk[len("data: "):])
        assert data["id"] == "chatcmpl-123"
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_chunk_with_finish_reason(self):
        chunk = format_openai_chunk(
            completion_id="chatcmpl-123",
            model="test-model",
            delta_content="",
            finish_reason="stop",
        )
        data = json.loads(chunk[len("data: "):])
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_chunk_with_thinking(self):
        chunk = format_openai_chunk(
            completion_id="chatcmpl-123",
            model="test-model",
            delta_content="visible",
            thinking_content="reasoning",
        )
        data = json.loads(chunk[len("data: "):])
        assert data["choices"][0]["delta"]["reasoning_content"] == "reasoning"

    def test_done_signal(self):
        assert format_openai_done() == "data: [DONE]\n\n"

    def test_non_stream_response(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test-model",
            content="Hello world",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert resp["id"] == "chatcmpl-123"
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"] == "Hello world"
        assert resp["usage"]["total_tokens"] == 15

    def test_non_stream_with_thinking(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test-model",
            content="visible",
            prompt_tokens=5,
            completion_tokens=3,
            thinking_content="reasoning",
        )
        assert resp["choices"][0]["message"]["reasoning_content"] == "reasoning"

    def test_unicode_content(self):
        chunk = format_openai_chunk(
            completion_id="chatcmpl-123",
            model="test-model",
            delta_content="你好世界 🌍",
        )
        data = json.loads(chunk[len("data: "):])
        assert data["choices"][0]["delta"]["content"] == "你好世界 🌍"


# ── Anthropic SSE Formatter ──


class TestAnthropicFormat:
    def test_content_block_delta(self):
        chunk = format_anthropic_chunk(
            message_id="msg-123",
            model="claude-3",
            delta_text="Hello",
        )
        lines = chunk.split("\n")
        assert lines[0] == "event: content_block_delta"
        data = json.loads(lines[1][len("data: "):])
        assert data["delta"]["text"] == "Hello"
        assert data["delta"]["type"] == "text_delta"

    def test_message_start(self):
        chunk = format_anthropic_chunk(
            message_id="msg-123",
            model="claude-3",
            delta_text="",
            event_type="message_start",
        )
        data = json.loads(chunk.split("\n")[1][len("data: "):])
        assert data["message"]["id"] == "msg-123"
        assert data["message"]["role"] == "assistant"

    def test_message_delta(self):
        chunk = format_anthropic_chunk(
            message_id="msg-123",
            model="claude-3",
            delta_text="",
            event_type="message_delta",
        )
        data = json.loads(chunk.split("\n")[1][len("data: "):])
        assert data["delta"]["stop_reason"] == "end_turn"


# ── extract_thinking ──


class TestExtractThinking:
    def test_normal(self):
        thinking, content = extract_thinking("<think/>reasoning</think/>answer")
        assert thinking == "reasoning"
        assert content == "answer"

    def test_no_thinking(self):
        thinking, content = extract_thinking("just answer")
        assert thinking == ""
        assert content == "just answer"

    def test_partial_no_open_tag(self):
        thinking, content = extract_thinking("reasoning</think/>answer")
        assert thinking == "reasoning"
        assert content == "answer"

    def test_empty_think(self):
        thinking, content = extract_thinking("<think/></think/>answer")
        assert thinking == ""
        assert content == "answer"

    def test_think_only(self):
        thinking, content = extract_thinking("<think/>reasoning</think/>")
        assert thinking == "reasoning"
        assert content == ""

    def test_empty_string(self):
        thinking, content = extract_thinking("")
        assert thinking == ""
        assert content == ""

    def test_multiple_blocks(self):
        thinking, content = extract_thinking(
            "<think/>first</think/>middle<think/>second</think/>end"
        )
        assert "first" in thinking
        assert "second" in thinking
        assert "middle" in content
        assert "end" in content


# ── Tool call extraction ──


class TestExtractToolCalls:
    def test_hermes_format(self):
        text = '<tool_call/>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call/>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert "city" in calls[0]["arguments"]

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call/>{"name": "fn1", "arguments": {"a": 1}}</tool_call/>'
            '<tool_call/>{"name": "fn2", "arguments": {"b": 2}}</tool_call/>'
        )
        calls = extract_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        calls = extract_tool_calls("just normal text")
        assert calls == []

    def test_code_block_format(self):
        text = '```json\n{"name": "search", "arguments": {"q": "test"}}\n```'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"


class TestCleanToolCallMarkup:
    def test_removes_tool_call_tags(self):
        text = 'before<tool_call/>{"name": "fn"}</tool_call/>after'
        cleaned = clean_tool_call_markup(text)
        assert "<tool_call" not in cleaned
        assert "</tool_call" not in cleaned
        assert "before" in cleaned
        assert "after" in cleaned

    def test_cleans_extra_newlines(self):
        text = "hello\n\n\n\nworld"
        cleaned = clean_tool_call_markup(text)
        assert cleaned == "hello\n\nworld"


class TestQwenXmlToolCalls:
    """Tests for Qwen/Llama XML format tool calling."""

    def test_qwen_xml_format(self):
        text = '<function=get_weather>{"city": "SF"}</function>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert "city" in calls[0]["arguments"]

    def test_qwen_xml_with_parameters(self):
        text = '<function=search><parameter=query>test query</parameter></function>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert "test query" in calls[0]["arguments"]

    def test_clean_removes_function_tags(self):
        text = 'before<function=fn>{"a": 1}</function>after'
        cleaned = clean_tool_call_markup(text)
        assert "<function" not in cleaned
        assert "before" in cleaned
        assert "after" in cleaned

    def test_argument_sanitization(self):
        from yunshu_gateway.streaming import _sanitize_arguments
        assert _sanitize_arguments({"a": 1}) == '{"a": 1}'
        assert _sanitize_arguments('{"a": 1}') == '{"a": 1}'
        assert _sanitize_arguments("not json") == "{}"
        assert _sanitize_arguments(None) == "{}"


# ── Sentinel ──


class TestSentinel:
    def test_sentinel_is_unique(self):
        assert _KEEPALIVE_SENTINEL is not None
        assert _KEEPALIVE_SENTINEL != 0
        assert _KEEPALIVE_SENTINEL != ""


class TestLogprobsFormatting:
    """Tests for logprobs in SSE formatters."""

    def test_non_stream_with_logprobs(self):
        lp = {"content": [{"token": "hello", "logprob": -0.5, "bytes": [104, 101], "top_logprobs": []}]}
        result = format_openai_non_stream(
            completion_id="test-123",
            model="test-model",
            content="hello world",
            prompt_tokens=5,
            completion_tokens=2,
            logprobs=lp,
        )
        assert result["choices"][0]["logprobs"] == lp

    def test_non_stream_without_logprobs(self):
        result = format_openai_non_stream(
            completion_id="test-123",
            model="test-model",
            content="hello",
            prompt_tokens=5,
            completion_tokens=1,
        )
        assert "logprobs" not in result["choices"][0]

    def test_chunk_with_logprobs(self):
        lp = {"content": [{"token": "hi", "logprob": -0.2, "bytes": [], "top_logprobs": []}]}
        chunk_str = format_openai_chunk(
            completion_id="test-123",
            model="test-model",
            delta_content="hi",
            logprobs=lp,
        )
        chunk = json.loads(chunk_str.split("data: ")[1].strip())
        assert chunk["choices"][0]["logprobs"] == lp

    def test_chunk_without_logprobs(self):
        chunk_str = format_openai_chunk(
            completion_id="test-123",
            model="test-model",
            delta_content="hi",
        )
        chunk = json.loads(chunk_str.split("data: ")[1].strip())
        assert "logprobs" not in chunk["choices"][0]
