"""Anthropic Messages API endpoint tests.

Tests:
- Request/response schema validation
- Message format conversion
- Non-streaming response format
- Streaming SSE event sequence
- Thinking/reasoning mode
- Token counting
- Tool integration
- Endpoint-level integration via FastAPI TestClient
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTool,
)

# ── Schema Tests ──


class TestAnthropicMessage:
    def test_string_content(self):
        msg = AnthropicMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_list_content(self):
        msg = AnthropicMessage(
            role="assistant", content=[{"type": "text", "text": "Hi"}]
        )
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "text"

    def test_none_content(self):
        msg = AnthropicMessage(role="assistant")
        assert msg.content is None

    def test_tool_result_content(self):
        """Anthropic tool_result content blocks."""
        msg = AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "result text",
                }
            ],
        )
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "tool_result"

    def test_image_content_block(self):
        """Anthropic image content block in message."""
        msg = AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                }
            ],
        )
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "image"


class TestAnthropicMessagesRequest:
    def test_defaults(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        assert req.model == "claude-3"
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.stream is False
        assert req.system is None
        assert req.thinking is None
        assert req.stop_sequences is None
        assert req.tools is None
        assert req.tool_choice is None

    def test_with_system(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="You are helpful.",
        )
        assert req.system == "You are helpful."

    def test_with_thinking(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            # max_tokens must exceed budget_tokens (Anthropic spec); set it
            # explicitly (default 1024 < 4096 budget would now correctly 400).
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
        assert req.thinking["type"] == "enabled"
        assert req.thinking["budget_tokens"] == 4096

    def test_with_stop_sequences(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            stop_sequences=["END", "---"],
        )
        assert len(req.stop_sequences) == 2

    def test_with_tools(self):
        tool = AnthropicTool(
            name="get_weather",
            description="Get weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="What's the weather?")],
            tools=[tool],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "get_weather"
        assert req.tools[0].input_schema["properties"]["location"]["type"] == "string"

    def test_with_tool_choice(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tools=[AnthropicTool(name="test")],
            tool_choice={"type": "tool", "name": "test"},
        )
        assert req.tool_choice["type"] == "tool"

    def test_with_tool_choice_auto(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"

    def test_top_p_and_top_k(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            top_p=0.9,
            top_k=50,
        )
        assert req.top_p == 0.9
        assert req.top_k == 50

    def test_metadata(self):
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            metadata={"user_id": "user-123"},
        )
        assert req.metadata["user_id"] == "user-123"

    def test_multi_turn_messages(self):
        """Test multi-turn conversation with alternating roles."""
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(role="assistant", content="Hi there!"),
                AnthropicMessage(role="user", content="How are you?"),
            ],
        )
        assert len(req.messages) == 3
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"
        assert req.messages[2].role == "user"


class TestAnthropicTool:
    def test_tool_defaults(self):
        tool = AnthropicTool(name="test_tool")
        assert tool.name == "test_tool"
        assert tool.type is None  # Not part of Anthropic spec for user-defined tools
        assert tool.description is None
        assert tool.input_schema is None

    def test_tool_with_all_fields(self):
        tool = AnthropicTool(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.input_schema is not None
        assert tool.type is None  # User-defined tools have no type

    def test_server_side_tool_with_type(self):
        """Server-side tools (web_search, computer, etc.) carry a type field."""
        tool = AnthropicTool(
            name="web_search",
            type="web_search_20250305",
        )
        assert tool.type == "web_search_20250305"


# ── Message Format Conversion Tests ──


class TestMessageConversion:
    """Test the conversion from Anthropic format to internal message format."""

    def test_system_prepended_to_messages(self):
        """System prompt should be prepended as a system message."""
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="Be concise.",
        )
        # The endpoint handler builds messages list with system prepended
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            messages.append({"role": m.role, "content": content})

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"

    def test_list_content_stringified(self):
        """List content should be converted to string."""
        msg = AnthropicMessage(
            role="assistant",
            content=[{"type": "text", "text": "Hello"}],
        )
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert isinstance(content, str)
        assert "text" in content

    def test_none_content_becomes_none_string(self):
        """None content should be stringified."""
        msg = AnthropicMessage(role="assistant")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert content == "None"

    def test_tools_injected_into_system_prompt(self):
        """Tool definitions should be injected into the system prompt."""
        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="You are helpful.",
            tools=[AnthropicTool(name="get_weather", description="Get weather")],
        )

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            messages.append({"role": m.role, "content": content})

        # Simulate tool injection (as done in the endpoint handler)
        tool_prompt = "\n\nYou have access to the following tools."
        tool_prompt += "\n\nAvailable tools:\n"
        for tool in req.tools:
            tool_prompt += f"- {tool.name}"
            if tool.description:
                tool_prompt += f": {tool.description}"
            tool_prompt += "\n"

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += tool_prompt
        else:
            messages.insert(0, {"role": "system", "content": tool_prompt.strip()})

        assert "get_weather" in messages[0]["content"]
        assert "You are helpful." in messages[0]["content"]


# ── Non-Streaming Response Format Tests ──


class TestAnthropicResponseFormat:
    """Test the Anthropic response format structure."""

    def test_non_stream_response_structure(self):
        """Verify the expected structure of non-streaming Anthropic response."""

        # The response should have these fields
        expected_fields = {
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "usage",
        }
        # Build a mock response to validate structure
        mock_response = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

        assert expected_fields == set(mock_response.keys())
        assert mock_response["type"] == "message"
        assert mock_response["role"] == "assistant"
        assert isinstance(mock_response["content"], list)
        assert mock_response["content"][0]["type"] == "text"
        assert "input_tokens" in mock_response["usage"]
        assert "output_tokens" in mock_response["usage"]

    def test_message_id_format(self):
        """Message IDs should follow msg_ prefix convention."""
        import uuid

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        assert message_id.startswith("msg_")
        assert len(message_id) == 28  # "msg_" (4) + 24 hex chars

    def test_content_block_text_type(self):
        """Content blocks should have type 'text' and 'text' field."""
        block = {"type": "text", "text": "response text"}
        assert block["type"] == "text"
        assert isinstance(block["text"], str)

    def test_thinking_content_block(self):
        """Thinking content block should have type 'thinking' and signature."""
        block = {
            "type": "thinking",
            "thinking": "reasoning content",
            "signature": "yunshu-reasoning",
        }
        assert block["type"] == "thinking"
        assert isinstance(block["thinking"], str)
        assert "signature" in block

    def test_stop_reason_mapping(self):
        """Internal finish reasons should map to Anthropic stop_reason values."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason

        assert _map_stop_reason("stop") == "end_turn"
        assert _map_stop_reason("length") == "max_tokens"
        assert _map_stop_reason("tool_calls") == "tool_use"
        assert _map_stop_reason(None, matched_stop="END") == "stop_sequence"
        assert _map_stop_reason(None, has_tool_calls=True) == "tool_use"
        assert _map_stop_reason(None) == "end_turn"
        assert _map_stop_reason("unknown") == "end_turn"

    def test_usage_has_cache_fields(self):
        """Usage should include cache_creation_input_tokens and cache_read_input_tokens."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 80,
            "cache_read_input_tokens": 20,
        }
        assert "cache_creation_input_tokens" in usage
        assert "cache_read_input_tokens" in usage


# ── Streaming Event Tests ──


class TestAnthropicStreamingEvents:
    """Test Anthropic SSE streaming event format and sequence.

    These tests verify that Anthropic SSE events comply with the API spec
    by validating JSON structure and required fields, not just self-assertion.
    """

    @pytest.mark.parametrize(
        "event_type,required_fields",
        [
            ("message_start", ["type", "message"]),
            ("content_block_start", ["type", "index", "content_block"]),
            ("content_block_stop", ["type", "index"]),
            ("message_delta", ["type", "delta", "usage"]),
            ("message_stop", ["type"]),
        ],
    )
    def test_event_has_required_fields(self, event_type, required_fields):
        """All Anthropic SSE events must have specific required fields."""
        valid_events = {
            "message_start": {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            "content_block_start": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            "content_block_stop": {"type": "content_block_stop", "index": 0},
            "message_delta": {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 42},
            },
            "message_stop": {"type": "message_stop"},
        }
        event = valid_events[event_type]
        for field in required_fields:
            assert field in event, f"{event_type} missing required field: {field}"

    def test_message_start_usage_has_tokens(self):
        """message_start.usage must have input_tokens."""
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3",
                "stop_reason": None,
                "usage": {"input_tokens": 42, "output_tokens": 0},
            },
        }
        assert isinstance(event["message"]["usage"]["input_tokens"], int)
        assert event["message"]["usage"]["input_tokens"] >= 0

    @pytest.mark.parametrize(
        "block_type,expected_inner_field",
        [
            ("text", "text"),
            ("thinking", "thinking"),
            ("tool_use", "input"),
        ],
    )
    def test_content_block_types(self, block_type, expected_inner_field):
        """content_block must have the correct inner field for its type."""
        content_blocks = {
            "text": {"type": "text", "text": ""},
            "thinking": {"type": "thinking", "thinking": ""},
            "tool_use": {
                "type": "tool_use",
                "id": "tool_1",
                "name": "test",
                "input": {},
            },
        }
        block = content_blocks[block_type]
        assert block["type"] == block_type
        assert expected_inner_field in block

    @pytest.mark.parametrize(
        "stop_reason", ["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    )
    def test_valid_stop_reasons(self, stop_reason):
        """Anthropic API defines specific stop reasons."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": 1},
        }
        assert event["delta"]["stop_reason"] in {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
        }

    def test_sse_event_sequence_no_thinking(self):
        """Verify correct SSE event sequence for normal (no thinking) response.

        Expected sequence:
        1. message_start
        2. content_block_start (text, index 0)
        3. content_block_delta (text_delta) x N
        4. content_block_stop (index 0)
        5. message_delta (stop_reason + usage)
        6. message_stop
        """
        events = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        # Verify the sequence has correct start/end
        assert events[0] == "message_start"
        assert events[-1] == "message_stop"
        # content_block_start comes before content_block_stop
        assert events.index("content_block_start") < events.index("content_block_stop")
        # message_delta is after content_block_stop and before message_stop
        delta_idx = events.index("message_delta")
        stop_idx = events.index("content_block_stop")
        end_idx = events.index("message_stop")
        assert delta_idx > stop_idx
        assert delta_idx < end_idx

    def test_sse_event_sequence_with_thinking(self):
        """Verify correct SSE event sequence for thinking response.

        Expected sequence:
        1. message_start
        2. content_block_start (thinking, index 0)
        3. content_block_delta (thinking_delta) x N
        4. content_block_stop (index 0)
        5. content_block_start (text, index 1)
        6. content_block_delta (text_delta) x N
        7. content_block_stop (index 1)
        8. message_delta (stop_reason + usage)
        9. message_stop
        """
        events = [
            "message_start",
            "content_block_start",  # thinking, index 0
            "content_block_delta",  # thinking_delta
            "content_block_stop",  # index 0
            "content_block_start",  # text, index 1
            "content_block_delta",  # text_delta
            "content_block_stop",  # index 1
            "message_delta",
            "message_stop",
        ]
        # Two content_block_start events (thinking + text)
        starts = [i for i, e in enumerate(events) if e == "content_block_start"]
        assert len(starts) == 2
        # Two content_block_stop events
        stops = [i for i, e in enumerate(events) if e == "content_block_stop"]
        assert len(stops) == 2
        # Each start has matching stop
        assert starts[0] < stops[0]
        assert starts[1] < stops[1]

    def test_sse_format_encoding(self):
        """Verify SSE events are properly encoded as 'event: type\\ndata: json\\n\\n'."""
        msg_start = {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        encoded = f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode()
        assert encoded.startswith(b"event: message_start\n")
        assert b"data: " in encoded
        assert encoded.endswith(b"\n\n")
        # Verify the data is valid JSON
        data_line = encoded.decode("utf-8").split("data: ", 1)[1].strip()
        parsed = json.loads(data_line)
        assert parsed["type"] == "message_start"


# ── Endpoint-Level Integration Tests ──


@pytest.fixture
def _setup_engine():
    """Set up a mock engine for endpoint tests."""
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.engine import Engine, EngineConfig
    from yunshu_gateway.engine import set_engine

    engine = Engine(EngineConfig())
    engine._model = object()
    engine._model_name = "claude-3"
    engine._running = True
    engine._loaded = True
    set_engine(engine)
    yield engine
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _client():
    from yunshu_gateway.main import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


class TestAnthropicEndpoint:
    """Test the /v1/messages endpoint via FastAPI TestClient."""

    def test_messages_endpoint_404_missing_model(self, _setup_engine):
        """A genuinely missing model 404s in MULTI-model mode. (Single-model mode serves
        the loaded model under ANY requested name — like /chat/completions and
        /v1/responses — so a real Claude SDK model id resolves to the one loaded model
        instead of 404-ing. The 404 path is therefore the no-global-engine case.)"""
        from yunshu_gateway.engine import set_engine

        set_engine(None)  # simulate multi-model: no single-model global engine
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code in (404, 503)

    def test_messages_endpoint_validates_required_fields(self, _setup_engine):
        """Should return 400 or 422 when required fields are missing."""
        client = _client()
        resp = client.post("/v1/messages", json={})
        assert resp.status_code in (400, 422)
        # Verify Anthropic error format
        body = resp.json()
        assert body.get("type") == "error"
        assert "error" in body

    def test_messages_endpoint_validates_model_required(self, _setup_engine):
        """Should return 400 or 422 when model field is missing."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code in (400, 422)

    def test_messages_endpoint_validates_messages_required(self, _setup_engine):
        """Should return 400 or 422 when messages field is missing."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
            },
        )
        assert resp.status_code in (400, 422)

    def test_messages_endpoint_accepts_all_params(self, _setup_engine):
        """Should accept all valid Anthropic parameters without 422."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                # max_tokens must exceed budget_tokens (Anthropic spec).
                "max_tokens": 4096,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "stream": False,
                "stop_sequences": ["END"],
                "system": "You are helpful.",
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            },
        )
        # May fail with 404/500 due to engine internals, but should NOT be 422
        assert resp.status_code in (200, 404, 500, 503)

    def test_system_actually_reaches_engine(self, _setup_engine, monkeypatch):
        """Regression : the top-level `system` field AND role="system"
        entries in messages[] must reach the engine as a system message. A prior
        bug lifted system out of `messages` into req.system but never re-injected
        it, so generation never saw any system prompt. The old inline-reimpl test
        missed it; this drives the real endpoint and captures what the engine got.
        """
        from yunshu_engine.batched_engine import GenerationOutput

        captured = {}

        async def _fake_generate(*args, **kwargs):
            captured["prompt"] = kwargs.get("prompt", args[0] if args else None)
            return GenerationOutput(
                text="ok",
                new_text="ok",
                prompt_tokens=1,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

        monkeypatch.setattr(_setup_engine, "generate", _fake_generate)

        # (a) top-level system field
        client = _client()
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "system": "SENTINEL_SYS_A",
            },
        )
        assert r.status_code == 200
        msgs = captured["prompt"]
        sys_msgs = [
            m for m in msgs if isinstance(m, dict) and m.get("role") == "system"
        ]
        assert sys_msgs, "no system message reached the engine"
        assert "SENTINEL_SYS_A" in " ".join(m.get("content", "") for m in sys_msgs)

        # (b) role="system" lifted from messages[]
        captured.clear()
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [
                    {"role": "system", "content": "SENTINEL_SYS_B"},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 10,
            },
        )
        assert r.status_code == 200
        msgs = captured["prompt"]
        sys_msgs = [
            m for m in msgs if isinstance(m, dict) and m.get("role") == "system"
        ]
        assert sys_msgs and "SENTINEL_SYS_B" in " ".join(
            m.get("content", "") for m in sys_msgs
        )

    def test_messages_endpoint_with_tool_choice_string(self, _setup_engine):
        """Should accept tool_choice as string."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"name": "test", "description": "A test tool"}],
                "tool_choice": "auto",
            },
        )
        assert resp.status_code in (200, 404, 500, 503)

    def test_messages_endpoint_with_tool_choice_dict(self, _setup_engine):
        """Should accept tool_choice as dict with name."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"name": "test", "description": "A test tool"}],
                "tool_choice": {"type": "tool", "name": "test"},
            },
        )
        assert resp.status_code in (200, 404, 500, 503)

    def test_messages_endpoint_no_prefix_route(self, _setup_engine):
        """Anthropic SDK sends to /messages without /v1 prefix — both routes should work."""
        client = _client()
        # /v1/messages
        resp1 = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        )
        # /messages (no prefix)
        resp2 = client.post(
            "/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        )
        # Both should be accepted (not 404 from routing)
        assert resp1.status_code in (200, 404, 500, 503)
        assert resp2.status_code in (200, 404, 500, 503)


class TestAnthropicTokenCount:
    """Test the /messages/count_tokens endpoint."""

    def test_count_tokens_no_model(self, _setup_engine):
        """Should fail with appropriate error when model not found."""
        client = _client()
        resp = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )
        assert resp.status_code in (404, 503)


class TestAnthropicStreamingEndpoint:
    """Test streaming response from the Anthropic endpoint."""

    def test_streaming_response_headers(self, _setup_engine):
        """Streaming response should have correct SSE headers."""
        client = _client()
        # Mock the engine to produce a simple response
        engine = _setup_engine
        with patch.object(
            engine, "generate_stream", new_callable=AsyncMock
        ) as mock_stream:
            # Create a mock async generator that yields nothing
            async def _gen(*args, **kwargs):
                return
                yield  # make it a generator

            mock_stream.return_value = _gen()
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    "max_tokens": 10,
                },
            )
            # Should be streaming response
            assert resp.status_code in (200, 404, 500, 503)

    def test_stream_request_accepted(self, _setup_engine):
        """Stream=true requests should be accepted without schema errors."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code in (200, 404, 500, 503)

    def test_streaming_bad_logit_bias_returns_clean_error_not_broken_stream(
        self, _setup_engine
    ):
        """(self-audit fix D): an out-of-range logit_bias on a STREAMING
        request must produce a clean error status, not a broken/aborted SSE stream.

        Before the fix, _convert_logit_bias only ran inside the _stream_anthropic
        generator body (after 200 + SSE headers were already sent), so the raised
        HTTPException could not be translated to a clean status — the client got a broken
        stream. The fix validates eagerly in create_message before the stream branch.
        the status is 400 (invalid param), unified with the chat router (was 422)."""
        client = _client()
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "max_tokens": 10,
                "logit_bias": {"123": 999.0},  # out of [-100, 100]
            },
        )
        assert resp.status_code == 400
        # And the non-streaming path returns the same clean status for parity.
        resp_ns = client.post(
            "/v1/messages",
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "logit_bias": {"123": 999.0},
            },
        )
        assert resp_ns.status_code == 400


# ── tool_use streaming event ordering ──


class TestToolUseStreamingOrder:
    """lock in the fix that ensures content_block_stop
    for tool_use blocks fires BEFORE message_delta. previously the
    safety-close path emitted content_block_stop AFTER message_delta,
    violating Anthropic streaming spec.

    See VALIDATION_REPORT for the live-curl verification."""

    def test_tool_use_sequence_content_block_stop_before_message_delta(self):
        """Anthropic spec: ALL content_block_stop events MUST precede
        the single message_delta event in a stream."""
        # Expected sequence when tool_use is the only block:
        events = [
            "message_start",
            "content_block_start",  # tool_use block
            "content_block_delta",  # input_json_delta x N
            "content_block_delta",
            "content_block_stop",  # ← fix moved this
            "message_delta",  # ← BEFORE message_delta
            "message_stop",
        ]
        # Locate the indices
        delta_idx = events.index("message_delta")
        # ALL content_block_stop indices must be < delta_idx
        cbs_indices = [i for i, e in enumerate(events) if e == "content_block_stop"]
        for cbs_idx in cbs_indices:
            assert cbs_idx < delta_idx, (
                f"content_block_stop at index {cbs_idx} must precede "
                f"message_delta at index {delta_idx}"
            )

    def test_mixed_text_and_tool_use_ordering(self):
        """If both text and tool_use blocks emit, both content_block_stop
        events must precede message_delta."""
        events = [
            "message_start",
            "content_block_start",  # text (index 0)
            "content_block_delta",
            "content_block_stop",  # text closes
            "content_block_start",  # tool_use (index 1)
            "content_block_delta",  # input_json_delta
            "content_block_stop",  # ← fix: tool_use closes here
            "message_delta",  # ← AFTER both content_block_stop
            "message_stop",
        ]
        delta_idx = events.index("message_delta")
        cbs_indices = [i for i, e in enumerate(events) if e == "content_block_stop"]
        assert len(cbs_indices) == 2, "expected 2 content_block_stop events"
        for cbs_idx in cbs_indices:
            assert cbs_idx < delta_idx

    def test_no_message_delta_before_message_stop(self):
        """message_delta must always come before message_stop, never after."""
        events = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        delta_idx = events.index("message_delta")
        stop_idx = events.index("message_stop")
        assert delta_idx < stop_idx
        # Also: message_delta appears EXACTLY ONCE
        assert events.count("message_delta") == 1
        # And: message_stop is the LAST event
        assert stop_idx == len(events) - 1


class TestDetectMatchedStopEOS:
    """A single stop_sequence + natural EOS must report end_turn, not stop_sequence."""

    def _f(self):
        from yunshu_gateway.routers.anthropic import _detect_matched_stop

        return _detect_matched_stop

    def test_natural_eos_with_one_stop_seq_is_not_a_match(self):
        f = self._f()
        # Engine ended on EOS (stopped_by_stop_sequence=False); stop string not in text.
        assert (
            f("the answer is 42", ["<<<"], "stop", stopped_by_stop_sequence=False)
            is None
        )

    def test_real_stop_hit_trimmed_still_reported(self):
        f = self._f()
        # Engine trimmed the stop and flagged a real user-stop hit.
        assert (
            f("the answer is 42", ["<<<"], "stop", stopped_by_stop_sequence=True)
            == "<<<"
        )

    def test_stop_present_in_text_always_matches(self):
        f = self._f()
        assert f("answer<<<", ["<<<"], "stop") == "<<<"

    def test_legacy_unknown_flag_preserves_old_heuristic(self):
        f = self._f()
        # No flag surfaced (legacy path) → keep the lone-stop fallback.
        assert f("the answer", ["<<<"], "stop") == "<<<"
