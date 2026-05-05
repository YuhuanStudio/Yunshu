"""Realtime API audio I/O unit tests.

Tests audio buffer management, TTS synthesis, session config,
conversation items, and message building — all using mock WebSocket objects.
"""

import base64

import pytest

from yunshu_gateway.routers.realtime import (
    Conversation,
    ConversationItem,
    RealtimeEvent,
    RealtimeSession,
    SessionConfig,
    _event,
)


# ── Mock WebSocket ──


class MockWebSocket:
    """Minimal async mock for WebSocket used by RealtimeSession."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive_text(self) -> str:
        raise Exception("test end")


def _make_session() -> RealtimeSession:
    """Create a RealtimeSession backed by a MockWebSocket."""
    ws = MockWebSocket()
    return RealtimeSession(ws)


# ── TestAudioBufferAppend ──


class TestAudioBufferAppend:
    """Tests for _handle_input_audio_buffer_append."""

    @pytest.mark.asyncio
    async def test_valid_base64_audio_stored_in_buffer(self):
        session = _make_session()
        audio_bytes = b"\x00\x01\x02\x03\x04\x05"
        audio_b64 = base64.b64encode(audio_bytes).decode()

        await session._handle_input_audio_buffer_append({"audio": audio_b64})

        assert hasattr(session, "_audio_buffer")
        assert bytes(session._audio_buffer) == audio_bytes

    @pytest.mark.asyncio
    async def test_empty_audio_string_does_nothing(self):
        session = _make_session()
        await session._handle_input_audio_buffer_append({"audio": ""})
        assert len(session._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_missing_audio_key_does_nothing(self):
        session = _make_session()
        await session._handle_input_audio_buffer_append({})
        assert len(session._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_multiple_appends_accumulate(self):
        session = _make_session()
        chunk1 = b"\x00\x01"
        chunk2 = b"\x02\x03"
        chunk3 = b"\x04\x05"

        await session._handle_input_audio_buffer_append(
            {"audio": base64.b64encode(chunk1).decode()},
        )
        await session._handle_input_audio_buffer_append(
            {"audio": base64.b64encode(chunk2).decode()},
        )
        await session._handle_input_audio_buffer_append(
            {"audio": base64.b64encode(chunk3).decode()},
        )

        assert bytes(session._audio_buffer) == chunk1 + chunk2 + chunk3


# ── TestAudioBufferCommit ──


class TestAudioBufferCommit:
    """Tests for _handle_input_audio_buffer_commit."""

    @pytest.mark.asyncio
    async def test_sends_committed_event(self):
        session = _make_session()
        # Populate buffer so there's something to commit
        session._audio_buffer = bytearray(b"\x00\x01\x02")

        await session._handle_input_audio_buffer_commit({})

        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        committed_events = [
            e for e in ws.sent
            if e.get("type") == RealtimeEvent.INPUT_AUDIO_BUFFER_COMMITTED
        ]
        assert len(committed_events) == 1
        assert "event_id" in committed_events[0]

    @pytest.mark.asyncio
    async def test_empty_buffer_no_asr_call(self):
        session = _make_session()
        # No _audio_buffer attribute at all
        await session._handle_input_audio_buffer_commit({})
        # Should only have the committed event, no conversation.item.created
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        item_events = [
            e for e in ws.sent
            if e.get("type") == "conversation.item.created"
        ]
        assert len(item_events) == 0

    @pytest.mark.asyncio
    async def test_buffer_cleared_after_commit(self):
        session = _make_session()
        session._audio_buffer = bytearray(b"\x00\x01\x02")

        await session._handle_input_audio_buffer_commit({})

        assert len(session._audio_buffer) == 0


# ── TestSynthesizeAudio ──


class TestSynthesizeAudio:
    """Tests for _synthesize_audio_response."""

    @pytest.mark.asyncio
    async def test_handles_no_engine_gracefully(self):
        session = _make_session()
        # get_model_manager returns None by default in test context
        # _synthesize_audio_response should not raise
        await session._synthesize_audio_response("hello", "resp_1", "item_1")
        # No audio events sent because there is no model manager
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        audio_events = [
            e for e in ws.sent
            if e.get("type") in (
                RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
                RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DONE,
            )
        ]
        assert len(audio_events) == 0

    @pytest.mark.asyncio
    async def test_handles_no_model_manager_gracefully(self):
        session = _make_session()
        # Even with empty text, should not crash
        await session._synthesize_audio_response("", "resp_2", "item_2")
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        # No events at all (empty text path or no manager)
        assert len(ws.sent) == 0


# ── TestSessionConfig ──


class TestSessionConfig:
    """Tests for SessionConfig defaults, update, and serialization."""

    def test_defaults(self):
        config = SessionConfig()
        assert config.model == "default"
        assert config.modalities == ["text"]
        assert config.voice == "alloy"
        assert config.input_audio_format == "pcm16"
        assert config.output_audio_format == "pcm16"
        assert config.turn_detection["type"] == "server_vad"
        assert config.max_response_output_tokens == 4096
        assert config.temperature == 0.7
        assert config.tools == []

    def test_update_applies_and_returns_changed_fields(self):
        config = SessionConfig()
        changed = config.update({
            "model": "qwen3",
            "temperature": 0.3,
            "modalities": ["text", "audio"],
        })
        assert "model" in changed
        assert "temperature" in changed
        assert "modalities" in changed
        assert config.model == "qwen3"
        assert config.temperature == 0.3
        assert config.modalities == ["text", "audio"]

    def test_update_ignores_unknown_keys(self):
        config = SessionConfig()
        changed = config.update({"nonexistent_field": 42})
        assert "nonexistent_field" not in changed
        assert not hasattr(config, "nonexistent_field")

    def test_to_dict_returns_all_fields(self):
        config = SessionConfig()
        d = config.to_dict()
        expected_keys = {
            "model",
            "modalities",
            "voice",
            "input_audio_format",
            "output_audio_format",
            "turn_detection",
            "max_response_output_tokens",
            "temperature",
            "tools",
        }
        assert set(d.keys()) == expected_keys


# ── TestConversationItem ──


class TestConversationItem:
    """Tests for ConversationItem creation and serialization."""

    def test_creation_and_to_dict(self):
        item = ConversationItem(
            item_id="item_abc",
            item_type="message",
            role="user",
            content=[{"type": "text", "text": "hello world"}],
        )
        item.status = "completed"
        d = item.to_dict()
        assert d["id"] == "item_abc"
        assert d["type"] == "message"
        assert d["role"] == "user"
        assert d["status"] == "completed"
        assert d["content"] == [{"type": "text", "text": "hello world"}]

    def test_status_defaults_to_incomplete(self):
        item = ConversationItem("item_x", "message")
        assert item.status == "incomplete"

    def test_to_dict_omits_role_when_none(self):
        item = ConversationItem("item_y", "function_call")
        d = item.to_dict()
        assert "role" not in d

    def test_to_dict_omits_content_when_empty(self):
        item = ConversationItem("item_z", "message", role="user")
        d = item.to_dict()
        assert "content" not in d


# ── TestConversation ──


class TestConversation:
    """Tests for Conversation add_item and get_item."""

    def test_add_and_get_item(self):
        conv = Conversation("conv_1")
        item = ConversationItem("item_1", "message", role="user")
        conv.add_item(item)
        assert conv.get_item("item_1") is item
        assert len(conv.items) == 1

    def test_get_item_returns_none_for_unknown(self):
        conv = Conversation("conv_2")
        assert conv.get_item("does_not_exist") is None

    def test_add_multiple_items(self):
        conv = Conversation("conv_3")
        item_a = ConversationItem("a", "message", role="user")
        item_b = ConversationItem("b", "message", role="assistant")
        conv.add_item(item_a)
        conv.add_item(item_b)
        assert len(conv.items) == 2
        assert conv.get_item("a") is item_a
        assert conv.get_item("b") is item_b


# ── TestBuildMessages ──


class TestBuildMessages:
    """Tests for RealtimeSession._build_messages."""

    def _make_session_with_conversation(self) -> RealtimeSession:
        session = _make_session()
        session.conversation = Conversation("conv_test")
        return session

    def test_builds_messages_from_conversation_items(self):
        session = self._make_session_with_conversation()
        user_item = ConversationItem(
            "item_1", "message", role="user",
            content=[{"type": "text", "text": "Hello"}],
        )
        user_item.status = "completed"
        session.conversation.add_item(user_item)

        messages = session._build_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_skips_items_without_role(self):
        session = self._make_session_with_conversation()
        # item_type is "message" but role is None
        item = ConversationItem("item_1", "message", role=None)
        session.conversation.add_item(item)

        messages = session._build_messages()
        assert len(messages) == 0

    def test_joins_multiple_text_content_parts(self):
        session = self._make_session_with_conversation()
        item = ConversationItem(
            "item_1", "message", role="user",
            content=[
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        )
        session.conversation.add_item(item)

        messages = session._build_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "Part one\nPart two"

    def test_skips_non_message_item_types(self):
        session = self._make_session_with_conversation()
        item = ConversationItem("item_1", "function_call", role="user")
        session.conversation.add_item(item)

        messages = session._build_messages()
        assert len(messages) == 0

    def test_handles_string_content(self):
        session = self._make_session_with_conversation()
        item = ConversationItem(
            "item_1", "message", role="user",
            content=["plain string content"],
        )
        session.conversation.add_item(item)

        messages = session._build_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "plain string content"

    def test_multiple_items_in_order(self):
        session = self._make_session_with_conversation()
        user = ConversationItem(
            "u1", "message", role="user",
            content=[{"type": "text", "text": "hi"}],
        )
        asst = ConversationItem(
            "a1", "message", role="assistant",
            content=[{"type": "text", "text": "hello"}],
        )
        user2 = ConversationItem(
            "u2", "message", role="user",
            content=[{"type": "text", "text": "how are you?"}],
        )
        session.conversation.add_item(user)
        session.conversation.add_item(asst)
        session.conversation.add_item(user2)

        messages = session._build_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
