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
            e
            for e in ws.sent
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
            e for e in ws.sent if e.get("type") == "conversation.item.created"
        ]
        assert len(item_events) == 0

    @pytest.mark.asyncio
    async def test_manual_empty_commit_returns_error(self):
        """a MANUAL commit of an empty buffer returns the OpenAI
        input_audio_buffer_commit_empty error (not a silent no-op)."""
        session = _make_session()
        await session._handle_input_audio_buffer_commit({})
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        errs = [e for e in ws.sent if e.get("type") == "error"]
        assert any(
            (e.get("error") or {}).get("code") == "input_audio_buffer_commit_empty"
            for e in errs
        )

    @pytest.mark.asyncio
    async def test_vad_empty_autocommit_stays_silent(self):
        """A server_vad auto-commit (vad_trim) on empty must NOT emit an error."""
        session = _make_session()
        await session._handle_input_audio_buffer_commit({}, vad_trim=True)
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        assert not [e for e in ws.sent if e.get("type") == "error"]

    @pytest.mark.asyncio
    async def test_buffer_cleared_after_commit(self):
        session = _make_session()
        session._audio_buffer = bytearray(b"\x00\x01\x02")

        await session._handle_input_audio_buffer_commit({})

        assert len(session._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_empty_commit_returns_false(self):
        """an empty commit creates no item → returns False so server_vad
        auto-commit won't fire a phantom response."""
        session = _make_session()
        created = await session._handle_input_audio_buffer_commit({})
        assert created is False

    @pytest.mark.asyncio
    async def test_commit_without_asr_engine_returns_false(self):
        """a populated buffer with no ASR engine loaded creates no item (emits
        no_asr_engine) → returns False; the auto-commit path must NOT respond."""
        session = _make_session()
        session._audio_buffer = bytearray(b"\x00\x01\x02\x03\x04\x05")
        created = await session._handle_input_audio_buffer_commit({}, vad_trim=True)
        assert created is False
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        assert not [e for e in ws.sent if e.get("type") == "conversation.item.created"]

    @pytest.mark.asyncio
    async def test_omni_realtime_commit_without_asr_emits_no_error(self, monkeypatch):
        """With native-omni realtime ACTIVE (env on AND a Talker model loadable), a
        missing ASR engine is NORMAL: the model consumes the raw audio (stashed as
        speech-in) and answers natively. The commit must NOT emit a no_asr_engine
        error in that mode."""
        monkeypatch.setenv("YUNSHU_OMNI_MODEL", "/fake/omni")
        monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "1")
        # Simulate a genuinely omni-capable model (Talker present) without loading
        # one — the capability probe is what _omni_realtime_active() gates on.
        monkeypatch.setattr(
            "yunshu_gateway.routers.realtime._omni_speech_ready", lambda: True
        )
        session = _make_session()
        session._audio_buffer = bytearray(b"\x00\x01\x02\x03\x04\x05")
        await session._handle_input_audio_buffer_commit({}, vad_trim=True)
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        errors = [
            e
            for e in ws.sent
            if e.get("type") == "error"
            and e.get("error", {}).get("code") == "no_asr_engine"
        ]
        assert not errors, "native-omni realtime must not error on absent ASR"
        # the raw audio was stashed for the omni model to consume as speech-in
        assert session._last_user_audio is not None

    @pytest.mark.asyncio
    async def test_omni_env_on_but_no_talker_falls_back_to_cascade(self, monkeypatch):
        """B1 fallback: omni is env-enabled but the configured model has no Talker
        (probe returns False). The realtime path must behave as the ASR→LLM→TTS
        cascade — so a missing ASR engine IS an error, and the raw audio is NOT
        stashed for an omni model that can't consume it."""
        monkeypatch.setenv("YUNSHU_OMNI_MODEL", "/fake/non-omni")
        monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "1")
        # Model can't speak natively → not active → cascade semantics.
        monkeypatch.setattr(
            "yunshu_gateway.routers.realtime._omni_speech_ready", lambda: False
        )
        session = _make_session()
        session._audio_buffer = bytearray(b"\x00\x01\x02\x03\x04\x05")
        await session._handle_input_audio_buffer_commit({}, vad_trim=True)
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        errors = [
            e
            for e in ws.sent
            if e.get("type") == "error"
            and e.get("error", {}).get("code") == "no_asr_engine"
        ]
        assert errors, "cascade fallback must surface the missing-ASR error"
        # not stashed for omni — the cascade path holds no extra audio refs
        assert session._last_user_audio is None


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
            e
            for e in ws.sent
            if e.get("type")
            in (
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
        # when synthesis is invoked it must emit EXACTLY ONE terminal
        # response.audio.done even with no manager / no engine — else an SDK client
        # tracking per-turn audio state waits forever for the terminal event.
        _types = [e.get("type") for e in ws.sent]
        assert _types == ["response.audio.done"]

    @pytest.mark.asyncio
    async def test_bytes_returning_synthesize_emits_audio_deltas(self, monkeypatch):
        # Regression : AudioEngine.synthesize() returns raw WAV bytes.
        # The non-streaming fallback previously did getattr(result,'audio')/
        # result.get('audio'), both None for bytes -> audio silently dropped.
        class _BytesTTS:  # no synthesize_stream -> falls back to synthesize()
            async def synthesize(self, text, voice=None):
                return b"\x00\x01" * 2048  # 4096 raw WAV bytes

        class _Entry:
            is_loaded = True
            engine = _BytesTTS()

        class _Manager:
            def list_entries(self):
                return [_Entry()]

        import yunshu_gateway.engine as gw_engine

        monkeypatch.setattr(gw_engine, "get_model_manager", lambda: _Manager())

        session = _make_session()
        await session._synthesize_audio_response("hello", "resp_3", "item_3")
        ws: MockWebSocket = session.ws  # type: ignore[assignment]
        deltas = [
            e for e in ws.sent if e.get("type") == RealtimeEvent.RESPONSE_AUDIO_DELTA
        ]
        assert deltas, "bytes synthesize must produce audio deltas"
        # Each delta is valid base64 and decodes to non-empty audio.
        total = b"".join(base64.b64decode(e["delta"]) for e in deltas)
        assert len(total) == 4096


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
        changed = config.update(
            {
                "model": "qwen3",
                "temperature": 0.3,
                "modalities": ["text", "audio"],
            }
        )
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
            "instructions",
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

    @pytest.mark.asyncio
    async def test_cancel_closes_open_output_item(self):
        """self-audit: a cancel/barge-in must CLOSE the open output item
        (content_part.done + output_item.done) — leaving them open leaks a dangling
        in-progress item for SDK clients."""
        import asyncio as _aio

        session = _make_session()
        session._response_item_open = True
        session._cancel_event = None
        session._active_modalities = ["text"]

        async def _sleeper():
            await _aio.sleep(10)

        task = _aio.create_task(_sleeper())
        await _aio.sleep(0)  # let it start (pending, not done)
        task._response_id = "resp_1"
        task._item_id = "item_1"
        session._active_response = task

        await session._handle_response_cancel({})
        types = [e["type"] for e in session.ws.sent]
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        # and the open flag is cleared (no double-close)
        assert session._response_item_open is False

    @pytest.mark.asyncio
    async def test_cancel_after_natural_finish_does_not_duplicate_response_done(self):
        """self-audit R2: if the generation task already finished its happy
        path (emitting response.done status=completed) in the same scheduling window
        as a cancel/barge-in, the cancel handler must NOT emit a SECOND response.done
        (status=cancelled). The _response_done_emitted marker is the discriminator."""
        import asyncio as _aio

        session = _make_session()
        # Simulate the post-happy-path state: item already closed, terminal sent.
        session._response_item_open = False
        session._response_done_emitted = True
        session._cancel_event = None
        session._active_modalities = ["text"]

        async def _sleeper():
            await _aio.sleep(10)

        task = _aio.create_task(_sleeper())
        await _aio.sleep(0)  # pending, not done → enters the teardown branch
        task._response_id = "resp_1"
        task._item_id = "item_1"
        session._active_response = task

        await session._handle_response_cancel({})
        types = [e["type"] for e in session.ws.sent]
        # The guard returned early: NO cancel-terminal response.done was emitted.
        assert "response.done" not in types

    @pytest.mark.asyncio
    async def test_vad_speech_started_interrupts_active_response(self):
        """barge-in: SUSTAINED VAD speech while a response is in flight must cancel
        it (so the user isn't talked over and the new turn isn't dropped). Debounced
        on barge_in_min_ms so a one-window blip can't kill a reply — so feed >=120ms."""
        import struct
        from unittest.mock import AsyncMock, MagicMock

        session = _make_session()
        session._vad_speaking = False
        session._vad_silence_start = None
        session._handle_response_cancel = AsyncMock()
        active = MagicMock()
        active.done = MagicMock(return_value=False)
        session._active_response = active
        # A loud, SUSTAINED PCM chunk (>=120ms @24k = >=2880 samples) → barge-in.
        loud = struct.pack("<3120h", *([12000] * 3120))
        session._audio_buffer = bytearray(loud)
        await session._run_vad(loud)
        assert session._vad_speaking is True
        session._handle_response_cancel.assert_awaited_once()

    def test_builds_messages_from_conversation_items(self):
        session = self._make_session_with_conversation()
        user_item = ConversationItem(
            "item_1",
            "message",
            role="user",
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

    def test_per_response_instructions_override(self):
        """a per-response instructions override (from response.create)
        must win over the session-level instructions."""
        session = self._make_session_with_conversation()
        session.session.instructions = "You are a helpful assistant."
        user = ConversationItem(
            "u", "message", role="user", content=[{"type": "text", "text": "hi"}]
        )
        user.status = "completed"
        session.conversation.add_item(user)

        # No override → session instructions used.
        base = session._build_messages()
        assert base[0]["role"] == "system"
        assert base[0]["content"] == "You are a helpful assistant."

        # Override → per-response instructions used.
        over = session._build_messages(instructions_override="Answer in one word.")
        assert over[0]["role"] == "system"
        assert over[0]["content"] == "Answer in one word."

    def test_joins_multiple_text_content_parts(self):
        session = self._make_session_with_conversation()
        item = ConversationItem(
            "item_1",
            "message",
            role="user",
            content=[
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        )
        session.conversation.add_item(item)

        messages = session._build_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "Part one\nPart two"

    def test_skips_unknown_item_types_but_maps_function_calls(self):
        # function_call is now mapped (tool roundtrip); a genuinely unknown type
        # is still skipped.
        session = self._make_session_with_conversation()
        session.conversation.add_item(
            ConversationItem("item_0", "some_unknown_type", role="user")
        )
        session.conversation.add_item(
            ConversationItem(
                "item_1", "function_call", call_id="c1", name="f", arguments="{}"
            )
        )

        messages = session._build_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "c1"

    def test_handles_string_content(self):
        session = self._make_session_with_conversation()
        item = ConversationItem(
            "item_1",
            "message",
            role="user",
            content=["plain string content"],
        )
        session.conversation.add_item(item)

        messages = session._build_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "plain string content"

    def test_multiple_items_in_order(self):
        session = self._make_session_with_conversation()
        user = ConversationItem(
            "u1",
            "message",
            role="user",
            content=[{"type": "text", "text": "hi"}],
        )
        asst = ConversationItem(
            "a1",
            "message",
            role="assistant",
            content=[{"type": "text", "text": "hello"}],
        )
        user2 = ConversationItem(
            "u2",
            "message",
            role="user",
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
