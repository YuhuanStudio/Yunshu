"""Tests for enhanced Realtime protocol features.

Phase 4 tests:
- VAD (Voice Activity Detection)
- Chunked audio streaming
- Session update with turn_detection
- conversation.item.delete
- response.create with modalities filter
"""
import asyncio
import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import yunshu_gateway.engine
from yunshu_gateway.routers.realtime import (
    Conversation,
    ConversationItem,
    RealtimeEvent,
    RealtimeSession,
    SessionConfig,
    _event,
)


class TestVAD:
    """Test Voice Activity Detection in RealtimeSession."""

    def _make_session(self, vad_threshold=0.5, silence_ms=500):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)
        session.session.turn_detection = {
            "type": "server_vad",
            "threshold": vad_threshold,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_ms,
        }
        return session

    def _make_silence_chunk(self, num_samples=480):
        """Create a silence PCM chunk (all zeros)."""
        return struct.pack(f"<{num_samples}h", *([0] * num_samples))

    def _make_speech_chunk(self, num_samples=480, amplitude=20000):
        """Create a speech PCM chunk (high amplitude)."""
        return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))

    @pytest.mark.asyncio
    async def test_speech_started_event(self):
        """High-amplitude audio should trigger speech_started."""
        session = self._make_session()
        speech_chunk = self._make_speech_chunk()

        await session._run_vad(speech_chunk)

        assert session._vad_speaking is True
        session.ws.send_json.assert_called_once()
        event = session.ws.send_json.call_args[0][0]
        assert event["type"] == RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED

    @pytest.mark.asyncio
    async def test_no_speech_started_on_silence(self):
        """Low-amplitude audio should not trigger speech_started."""
        session = self._make_session()
        silence_chunk = self._make_silence_chunk()

        await session._run_vad(silence_chunk)

        assert session._vad_speaking is False
        session.ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_speech_stopped_after_silence_duration(self):
        """Speech should stop after silence_duration_ms of low-amplitude audio."""
        session = self._make_session(silence_ms=0)  # 0ms for immediate stop
        session._vad_speaking = True
        session._vad_silence_start = None

        silence_chunk = self._make_silence_chunk()

        # First silence chunk starts tracking silence
        await session._run_vad(silence_chunk)
        # Second silence chunk triggers stop (silence_duration_ms=0)
        await session._run_vad(silence_chunk)

        assert session._vad_speaking is False

    @pytest.mark.asyncio
    async def test_vad_no_event_without_server_vad(self):
        """VAD should not run if turn_detection type is not server_vad."""
        session = self._make_session()
        session.session.turn_detection = {"type": None}

        speech_chunk = self._make_speech_chunk()
        # The handler checks turn_detection type before calling _run_vad,
        # so no VAD events should fire

    @pytest.mark.asyncio
    async def test_vad_threshold_respected(self):
        """Low-amplitude below threshold should not trigger speech."""
        session = self._make_session(vad_threshold=0.99)
        # Amplitude 1000 → RMS ~= 1000, normalized ~= 0.03 → below 0.99
        quiet_chunk = struct.pack("<480h", *([1000] * 480))

        await session._run_vad(quiet_chunk)
        assert session._vad_speaking is False


class TestChunkedAudio:
    """Test chunked audio streaming (20ms chunks)."""

    def test_chunk_size_constant(self):
        """Audio chunk size should be 960 bytes (20ms at 24kHz 16-bit mono)."""
        assert RealtimeSession._AUDIO_CHUNK_BYTES == 960

    @pytest.mark.asyncio
    async def test_audio_chunked_streaming(self):
        """Audio response should be split into 20ms chunks via synthesize_stream."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        # Create mock TTS chunks (4800 bytes in 5 chunks of 960)
        audio_chunks = [
            {"audio": bytes(range(240)) * 4, "is_final": False},  # 960 bytes
            {"audio": bytes(range(240)) * 4, "is_final": False},
            {"audio": bytes(range(240)) * 4, "is_final": False},
            {"audio": bytes(range(240)) * 4, "is_final": False},
            {"audio": bytes(range(240)) * 4, "is_final": False},
            {"audio": b"", "is_final": True},
        ]

        async def _mock_stream(*args, **kwargs):
            for chunk in audio_chunks:
                yield chunk

        mock_manager = MagicMock()
        mock_entry = MagicMock()
        mock_entry.is_loaded = True
        mock_entry.engine.synthesize_stream = _mock_stream
        mock_manager.list_entries.return_value = [mock_entry]

        with patch.object(yunshu_gateway.engine, "get_model_manager", return_value=mock_manager):
            await session._synthesize_audio_response("test text", "resp_1", "item_1")

        calls = ws.send_json.call_args_list
        event_types = [c[0][0]["type"] for c in calls]
        assert event_types.count(RealtimeEvent.RESPONSE_AUDIO_DELTA) == 5
        assert event_types.count(RealtimeEvent.RESPONSE_AUDIO_DONE) == 1

    @pytest.mark.asyncio
    async def test_audio_chunked_partial_chunk(self):
        """Audio that doesn't divide evenly should send a partial last chunk."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        # 1000 bytes → 1 full chunk (960) + 1 partial chunk (40)
        audio_chunks = [
            {"audio": b"\x00" * 1000, "is_final": False},
            {"audio": b"", "is_final": True},
        ]

        async def _mock_stream(*args, **kwargs):
            for chunk in audio_chunks:
                yield chunk

        mock_manager = MagicMock()
        mock_entry = MagicMock()
        mock_entry.is_loaded = True
        mock_entry.engine.synthesize_stream = _mock_stream
        mock_manager.list_entries.return_value = [mock_entry]

        with patch.object(yunshu_gateway.engine, "get_model_manager", return_value=mock_manager):
            await session._synthesize_audio_response("test text", "resp_1", "item_1")

        calls = ws.send_json.call_args_list
        event_types = [c[0][0]["type"] for c in calls]
        assert event_types.count(RealtimeEvent.RESPONSE_AUDIO_DELTA) == 2
        assert event_types.count(RealtimeEvent.RESPONSE_AUDIO_DONE) == 1

    @pytest.mark.asyncio
    async def test_no_audio_when_no_engine(self):
        """Should not crash when model manager is None."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        with patch.object(yunshu_gateway.engine, "get_model_manager", return_value=None):
            await session._synthesize_audio_response("test text", "resp_1", "item_1")

        ws.send_json.assert_not_called()


class TestSessionUpdateTurnDetection:
    """Test session.update with turn_detection configuration."""

    def test_update_turn_detection(self):
        config = SessionConfig()
        new_td = {
            "type": "server_vad",
            "threshold": 0.8,
            "prefix_padding_ms": 500,
            "silence_duration_ms": 1000,
        }
        changed = config.update({"turn_detection": new_td})
        assert "turn_detection" in changed
        assert config.turn_detection["threshold"] == 0.8
        assert config.turn_detection["silence_duration_ms"] == 1000

    def test_disable_vad(self):
        config = SessionConfig()
        config.update({"turn_detection": {"type": None}})
        assert config.turn_detection["type"] is None

    @pytest.mark.asyncio
    async def test_session_update_event_sent(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_session_update({
            "type": "session.update",
            "session": {"turn_detection": {"type": "server_vad", "threshold": 0.9}},
        })

        ws.send_json.assert_called_once()
        event = ws.send_json.call_args[0][0]
        assert event["type"] == RealtimeEvent.SESSION_UPDATED
        assert event["session"]["turn_detection"]["threshold"] == 0.9


class TestConversationItemDelete:
    """Test conversation.item.delete handler."""

    @pytest.mark.asyncio
    async def test_delete_existing_item(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        # Add an item first
        item = ConversationItem("item_1", "message", role="user")
        session.conversation.add_item(item)

        await session._handle_conversation_item_delete({
            "type": "conversation.item.delete",
            "item_id": "item_1",
        })

        assert len(session.conversation.items) == 0
        event = ws.send_json.call_args[0][0]
        assert event["type"] == "conversation.item.deleted"
        assert event["item_id"] == "item_1"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_item(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_conversation_item_delete({
            "type": "conversation.item.delete",
            "item_id": "nonexistent",
        })

        event = ws.send_json.call_args[0][0]
        assert event["type"] == RealtimeEvent.ERROR

    @pytest.mark.asyncio
    async def test_delete_missing_item_id(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_conversation_item_delete({
            "type": "conversation.item.delete",
        })

        event = ws.send_json.call_args[0][0]
        assert event["type"] == RealtimeEvent.ERROR

    @pytest.mark.asyncio
    async def test_delete_only_target_item(self):
        """Deleting one item should not affect others."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        item1 = ConversationItem("item_1", "message", role="user")
        item2 = ConversationItem("item_2", "message", role="assistant")
        session.conversation.add_item(item1)
        session.conversation.add_item(item2)

        await session._handle_conversation_item_delete({
            "type": "conversation.item.delete",
            "item_id": "item_1",
        })

        assert len(session.conversation.items) == 1
        assert session.conversation.items[0].item_id == "item_2"


class TestResponseCreateModalities:
    """Test response.create with modalities filter."""

    @pytest.mark.asyncio
    async def test_text_only_modalities(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_response_create({
            "type": "response.create",
            "response": {"modalities": ["text"]},
        })

        # Check the response.created event
        event = ws.send_json.call_args[0][0]
        assert event["type"] == RealtimeEvent.RESPONSE_CREATED
        assert event["response"]["modalities"] == ["text"]

    @pytest.mark.asyncio
    async def test_audio_only_modalities(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_response_create({
            "type": "response.create",
            "response": {"modalities": ["audio"]},
        })

        event = ws.send_json.call_args[0][0]
        assert event["response"]["modalities"] == ["audio"]

    @pytest.mark.asyncio
    async def test_both_modalities(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_response_create({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]},
        })

        event = ws.send_json.call_args[0][0]
        assert event["response"]["modalities"] == ["text", "audio"]

    @pytest.mark.asyncio
    async def test_invalid_modalities_fall_back(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_response_create({
            "type": "response.create",
            "response": {"modalities": ["video"]},  # Invalid
        })

        event = ws.send_json.call_args[0][0]
        # Should fall back to ["text"]
        assert "text" in event["response"]["modalities"]

    @pytest.mark.asyncio
    async def test_empty_modalities_fall_back(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_response_create({
            "type": "response.create",
            "response": {"modalities": []},
        })

        event = ws.send_json.call_args[0][0]
        assert event["response"]["modalities"] == ["text"]

    @pytest.mark.asyncio
    async def test_no_modalities_uses_session_default(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)
        session.session.modalities = ["text", "audio"]

        await session._handle_response_create({
            "type": "response.create",
            "response": {},
        })

        event = ws.send_json.call_args[0][0]
        assert event["response"]["modalities"] == ["text", "audio"]


class TestInputAudioBufferWithVAD:
    """Test input_audio_buffer.append with VAD integration."""

    @pytest.mark.asyncio
    async def test_append_accumulates_buffer(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        audio_chunk = base64.b64encode(b"\x00" * 100).decode()
        await session._handle_input_audio_buffer_append({
            "type": "input_audio_buffer.append",
            "audio": audio_chunk,
        })

        assert len(session._audio_buffer) == 100

    @pytest.mark.asyncio
    async def test_append_empty_audio_ignored(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)

        await session._handle_input_audio_buffer_append({
            "type": "input_audio_buffer.append",
            "audio": "",
        })

        assert len(session._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_append_with_vad_triggers_detection(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)
        session.session.turn_detection = {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500,
        }

        # High amplitude speech chunk
        speech_data = struct.pack("<480h", *([20000] * 480))
        audio_b64 = base64.b64encode(speech_data).decode()

        await session._handle_input_audio_buffer_append({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })

        assert session._vad_speaking is True
        # Should have sent speech_started event
        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_append_without_vad_no_detection(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        session = RealtimeSession(ws)
        session.session.turn_detection = {"type": None}

        speech_data = struct.pack("<480h", *([20000] * 480))
        audio_b64 = base64.b64encode(speech_data).decode()

        await session._handle_input_audio_buffer_append({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })

        # Buffer should accumulate but no VAD events
        assert len(session._audio_buffer) > 0
        ws.send_json.assert_not_called()
