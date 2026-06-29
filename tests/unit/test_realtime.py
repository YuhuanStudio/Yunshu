"""Realtime WebSocket protocol tests."""



from yunshu_gateway.routers.realtime import (
    Conversation,
    ConversationItem,
    RealtimeEvent,
    RealtimeSession,
    SessionConfig,
    _event,
)


class TestSessionConfig:
    def test_defaults(self):
        config = SessionConfig()
        assert config.model == "default"
        assert "text" in config.modalities
        assert config.temperature == 0.7

    def test_update(self):
        config = SessionConfig()
        changed = config.update({"model": "qwen3", "temperature": 0.5})
        assert "model" in changed
        assert "temperature" in changed
        assert config.model == "qwen3"
        assert config.temperature == 0.5

    def test_max_response_output_tokens_inf_coerced(self):
        # OpenAI allows the string "inf"; it must NOT reach the engine as a str
        # max_tokens (which crashes generation). Coerced to a large finite int.
        config = SessionConfig()
        config.update({"max_response_output_tokens": "inf"})
        assert isinstance(config.max_response_output_tokens, int)
        assert config.max_response_output_tokens >= 1 << 20

    def test_max_response_output_tokens_numeric_string_coerced(self):
        config = SessionConfig()
        config.update({"max_response_output_tokens": "256"})
        assert config.max_response_output_tokens == 256

    def test_max_response_output_tokens_int_passthrough(self):
        config = SessionConfig()
        config.update({"max_response_output_tokens": 512})
        assert config.max_response_output_tokens == 512

    def test_to_dict(self):
        config = SessionConfig()
        d = config.to_dict()
        assert "model" in d
        assert "modalities" in d
        assert "temperature" in d

    def test_non_dict_turn_detection_rejected(self):
        # a non-dict turn_detection (bare string / list) must be REJECTED,
        # not stored verbatim — otherwise the next input_audio_buffer.append does
        # turn_detection.get("type") on a str → AttributeError → 500 on every append.
        config = SessionConfig()
        _default = dict(config.turn_detection)
        for bad in ("server_vad", [], 42, "null"):
            changed = config.update({"turn_detection": bad})
            assert "turn_detection" not in changed
            assert config.turn_detection == _default          # unchanged
            assert isinstance(config.turn_detection, dict)     # never a str/list

    def test_null_turn_detection_disables_vad(self):
        # null is the documented way to disable VAD — must be accepted.
        config = SessionConfig()
        changed = config.update({"turn_detection": None})
        assert "turn_detection" in changed
        assert config.turn_detection is None

    def test_valid_dict_turn_detection_accepted(self):
        config = SessionConfig()
        changed = config.update({"turn_detection": {"type": "server_vad", "threshold": 0.7}})
        assert "turn_detection" in changed
        assert config.turn_detection["threshold"] == 0.7
        # an unsupported type is still rejected (existing behavior preserved)
        changed2 = config.update({"turn_detection": {"type": "semantic_vad"}})
        assert "turn_detection" not in changed2


class TestConversation:
    def test_add_item(self):
        conv = Conversation("conv_test")
        item = ConversationItem("item_1", "message", role="user", content=[{"type": "text", "text": "hello"}])
        conv.add_item(item)
        assert len(conv.items) == 1
        assert conv.items[0].item_id == "item_1"

    def test_get_item(self):
        conv = Conversation("conv_test")
        item = ConversationItem("item_1", "message", role="user")
        conv.add_item(item)
        assert conv.get_item("item_1") is item
        assert conv.get_item("nonexistent") is None


class TestConversationItem:
    def test_to_dict(self):
        item = ConversationItem("item_1", "message", role="user", content=[{"type": "text", "text": "hi"}])
        item.status = "completed"
        d = item.to_dict()
        assert d["id"] == "item_1"
        assert d["type"] == "message"
        assert d["role"] == "user"
        assert d["status"] == "completed"
        assert len(d["content"]) == 1

    def test_default_status(self):
        item = ConversationItem("item_2", "message")
        assert item.status == "incomplete"


class TestEventBuilder:
    def test_event_has_type(self):
        e = _event(RealtimeEvent.SESSION_CREATED)
        assert e["type"] == "session.created"
        assert "event_id" in e

    def test_event_with_kwargs(self):
        e = _event(RealtimeEvent.ERROR, error={"message": "test"})
        assert e["type"] == "error"
        assert e["error"]["message"] == "test"


class TestRealtimeEventConstants:
    def test_all_events_defined(self):
        assert RealtimeEvent.SESSION_CREATED == "session.created"
        assert RealtimeEvent.RESPONSE_CREATED == "response.created"
        assert RealtimeEvent.RESPONSE_TEXT_DELTA == "response.text.delta"
        assert RealtimeEvent.RESPONSE_TEXT_DONE == "response.text.done"
        assert RealtimeEvent.RESPONSE_DONE == "response.done"
        assert RealtimeEvent.ERROR == "error"


class TestRealtimeSession:
    def test_build_messages(self):
        """Test message building from conversation items."""
        session = RealtimeSession.__new__(RealtimeSession)
        session.conversation = Conversation("conv_test")

        user_item = ConversationItem("item_1", "message", role="user",
                                     content=[{"type": "text", "text": "Hello"}])
        user_item.status = "completed"
        session.conversation.add_item(user_item)

        assistant_item = ConversationItem("item_2", "message", role="assistant",
                                          content=[{"type": "text", "text": "Hi there"}])
        assistant_item.status = "completed"
        session.conversation.add_item(assistant_item)

        messages = session._build_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there"

    def test_build_messages_maps_function_calls_and_skips_unknown(self):
        """function_call / function_call_output items are mapped into the message
        list (tool roundtrip); genuinely unknown item types are still skipped."""
        session = RealtimeSession.__new__(RealtimeSession)
        session.conversation = Conversation("conv_test")
        session.session = None

        session.conversation.add_item(ConversationItem(
            "item_1", "function_call", call_id="call_x", name="get_weather", arguments='{"city":"NYC"}'))
        session.conversation.add_item(ConversationItem(
            "item_2", "function_call_output", call_id="call_x", output="sunny"))
        session.conversation.add_item(ConversationItem("item_3", "some_unknown_type"))

        messages = session._build_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "call_x"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_x"
        assert messages[1]["content"] == "sunny"

    def test_resolve_engine_returns_none_when_no_engine(self):
        # make this deterministic. The old version asserted on whatever
        # global engine state prior tests left behind (hasattr(result,'generate')),
        # which a junk/Mock engine in the registry could fail under some orderings.
        # Patch the resolvers so we actually test the "no engine" path the name
        # promises.
        from unittest.mock import patch
        session = RealtimeSession.__new__(RealtimeSession)
        with patch("yunshu_gateway.engine.get_model_manager", return_value=None), \
             patch("yunshu_gateway.engine.get_engine", return_value=None):
            assert session._resolve_engine() is None

    def test_item_truncate_trims_assistant_audio_transcript(self):
        """conversation.item.truncate must actually trim what the model
        re-sees after a barge-in. The assistant audio reply is stored as an audio
        part (transcript + duration_ms); truncate trims the transcript
        proportional to audio_end_ms; _build_messages reads the trimmed transcript.
        (Was a silent no-op when the reply was stored text-only.)"""
        import asyncio

        from yunshu_gateway.routers.realtime import Conversation, ConversationItem

        session = RealtimeSession.__new__(RealtimeSession)
        session.conversation = Conversation("c1")
        session.session = type("X", (), {"instructions": ""})()

        async def _noop(*a, **k):
            pass

        session.send_event = _noop
        item = ConversationItem(
            item_id="a1", item_type="message", role="assistant",
            content=[{"type": "audio", "transcript": "A" * 40, "duration_ms": 2800}],
        )
        item.status = "completed"
        session.conversation.add_item(item)
        assert session._build_messages()[-1]["content"] == "A" * 40
        asyncio.run(session._handle_conversation_item_truncate(
            {"item_id": "a1", "content_index": 0, "audio_end_ms": 1400}))
        kept = len(session._build_messages()[-1]["content"])
        assert 15 <= kept <= 25  # ~half (proportional), no longer the full 40


class TestG711OutputEncoding:
    """the realtime output path was decode-only, so g711 clients got raw 24kHz
    PCM16 mislabeled as 8kHz g711 → garbage. These verify the inverse-table encoders."""

    def _R(self):
        from yunshu_gateway.routers.realtime import RealtimeSession
        return RealtimeSession

    def test_ulaw_encode_roundtrips(self):
        import math
        R = self._R()
        dec = R._get_ulaw_table()
        enc = R._get_ulaw_encode_lut()
        # encode→decode a 440Hz tone, expect G.711-grade SNR (~33-38 dB)
        sig = [int(20000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(4000)]
        recov = [dec[enc[s & 0xFFFF]] for s in sig]
        err = (sum((r - s) ** 2 for r, s in zip(recov, sig, strict=False)) / len(sig)) ** 0.5
        rms = (sum(s * s for s in sig) / len(sig)) ** 0.5
        snr = 20 * math.log10(rms / err) if err else 99
        assert snr > 28, f"ulaw round-trip SNR {snr:.1f} too low"

    def test_alaw_encode_roundtrips(self):
        import math
        R = self._R()
        dec = R._get_alaw_table()
        enc = R._get_alaw_encode_lut()
        sig = [int(20000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(4000)]
        recov = [dec[enc[s & 0xFFFF]] for s in sig]
        err = (sum((r - s) ** 2 for r, s in zip(recov, sig, strict=False)) / len(sig)) ** 0.5
        rms = (sum(s * s for s in sig) / len(sig)) ** 0.5
        snr = 20 * math.log10(rms / err) if err else 99
        assert snr > 28, f"alaw round-trip SNR {snr:.1f} too low"

    def test_output_format_dispatch(self):
        import struct
        R = self._R()
        r = R.__new__(R)

        class _S:
            output_audio_format = "g711_ulaw"
        r.session = _S()
        pcm24 = struct.pack("<720h", *([1000] * 720))  # 30ms @ 24kHz
        enc, csz = r._encode_output_audio(pcm24)
        assert csz == 160                # 20ms @ 8kHz, 1 byte/sample
        assert 230 <= len(enc) <= 245    # 720/3 ≈ 240 g711 bytes

        class _S2:
            output_audio_format = "pcm16"
        r.session = _S2()
        out, csz2 = r._encode_output_audio(pcm24)
        assert out == pcm24 and csz2 == 960   # pcm16 unchanged, 20ms @ 24kHz

    def test_g711_resample_continuous_across_chunks(self):
        """streaming g711 output must NOT drop samples or reset the
        decimation phase at chunk boundaries. Feeding N chunks whose sample
        counts aren't multiples of 3 must yield the same total output as one
        contiguous resample (= floor(total_samples/3) g711 bytes)."""
        import struct
        R = self._R()
        r = R.__new__(R)

        class _S:
            output_audio_format = "g711_ulaw"
        r.session = _S()
        r._g711_resample_remainder = b""

        # Chunk sizes deliberately not multiples of 3: 100, 101, 102 samples.
        sizes = [100, 101, 102]
        total_samples = sum(sizes)
        total_out = 0
        for sz in sizes:
            pcm = struct.pack(f"<{sz}h", *([500] * sz))
            enc, _ = r._encode_output_audio(pcm)
            total_out += len(enc)
        # Continuous resample loses at most the final <3 samples, not per-chunk.
        assert total_out == total_samples // 3, (
            f"got {total_out}, expected {total_samples // 3} "
            f"(per-chunk drop would give fewer)"
        )
