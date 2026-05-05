"""Tests for Yunshu SDK."""

import pytest
import json
from unittest.mock import patch, MagicMock, PropertyMock

from python.yunshu_sdk import YunshuClient, AsyncYunshuClient
from python.yunshu_sdk.chat import ChatCompletion, ChatCompletionChunk
from python.yunshu_sdk.models import Model, ModelList
from python.yunshu_sdk.audio import AudioSpeech, AudioTranscription


class TestSDKClient:
    """Test SDK client initialization."""

    def test_client_init(self):
        client = YunshuClient(base_url="http://test:8000", api_key="test-key")
        assert client._base_url == "http://test:8000"
        assert client._api_key == "test-key"
        client.close()

    def test_client_context_manager(self):
        with YunshuClient(base_url="http://test:8000") as client:
            assert client is not None

    def test_client_namespaces(self):
        client = YunshuClient(base_url="http://test:8000")
        assert hasattr(client, "chat")
        assert hasattr(client, "models")
        assert hasattr(client, "audio")
        assert hasattr(client, "admin")
        assert hasattr(client, "monitoring")
        assert hasattr(client, "realtime")
        client.close()

    def test_async_client_init(self):
        client = AsyncYunshuClient(base_url="http://test:8000")
        assert client._base_url == "http://test:8000"


class TestChatCompletion:
    """Test chat completion data classes."""

    def test_completion_parse(self):
        data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        comp = ChatCompletion(data)
        assert comp.id == "chatcmpl-abc123"
        assert comp.content == "Hello!"
        assert comp.finish_reason == "stop"
        assert comp.usage["total_tokens"] == 15

    def test_chunk_parse(self):
        data = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        chunk = ChatCompletionChunk(data)
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None

    def test_chunk_with_reasoning(self):
        data = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "thinking...", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        chunk = ChatCompletionChunk(data)
        assert chunk.delta_reasoning == "thinking..."


class TestModelTypes:
    """Test model data types."""

    def test_model(self):
        m = Model({"id": "test-model", "object": "model", "created": 0, "owned_by": "yunshu"})
        assert m.id == "test-model"
        assert repr(m) == "Model(id='test-model')"

    def test_model_list(self):
        ml = ModelList({
            "object": "list",
            "data": [
                {"id": "model-1", "object": "model"},
                {"id": "model-2", "object": "model"},
            ],
        })
        assert len(ml) == 2
        assert list(ml)[0].id == "model-1"


class TestAudioTypes:
    """Test audio data types."""

    def test_speech(self):
        speech = AudioSpeech(b"fake-audio-data", "audio/wav")
        assert speech.data == b"fake-audio-data"
        assert speech.content_type == "audio/wav"

    def test_transcription(self):
        t = AudioTranscription({
            "text": "Hello world",
            "language": "en",
            "duration": 2.5,
        })
        assert t.text == "Hello world"
        assert t.language == "en"
        assert t.duration == 2.5


class TestAdminSDK:
    """Test Admin namespace with real implementations."""

    def test_admin_has_keys_namespace(self):
        """Admin should have a keys sub-namespace."""
        client = YunshuClient(base_url="http://test:8000")
        assert hasattr(client.admin, "keys")
        client.close()

    def test_admin_has_models_namespace(self):
        """Admin should have a models sub-namespace."""
        client = YunshuClient(base_url="http://test:8000")
        assert hasattr(client.admin, "models")
        client.close()

    def test_admin_models_list(self):
        """Admin.models.list() should make GET request."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"id": "test-model"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'get', return_value=mock_resp):
            client = YunshuClient(base_url="http://test:8000")
            result = client.admin.models.list()
            assert result == [{"id": "test-model"}]
            client.close()

    def test_admin_keys_create(self):
        """Admin.keys.create() should POST with correct payload."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"key": "ys-abc123", "key_id": "k1"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'post', return_value=mock_resp) as mock_post:
            client = YunshuClient(base_url="http://test:8000")
            result = client.admin.keys.create(name="test-key", permissions=["inference"])
            assert result["key"] == "ys-abc123"
            mock_post.assert_called_once()
            client.close()

    def test_admin_keys_revoke(self):
        """Admin.keys.revoke() should DELETE the key."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"revoked": True}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'delete', return_value=mock_resp) as mock_delete:
            client = YunshuClient(base_url="http://test:8000")
            result = client.admin.keys.revoke("k1")
            assert result["revoked"] is True
            mock_delete.assert_called_once()
            client.close()

    def test_admin_config_get_scheduler(self):
        """Admin.config.get_scheduler() should GET scheduler config."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"completion_batch_size": 32}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'get', return_value=mock_resp):
            client = YunshuClient(base_url="http://test:8000")
            result = client.admin.config.get_scheduler()
            assert result["completion_batch_size"] == 32
            client.close()


class TestMonitoringSDK:
    """Test Monitoring namespace with real implementations."""

    def test_monitoring_system(self):
        """monitoring.system() should return system stats."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "cpu_usage_pct": 45.2,
            "memory_total_bytes": 192 * 1024**3,
            "gpu_cores": 76,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'get', return_value=mock_resp):
            client = YunshuClient(base_url="http://test:8000")
            result = client.monitoring.system()
            assert result["cpu_usage_pct"] == 45.2
            client.close()

    def test_monitoring_engine(self):
        """monitoring.engine() should return engine stats."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "scheduler": {"waiting": 0, "running": 2},
            "step_counter": 100,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'get', return_value=mock_resp):
            client = YunshuClient(base_url="http://test:8000")
            result = client.monitoring.engine()
            assert result["step_counter"] == 100
            client.close()

    def test_monitoring_kv_cache(self):
        """monitoring.kv_cache() should return KV tier stats."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "hot": {"usage_pct": 50.0},
            "warm": {"num_blocks": 100},
            "ssd": {"num_entries": 50},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'get', return_value=mock_resp):
            client = YunshuClient(base_url="http://test:8000")
            result = client.monitoring.kv_cache()
            assert "hot" in result
            client.close()

    def test_monitoring_timeline(self):
        """monitoring.timeline() should return time-series data."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "points": [{"t": 1, "tps": 50}, {"t": 2, "tps": 52}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, 'get', return_value=mock_resp):
            client = YunshuClient(base_url="http://test:8000")
            result = client.monitoring.timeline(duration_seconds=60)
            assert len(result) == 2
            client.close()


class TestRealtimeSDK:
    """Test Realtime namespace with real implementations."""

    def test_realtime_ws_url_conversion(self):
        """RealtimeNamespace should convert http to ws URLs."""
        from python.yunshu_sdk.realtime import RealtimeNamespace
        ns = RealtimeNamespace("http://localhost:8000")
        assert "ws://localhost:8000" == ns._base_url

    def test_realtime_wss_url_conversion(self):
        """RealtimeNamespace should convert https to wss URLs."""
        from python.yunshu_sdk.realtime import RealtimeNamespace
        ns = RealtimeNamespace("https://example.com")
        assert "wss://example.com" == ns._base_url

    def test_realtime_session_send_raises_when_closed(self):
        """RealtimeSession.send() should raise when closed."""
        import asyncio
        from python.yunshu_sdk.realtime import RealtimeSession
        session = RealtimeSession(MagicMock())
        session._closed = True
        with pytest.raises(RuntimeError, match="closed"):
            asyncio.run(session.send({"type": "text"}))

    def test_realtime_session_recv_raises_when_closed(self):
        """RealtimeSession.recv() should raise when closed."""
        import asyncio
        from python.yunshu_sdk.realtime import RealtimeSession
        session = RealtimeSession(MagicMock())
        session._closed = True
        with pytest.raises(RuntimeError, match="closed"):
            asyncio.run(session.recv())

    def test_realtime_session_on_off(self):
        """RealtimeSession should register and remove handlers."""
        from python.yunshu_sdk.realtime import RealtimeSession
        session = RealtimeSession(MagicMock())
        handler = lambda d: None
        session.on("text", handler)
        assert "text" in session._handlers
        assert len(session._handlers["text"]) == 1
        session.off("text", handler)
        assert len(session._handlers["text"]) == 0

    def test_realtime_session_off_all(self):
        """off() without handler should remove all handlers for that type."""
        from python.yunshu_sdk.realtime import RealtimeSession
        session = RealtimeSession(MagicMock())
        session.on("text", lambda d: None)
        session.on("text", lambda d: d)
        session.off("text")
        assert "text" not in session._handlers

    def test_realtime_session_context_manager(self):
        """RealtimeSession should work as async context manager."""
        import asyncio
        from python.yunshu_sdk.realtime import RealtimeSession

        async def _mock_close():
            pass

        mock_ws = MagicMock()
        mock_ws.close = _mock_close

        session = RealtimeSession(mock_ws)

        async def _test():
            async with session as s:
                assert s is session

        asyncio.run(_test())
        assert session._closed is True
