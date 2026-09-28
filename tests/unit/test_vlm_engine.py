"""Tests for yunshu_engine.vlm_engine — VLM Engine."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlx_vlm")

from yunshu_engine.vlm_engine import (
    _VALIDATE_URL,
    VLMEngine,
    _VLMTextPromptCache,
)


class TestVLMEngineInit:
    def test_default_state(self):
        engine = VLMEngine("/models/test-model")
        assert engine.model_name == "test-model"
        assert engine.is_loaded is False
        assert engine.is_running is False
        assert engine.has_active_requests() is False
        assert engine.has_vision is False

    def test_model_name_nested_path(self):
        engine = VLMEngine("/home/user/models/qwen3-vl")
        assert engine.model_name == "qwen3-vl"

    def test_model_name_no_slash(self):
        engine = VLMEngine("local-model")
        assert engine.model_name == "local-model"


class TestExtractText:
    def test_string_content(self):
        assert VLMEngine._extract_text("hello") == "hello"

    def test_list_with_text_dicts(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert VLMEngine._extract_text(content) == "hello world"

    def test_list_with_strings(self):
        assert VLMEngine._extract_text(["a", "b"]) == "a b"

    def test_list_with_image_url_ignored(self):
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "http://img"}},
        ]
        assert VLMEngine._extract_text(content) == "describe"

    def test_empty_list(self):
        assert VLMEngine._extract_text([]) == ""

    def test_non_string_non_list(self):
        assert VLMEngine._extract_text(42) == "42"

    def test_none(self):
        assert VLMEngine._extract_text(None) == "None"


class TestFormatPrompt:
    def test_fallback_without_tokenizer(self):
        engine = VLMEngine("/models/test")
        engine._tokenizer = None
        result = engine._format_prompt(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
            ]
        )
        assert "User: Hello" in result
        assert "Assistant: Hi" in result
        assert "User: How are you?" in result
        assert result.endswith("Assistant:")

    def test_with_chat_template(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nHi<|im_end|>"
        )
        engine._tokenizer = mock_tokenizer
        result = engine._format_prompt([{"role": "user", "content": "Hi"}])
        assert "<|im_start|>" in result
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_chat_template_exception_falls_back(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = RuntimeError("no template")
        engine._tokenizer = mock_tokenizer
        result = engine._format_prompt([{"role": "user", "content": "Hi"}])
        assert "User: Hi" in result


class TestExtractImages:
    @pytest.mark.asyncio
    async def test_no_images(self):
        engine = VLMEngine("/models/test")
        messages = [{"role": "user", "content": "Hello"}]
        result = await engine._extract_images(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_text_only_content_list(self):
        engine = VLMEngine("/models/test")
        messages = [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
        result = await engine._extract_images(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_base64_image(self):
        engine = VLMEngine("/models/test")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            }
        ]
        result = await engine._extract_images(messages)
        assert len(result) == 1
        assert os.path.exists(result[0])
        engine._cleanup_temp_files()
        assert not os.path.exists(result[0])

    @pytest.mark.asyncio
    async def test_nonexistent_url_raises(self):
        # an image_url routed to the VLM that resolves to nothing must FAIL
        # LOUD, not be silently dropped (silent drop → the model hallucinates about an
        # image it never saw + shifts placeholder positions). Was: silently skipped.
        engine = VLMEngine("/models/test")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "/nonexistent/path.png"},
                    },
                ],
            }
        ]
        with pytest.raises(ValueError):
            await engine._extract_images(messages)

    @pytest.mark.asyncio
    async def test_empty_and_nonimage_data_uri_raise(self):
        engine = VLMEngine("/models/test")
        for bad in ("", "data:application/octet-stream;base64,QUJD", "ftp://x/y.png"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": bad}},
                    ],
                }
            ]
            with pytest.raises(ValueError):
                await engine._extract_images(messages)

    @pytest.mark.asyncio
    async def test_file_path_image(self, tmp_path, monkeypatch):
        # bare-path image access is gated by YUNSHU_MEDIA_DIR
        # (security: prevent /etc/passwd reads). Point YUNSHU_MEDIA_DIR at
        # the tmp_path so the test image is within the allow-list.
        monkeypatch.setenv("YUNSHU_MEDIA_DIR", str(tmp_path))
        engine = VLMEngine("/models/test")
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(img)}},
                ],
            }
        ]
        result = await engine._extract_images(messages)
        assert len(result) == 1
        # Path may be resolved (symlinks/realpath); compare resolved forms
        from pathlib import Path

        assert Path(result[0]).resolve() == Path(str(img)).resolve()

    @pytest.mark.asyncio
    async def test_file_path_image_blocked_outside_media_dir(
        self, tmp_path, monkeypatch
    ):
        """SECURITY: bare paths outside YUNSHU_MEDIA_DIR must be rejected."""
        # Point YUNSHU_MEDIA_DIR at a DIFFERENT directory
        media_dir = tmp_path / "media"
        media_dir.mkdir()
        monkeypatch.setenv("YUNSHU_MEDIA_DIR", str(media_dir))
        engine = VLMEngine("/models/test")
        # Try to access a file OUTSIDE the media dir
        img = tmp_path / "outside.png"
        img.write_bytes(b"\x89PNG\r\n")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(img)}},
                ],
            }
        ]
        # a blocked image now RAISES rather than being silently skipped.
        # The security goal (the path is NOT accessed) is still met — but failing
        # loudly avoids the VLM confidently answering about an image it never saw
        # (silent skip → hallucination).
        import pytest as _pytest

        with _pytest.raises(ValueError, match="YUNSHU_MEDIA_DIR"):
            await engine._extract_images(messages)


class TestCleanupTempFiles:
    def test_cleanup(self):
        engine = VLMEngine("/models/test")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name
        engine._temp_files = [tmp_path]
        engine._cleanup_temp_files()
        assert not os.path.exists(tmp_path)
        assert engine._temp_files == []

    def test_cleanup_no_temp_files(self):
        engine = VLMEngine("/models/test")
        engine._cleanup_temp_files()  # should not raise

    def test_cleanup_none_temp_files(self):
        engine = VLMEngine("/models/test")
        engine._temp_files = None
        engine._cleanup_temp_files()  # should not raise


class TestGetStats:
    def test_initial_stats(self):
        engine = VLMEngine("/models/test-model")
        stats = engine.get_stats()
        assert stats["model"] == "/models/test-model"
        assert stats["loaded"] is False
        assert stats["running"] is False
        assert stats["has_vision"] is False
        assert stats["is_vlm"] is False
        assert stats["num_requests_processed"] == 0
        assert "uptime_seconds" in stats

    def test_stats_after_requests(self):
        engine = VLMEngine("/models/test")
        engine._num_requests_processed = 5
        engine._start_time = 100.0
        stats = engine.get_stats()
        assert stats["num_requests_processed"] == 5


class TestVLMTextPromptCache:
    """Tests for _VLMTextPromptCache — LRU tokenization result cache."""

    def test_get_returns_none_on_miss(self):
        cache = _VLMTextPromptCache(max_entries=10)
        assert cache.get_token_ids("nonexistent_key") is None
        assert cache.stats["misses"] == 1

    def test_put_and_get_roundtrip(self):
        cache = _VLMTextPromptCache(max_entries=10)
        token_ids = [1, 2, 3, 4, 5]
        cache.put_token_ids("key1", token_ids)
        result = cache.get_token_ids("key1")
        assert result == token_ids
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 0

    def test_overwrite_existing_key(self):
        cache = _VLMTextPromptCache(max_entries=10)
        cache.put_token_ids("key1", [1, 2, 3])
        cache.put_token_ids("key1", [4, 5, 6])
        result = cache.get_token_ids("key1")
        assert result == [4, 5, 6]
        assert cache.stats["tokenization_entries"] == 1

    def test_lru_eviction(self):
        cache = _VLMTextPromptCache(max_entries=3)
        cache.put_token_ids("a", [1])
        cache.put_token_ids("b", [2])
        cache.put_token_ids("c", [3])
        # Cache is full (3 entries). Adding a 4th should evict "a" (LRU).
        cache.put_token_ids("d", [4])
        assert cache.get_token_ids("a") is None  # evicted
        assert cache.get_token_ids("b") == [2]  # still present
        assert cache.get_token_ids("c") == [3]
        assert cache.get_token_ids("d") == [4]
        assert cache.stats["evictions"] == 1

    def test_lru_access_promotes(self):
        """Accessing an entry should move it to the end, preventing eviction."""
        cache = _VLMTextPromptCache(max_entries=3)
        cache.put_token_ids("a", [1])
        cache.put_token_ids("b", [2])
        cache.put_token_ids("c", [3])
        # Access "a" to promote it — now "b" is LRU
        cache.get_token_ids("a")
        cache.put_token_ids("d", [4])
        # "b" should have been evicted (was LRU), "a" survives
        assert cache.get_token_ids("a") == [1]
        assert cache.get_token_ids("b") is None  # evicted
        assert cache.stats["evictions"] == 1

    def test_template_text_cache(self):
        cache = _VLMTextPromptCache(max_entries=10)
        template = "<|im_start|>user\nHi<|im_end|>"
        assert cache.get_template_text("tpl1") is None
        assert cache.stats["misses"] == 1
        cache.put_template_text("tpl1", template)
        assert cache.get_template_text("tpl1") == template
        assert cache.stats["hits"] == 1

    def test_template_cache_eviction(self):
        cache = _VLMTextPromptCache(max_entries=2)
        cache.put_template_text("t1", "text1")
        cache.put_template_text("t2", "text2")
        cache.put_template_text("t3", "text3")
        assert cache.get_template_text("t1") is None  # evicted
        assert cache.stats["evictions"] == 1

    def test_clear(self):
        cache = _VLMTextPromptCache(max_entries=10)
        cache.put_token_ids("k1", [1])
        cache.put_template_text("t1", "hello")
        cache.clear()
        assert cache.get_token_ids("k1") is None
        assert cache.get_template_text("t1") is None
        assert cache.stats["tokenization_entries"] == 0
        assert cache.stats["template_entries"] == 0

    def test_compute_messages_hash_stable(self):
        """Same messages should produce same hash."""
        msgs = [{"role": "user", "content": "hello"}]
        h1 = _VLMTextPromptCache._compute_messages_hash(msgs)
        h2 = _VLMTextPromptCache._compute_messages_hash(msgs)
        assert h1 == h2

    def test_compute_messages_hash_different_messages(self):
        """Different messages should produce different hashes."""
        h1 = _VLMTextPromptCache._compute_messages_hash(
            [{"role": "user", "content": "hello"}],
        )
        h2 = _VLMTextPromptCache._compute_messages_hash(
            [{"role": "user", "content": "world"}],
        )
        assert h1 != h2

    def test_compute_messages_hash_thinking_variation(self):
        """enable_thinking changes the hash."""
        msgs = [{"role": "user", "content": "hello"}]
        h1 = _VLMTextPromptCache._compute_messages_hash(msgs, enable_thinking=True)
        h2 = _VLMTextPromptCache._compute_messages_hash(msgs, enable_thinking=False)
        assert h1 != h2

    def test_thread_safety(self):
        """Concurrent put/get should not crash or corrupt state."""
        import threading

        cache = _VLMTextPromptCache(max_entries=100)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 50):
                    cache.put_token_ids(f"key-{i}", list(range(i)))
            except Exception as e:
                errors.append(e)

        def reader(start):
            try:
                for i in range(start, start + 50):
                    cache.get_token_ids(f"key-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(50,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(25,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"


class TestTokenizeWithCache:
    """Tests for VLMEngine._tokenize_with_cache integration."""

    def test_tokenize_with_cache_miss_then_hit(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|user|>Hi"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hi"}]

        # First call: cache miss, should call _format_prompt + encode
        ids1 = engine._tokenize_with_cache(messages)
        assert ids1.tolist() == [1, 2, 3]
        assert mock_tokenizer.encode.call_count == 1

        # Second call with same messages: cache hit, should NOT call encode again
        ids2 = engine._tokenize_with_cache(messages)
        assert ids2.tolist() == [1, 2, 3]
        # encode should still only have been called once (cache hit on second)
        assert mock_tokenizer.encode.call_count == 1

        # Verify stats
        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["tokenization_entries"] == 1

    def test_tokenize_with_cache_different_messages(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.encode.return_value = [1, 2]
        engine._tokenizer = mock_tokenizer

        msgs_a = [{"role": "user", "content": "Hello A"}]
        msgs_b = [{"role": "user", "content": "Hello B"}]

        engine._tokenize_with_cache(msgs_a)
        engine._tokenize_with_cache(msgs_b)

        # Both should be misses (different message content)
        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["misses"] == 2
        assert stats["hits"] == 0
        assert stats["tokenization_entries"] == 2

    def test_tokenize_with_cache_enable_thinking_variation(self):
        engine = VLMEngine("/models/test")
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.encode.return_value = [1, 2]
        engine._tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hi"}]

        # Same messages but different enable_thinking should be different cache keys
        engine._tokenize_with_cache(messages, enable_thinking=True)
        engine._tokenize_with_cache(messages, enable_thinking=False)

        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["misses"] == 2
        assert stats["tokenization_entries"] == 2

        # Same enable_thinking as first call should be a hit
        engine._tokenize_with_cache(messages, enable_thinking=True)
        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["hits"] == 1


class TestApplyVLMTemplateWithCache:
    """Tests for VLMEngine._apply_vlm_template_with_cache integration."""

    def test_template_cache_miss_then_hit(self):
        engine = VLMEngine("/models/test")
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "<|user|>Hi"
        engine._processor = mock_processor

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

        # First call: cache miss
        text1 = engine._apply_vlm_template_with_cache(messages)
        assert text1 == "<|user|>Hi"
        assert mock_processor.apply_chat_template.call_count == 1

        # Second call: cache hit — processor not called again
        text2 = engine._apply_vlm_template_with_cache(messages)
        assert text2 == "<|user|>Hi"
        assert mock_processor.apply_chat_template.call_count == 1

        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["template_entries"] == 1

    def test_template_cache_different_num_audios(self):
        engine = VLMEngine("/models/test")
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "template"
        engine._processor = mock_processor

        messages = [{"role": "user", "content": "Hi"}]

        # Different num_audios should produce different cache keys
        engine._apply_vlm_template_with_cache(messages, num_audios=0)
        engine._apply_vlm_template_with_cache(messages, num_audios=2)

        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["misses"] == 2
        assert stats["template_entries"] == 2

        # Same num_audios should be a hit
        engine._apply_vlm_template_with_cache(messages, num_audios=0)
        stats = engine.get_stats()["text_prompt_cache"]
        assert stats["hits"] == 1


class TestSSRFValidation:
    """_VALIDATE_URL should block private/reserved IPs and non-http(s) schemes."""

    def _make_addrinfo(self, ip: str):
        """Build a minimal socket.getaddrinfo return value for a given IP."""
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (ip, 80))]

    def test_blocks_loopback(self):
        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("127.0.0.1")):
            with pytest.raises(ValueError, match="SSRF blocked"):
                _VALIDATE_URL("http://localhost/image.png")

    def test_blocks_private_10(self):
        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("10.0.0.1")):
            with pytest.raises(ValueError, match="SSRF blocked"):
                _VALIDATE_URL("http://internal.corp/img.jpg")

    def test_blocks_private_172(self):
        with patch(
            "socket.getaddrinfo", return_value=self._make_addrinfo("172.16.0.5")
        ):
            with pytest.raises(ValueError, match="SSRF blocked"):
                _VALIDATE_URL("https://10.172.16.5/img.jpg")

    def test_blocks_private_192_168(self):
        with patch(
            "socket.getaddrinfo", return_value=self._make_addrinfo("192.168.1.1")
        ):
            with pytest.raises(ValueError, match="SSRF blocked"):
                _VALIDATE_URL("http://router.local/img.png")

    def test_blocks_link_local(self):
        with patch(
            "socket.getaddrinfo", return_value=self._make_addrinfo("169.254.169.254")
        ):
            with pytest.raises(ValueError, match="SSRF blocked"):
                _VALIDATE_URL("http://metadata.aws.internal/img.png")

    def test_blocks_ipv6_loopback(self):
        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("::1")):
            with pytest.raises(ValueError, match="SSRF blocked"):
                _VALIDATE_URL("http://[::1]/img.png")

    def test_blocks_non_http_scheme(self):
        with pytest.raises(ValueError, match="Blocked URL scheme"):
            _VALIDATE_URL("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        with pytest.raises(ValueError, match="Blocked URL scheme"):
            _VALIDATE_URL("ftp://example.com/img.png")

    def test_allows_public_ip(self):
        with patch(
            "socket.getaddrinfo", return_value=self._make_addrinfo("203.0.113.1")
        ):
            _VALIDATE_URL("https://example.com/image.jpg")  # must not raise

    def test_allows_public_ipv6(self):
        with patch(
            "socket.getaddrinfo", return_value=self._make_addrinfo("2001:db8::1")
        ):
            _VALIDATE_URL("https://example.com/image.jpg")  # must not raise

    def test_unresolvable_hostname_raises(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            with pytest.raises(ValueError, match="Cannot resolve hostname"):
                _VALIDATE_URL("http://does-not-exist.invalid/img.png")


class TestFinishVLMLoad:
    """_finish_vlm_load must call load_processor with a Path, not a string."""

    def test_processor_loaded_with_path_object(self):
        engine = VLMEngine("/models/qwen3-omni")
        engine._has_vision = True
        engine._is_vlm = True
        engine._config = {}

        mock_processor = MagicMock()
        with patch.object(VLMEngine, "_build_batch_runner", return_value=MagicMock()):
            with patch(
                "mlx_vlm.utils.load_processor", return_value=mock_processor
            ) as mock_load:
                engine._finish_vlm_load("/models/qwen3-omni")
                # Must be called with a Path, not a bare string
                call_arg = mock_load.call_args[0][0]
                assert isinstance(call_arg, Path), (
                    f"load_processor called with {type(call_arg).__name__}, expected Path; "
                    "passing a str causes TypeError: unsupported operand type(s) for /: 'str' and 'str'"
                )
                assert engine._processor is mock_processor

    def test_processor_warning_on_exception(self):
        engine = VLMEngine("/models/qwen3-omni")
        engine._has_vision = True
        engine._is_vlm = True
        engine._config = {}

        with patch.object(VLMEngine, "_build_batch_runner", return_value=MagicMock()):
            with patch("mlx_vlm.utils.load_processor", side_effect=RuntimeError("bad")):
                engine._finish_vlm_load("/models/qwen3-omni")
                # Should survive — _processor stays None
                assert engine._processor is None
