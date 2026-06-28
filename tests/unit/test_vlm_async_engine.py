"""Tests for VLMAsyncEngineCore — concurrent VLM inference ."""

import asyncio
import time

import pytest
from python.yunshu_engine.vlm_async_engine import (
    VLMAsyncEngineCore,
    VLMRequestConfig,
    VLMStreamChunk,
    _VLMRequestState,
)

# ── Fakes ──


class FakeVLMEngine:
    """Fake VLM engine with controllable generate/generate_stream."""

    def __init__(self, text: str = "hello world", n_chunks: int = 3):
        self._text = text
        self._n_chunks = n_chunks
        self.generate_calls = []
        self.generate_stream_calls = []

    async def generate(self, *, messages, max_tokens=512, temperature=0.7,
                       top_p=1.0, top_k=0, seed=None, repetition_penalty=1.0,
                       stop=None, enable_thinking=None, frequency_penalty=0.0,
                       presence_penalty=0.0, logit_bias=None, json_schema=None,
                       **kwargs):
        self.generate_calls.append({
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return {
            "text": self._text,
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

    async def generate_stream(self, *, messages, max_tokens=512, temperature=0.7,
                               top_p=1.0, top_k=0, seed=None, repetition_penalty=1.0,
                               stop=None, enable_thinking=None, frequency_penalty=0.0,
                               presence_penalty=0.0, logit_bias=None, json_schema=None,
                               **kwargs):
        self.generate_stream_calls.append({
            "messages": messages, "max_tokens": max_tokens,
        })
        words = self._text.split()
        for i, w in enumerate(words):
            yield VLMStreamChunk(
                token_text=w + " ",
                token_id=i + 1,
                finish_reason="stop" if i == len(words) - 1 else None,
                prompt_tokens=10,
                completion_tokens=i + 1,
            )


class SlowFakeVLMEngine(FakeVLMEngine):
    """Engine that introduces delays."""

    async def generate(self, **kwargs):
        await asyncio.sleep(0.05)
        return await super().generate(**kwargs)

    async def generate_stream(self, **kwargs):
        kwargs.pop("messages")  # consume
        kwargs.pop("max_tokens")
        for i in range(3):
            await asyncio.sleep(0.02)
            yield VLMStreamChunk(
                token_text=f"tok{i} ",
                token_id=i,
                finish_reason="stop" if i == 2 else None,
            )


class FailingFakeVLMEngine(FakeVLMEngine):
    """Engine that raises on generate."""

    async def generate(self, **kwargs):
        raise RuntimeError("VLM generation failed")

    async def generate_stream(self, **kwargs):
        raise RuntimeError("VLM stream failed")


# ── Dataclass Tests ──


class TestVLMRequestConfig:
    def test_defaults(self):
        cfg = VLMRequestConfig()
        assert cfg.request_id == ""
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.7
        assert cfg.stream is False
        assert cfg.messages == []
        assert cfg.stop == []
        assert cfg.logit_bias is None
        assert cfg.json_schema is None
        assert cfg.tools is None

    def test_custom_values(self):
        cfg = VLMRequestConfig(
            request_id="vlm-abc123",
            messages=[{"role": "user", "content": "describe image"}],
            max_tokens=1024,
            temperature=0.3,
            stream=True,
            json_schema={"type": "object"},
            tools=[{"type": "function"}],
        )
        assert cfg.request_id == "vlm-abc123"
        assert cfg.max_tokens == 1024
        assert cfg.stream is True
        assert cfg.json_schema == {"type": "object"}
        assert len(cfg.tools) == 1


class TestVLMStreamChunk:
    def test_defaults(self):
        chunk = VLMStreamChunk()
        assert chunk.token_text == ""
        assert chunk.token_id == 0
        assert chunk.finish_reason is None
        assert chunk.prompt_tokens == 0
        assert chunk.completion_tokens == 0
        assert chunk.ttft_ms == 0.0

    def test_with_values(self):
        chunk = VLMStreamChunk(
            token_text="hello",
            token_id=42,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            ttft_ms=123.4,
        )
        assert chunk.token_text == "hello"
        assert chunk.token_id == 42
        assert chunk.finish_reason == "stop"
        assert chunk.ttft_ms == 123.4


class TestVLMRequestState:
    def test_creation(self):
        cfg = VLMRequestConfig(request_id="test-req")
        state = _VLMRequestState(
            config=cfg,
            output_queue=asyncio.Queue(),
            finished_event=asyncio.Event(),
            start_time=time.monotonic(),
        )
        assert state.config.request_id == "test-req"
        assert state.done is False
        assert state.output_queue.empty()
        assert not state.finished_event.is_set()


# ── Engine Lifecycle Tests ──


class TestVLMAsyncEngineCoreLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine, max_concurrent=2)
        assert not core.is_running
        await core.start()
        assert core.is_running
        await core.stop()
        assert not core.is_running

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()
        await core.start()  # no error
        assert core.is_running
        await core.stop()

    @pytest.mark.asyncio
    async def test_add_request_before_start_raises(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        with pytest.raises(RuntimeError, match="not started"):
            await core.add_request(messages=[{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_max_concurrent_from_env(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_VLM_MAX_CONCURRENT", "8")
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine, max_concurrent=4)
        assert core._max_concurrent == 8

    @pytest.mark.asyncio
    async def test_stats_initial(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        stats = core.get_stats()
        assert stats["running"] is False
        assert stats["max_concurrent"] == 4
        assert stats["total_requests"] == 0
        assert stats["active_requests"] == 0
        assert stats["completed_requests"] == 0
        assert stats["failed_requests"] == 0


# ── Non-Streaming Tests ──


class TestNonStreaming:
    @pytest.mark.asyncio
    async def test_generate_single_request(self):
        engine = FakeVLMEngine(text="apple banana cherry")
        core = VLMAsyncEngineCore(engine, max_concurrent=4)
        await core.start()

        result = await core.generate(
            messages=[{"role": "user", "content": "list fruits"}],
            max_tokens=256,
        )
        assert result is not None
        assert result.token_text == "apple banana cherry"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

        await core.stop()

    @pytest.mark.asyncio
    async def test_generate_tracks_stats(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        await core.generate(messages=[{"role": "user", "content": "hi"}])

        stats = core.get_stats()
        assert stats["total_requests"] == 1
        assert stats["completed_requests"] == 1
        assert stats["failed_requests"] == 0

        await core.stop()

    @pytest.mark.asyncio
    async def test_generate_multiple_requests(self):
        engine = FakeVLMEngine(text="response text")
        core = VLMAsyncEngineCore(engine, max_concurrent=4)
        await core.start()

        coros = [
            core.generate(messages=[{"role": "user", "content": f"msg{i}"}])
            for i in range(5)
        ]
        results = await asyncio.gather(*coros)
        assert len(results) == 5
        assert all(r is not None and r.token_text == "response text" for r in results)

        await core.stop()


# ── Streaming Tests ──


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_basic(self):
        engine = FakeVLMEngine(text="hello world foo")
        core = VLMAsyncEngineCore(engine, max_concurrent=4)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "stream test"}],
            stream=True,
        )

        chunks = []
        async for chunk in core.stream_outputs(req_id):
            chunks.append(chunk)

        assert len(chunks) == 3  # "hello ", "world ", "foo "
        full_text = "".join(c.token_text for c in chunks)
        assert "hello" in full_text
        assert "foo" in full_text
        assert chunks[-1].finish_reason == "stop"

        await core.stop()

    @pytest.mark.asyncio
    async def test_stream_ttft_on_first_chunk_only(self):
        engine = FakeVLMEngine(text="a b c")
        core = VLMAsyncEngineCore(engine)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

        chunks = []
        async for chunk in core.stream_outputs(req_id):
            chunks.append(chunk)

        # Only first chunk should have ttft > 0
        assert chunks[0].ttft_ms > 0
        assert chunks[1].ttft_ms == 0.0
        assert chunks[2].ttft_ms == 0.0

        await core.stop()

    @pytest.mark.asyncio
    async def test_stream_unknown_request_yields_nothing(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        chunks = []
        async for chunk in core.stream_outputs("nonexistent-id"):
            chunks.append(chunk)
        # Unknown request yields an error sentinel instead of silently
        # returning empty, so callers can distinguish "no data" from "error".
        assert len(chunks) == 1
        assert chunks[0].finish_reason == "error"

        await core.stop()


# ── Concurrency Tests ──


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        engine = SlowFakeVLMEngine(text="slow response")
        core = VLMAsyncEngineCore(engine, max_concurrent=2)
        await core.start()

        # Fire 4 requests — only 2 should be active at a time
        req_ids = []
        for i in range(4):
            rid = await core.add_request(
                messages=[{"role": "user", "content": f"concurrent {i}"}],
            )
            req_ids.append(rid)

        # Wait for all to finish
        for rid in req_ids:
            state = core._requests.get(rid)
            if state:
                await state.finished_event.wait()

        stats = core.get_stats()
        assert stats["completed_requests"] == 4
        assert stats["total_requests"] == 4

        await core.stop()

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self):
        engine = FakeVLMEngine(text="alpha beta gamma")
        core = VLMAsyncEngineCore(engine, max_concurrent=4)
        await core.start()

        req_ids = []
        for i in range(3):
            rid = await core.add_request(
                messages=[{"role": "user", "content": f"stream {i}"}],
                stream=True,
            )
            req_ids.append(rid)

        all_chunks = {}
        for rid in req_ids:
            chunks = []
            async for chunk in core.stream_outputs(rid):
                chunks.append(chunk)
            all_chunks[rid] = chunks

        for chunks in all_chunks.values():
            assert len(chunks) == 3
            assert chunks[-1].finish_reason == "stop"

        await core.stop()


# ── Error Handling Tests ──


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_failed_request_returns_error_chunk(self):
        engine = FailingFakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        result = await core.generate(
            messages=[{"role": "user", "content": "fail me"}],
        )
        assert result is not None
        assert result.finish_reason == "error"

        stats = core.get_stats()
        assert stats["failed_requests"] == 1
        assert stats["completed_requests"] == 0  # failures not counted as completed

        await core.stop()

    @pytest.mark.asyncio
    async def test_failed_streaming_request(self):
        engine = FailingFakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "stream fail"}],
            stream=True,
        )

        chunks = []
        async for chunk in core.stream_outputs(req_id):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].finish_reason == "error"

        await core.stop()


# ── Abort Tests ──


class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_pending_request(self):
        engine = SlowFakeVLMEngine(text="slow")
        core = VLMAsyncEngineCore(engine, max_concurrent=1)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "abort test"}],
            stream=True,
        )
        await core.abort_request(req_id)

        state = core._requests.get(req_id)
        if state:
            assert state.done is True

        await core.stop()

    @pytest.mark.asyncio
    async def test_abort_unknown_request_no_error(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()
        # Should not raise
        await core.abort_request("nonexistent")
        await core.stop()


# ── Stop/Cleanup Tests ──


class TestStopCleanup:
    @pytest.mark.asyncio
    async def test_stop_aborts_active_requests(self):
        engine = SlowFakeVLMEngine(text="running")
        core = VLMAsyncEngineCore(engine, max_concurrent=4)
        await core.start()

        req_ids = []
        for i in range(3):
            rid = await core.add_request(
                messages=[{"role": "user", "content": f"req{i}"}],
                stream=True,
            )
            req_ids.append(rid)

        await core.stop()
        assert not core.is_running
        stats = core.get_stats()
        assert stats["active_requests"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_removes_request(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "cleanup test"}],
        )
        assert req_id in core._requests

        # Wait for completion
        state = core._requests.get(req_id)
        if state:
            await state.finished_event.wait()

        # Manually cleanup (stream_outputs would do this)
        core._cleanup_request(req_id)
        assert req_id not in core._requests

        await core.stop()


# ── Request ID Tests ──


class TestRequestIds:
    @pytest.mark.asyncio
    async def test_request_ids_are_unique(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        ids = set()
        for i in range(20):
            rid = await core.add_request(
                messages=[{"role": "user", "content": f"uniq{i}"}],
            )
            ids.add(rid)

        assert len(ids) == 20

        # Cleanup
        for rid in ids:
            state = core._requests.get(rid)
            if state:
                await state.finished_event.wait()
            core._cleanup_request(rid)

        await core.stop()

    @pytest.mark.asyncio
    async def test_request_id_has_vlm_prefix(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        rid = await core.add_request(
            messages=[{"role": "user", "content": "prefix test"}],
        )
        assert rid.startswith("vlm-")

        state = core._requests.get(rid)
        if state:
            await state.finished_event.wait()
        core._cleanup_request(rid)
        await core.stop()


# ── Token Counting Tests ──


class TestTokenCounting:
    @pytest.mark.asyncio
    async def test_non_streaming_estimates_tokens(self):
        engine = FakeVLMEngine(text="a" * 100)  # 100 chars → ~25 tokens
        core = VLMAsyncEngineCore(engine)
        await core.start()

        await core.generate(messages=[{"role": "user", "content": "count"}])

        stats = core.get_stats()
        assert stats["total_tokens_generated"] > 0

        await core.stop()

    @pytest.mark.asyncio
    async def test_streaming_counts_per_chunk(self):
        engine = FakeVLMEngine(text="one two three")
        core = VLMAsyncEngineCore(engine)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        async for _ in core.stream_outputs(req_id):
            pass

        stats = core.get_stats()
        assert stats["total_tokens_generated"] == 3  # 3 chunks

        await core.stop()


# ── Full Parameter Passthrough Tests ──


class TestParameterPassthrough:
    @pytest.mark.asyncio
    async def test_all_params_passed_non_streaming(self):
        engine = FakeVLMEngine()
        core = VLMAsyncEngineCore(engine)
        await core.start()

        await core.generate(
            messages=[{"role": "user", "content": "params"}],
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            seed=42,
            repetition_penalty=1.2,
            stop=["END"],
            enable_thinking=True,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            logit_bias={100: -1.0},
            json_schema={"type": "string"},
        )

        assert len(engine.generate_calls) == 1
        call = engine.generate_calls[0]
        assert call["max_tokens"] == 2048
        assert call["temperature"] == 0.1

        await core.stop()

    @pytest.mark.asyncio
    async def test_all_params_passed_streaming(self):
        engine = FakeVLMEngine(text="streaming params")
        core = VLMAsyncEngineCore(engine)
        await core.start()

        req_id = await core.add_request(
            messages=[{"role": "user", "content": "params"}],
            max_tokens=1024,
            temperature=0.5,
            top_p=0.8,
            top_k=10,
            seed=123,
            repetition_penalty=1.5,
            stop=["STOP"],
            enable_thinking=False,
            stream=True,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            logit_bias={200: 2.0},
            json_schema={"type": "object"},
        )

        async for _ in core.stream_outputs(req_id):
            pass

        assert len(engine.generate_stream_calls) == 1
        call = engine.generate_stream_calls[0]
        assert call["max_tokens"] == 1024

        await core.stop()
