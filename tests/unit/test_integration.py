"""Comprehensive integration tests for the Yunshu platform.

Tests the full pipeline: Engine → Gateway → SSE → OpenAI format
with both mocked BatchGenerator and real model (when available).
"""

import asyncio
import json
import time

import pytest

from yunshu_engine.engine import (
    Engine,
    EngineConfig,
    RequestOutput,
    RequestPhase,
    RequestState,
)


# ── Fake components for unit-level integration ──


class _FakeDetokenizer:
    def __init__(self):
        self._tokens = []
        self.last_segment = ""

    def reset(self):
        self._tokens = []
        self.last_segment = ""

    def add_token(self, token_id):
        text_map = {0: "Hello", 1: " world", 2: "!"}
        self.last_segment = text_map.get(token_id, f"tok{token_id}")
        self._tokens.append(token_id)

    def finalize(self):
        self.last_segment = ""


class _FakeTokenizer:
    eos_token_ids = [3]
    has_thinking = False

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(f"t{t}" for t in tokens)

    @property
    def detokenizer(self):
        return _FakeDetokenizer()


class _FakeGenResponse:
    def __init__(self, uid, token, finish_reason=None, current_state="normal"):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.current_state = current_state
        self.match_sequence = None
        self.prompt_cache = None
        self.all_tokens = None
        import mlx.core as mx
        self.logprobs = mx.zeros(4)


class _FakeBatchGen:
    def __init__(self):
        self._uid_counter = 0
        self._pending = {}

    def insert(self, prompts, max_tokens=None, samplers=None, state_machines=None):
        uids = []
        for prompt, mt in zip(prompts, max_tokens or [128]):
            uid = self._uid_counter
            self._uid_counter += 1
            uids.append(uid)
            self._pending[uid] = [
                _FakeGenResponse(uid, 0),
                _FakeGenResponse(uid, 1),
                _FakeGenResponse(uid, 2, finish_reason="stop"),
            ]
        return uids

    def next(self):
        gen_responses = []
        finished = []
        for uid, responses in self._pending.items():
            if responses:
                gen_responses.append(responses.pop(0))
                if not responses:
                    finished.append(uid)
        for uid in finished:
            del self._pending[uid]
        return [], gen_responses

    def remove(self, uids):
        for uid in uids:
            self._pending.pop(uid, None)

    def close(self):
        pass


@pytest.fixture
def engine():
    eng = Engine(EngineConfig())
    eng._model = object()
    eng._tokenizer = _FakeTokenizer()
    eng._model_name = "test-model"
    eng._batch_gen = _FakeBatchGen()
    return eng


# ── Tests ──


class TestOutputTokenTracking:
    """Verify output_token_ids are properly tracked (oMLX pattern)."""

    @pytest.mark.asyncio
    async def test_output_token_ids_tracked(self, engine):
        engine._running = True
        state = await engine.add_request(
            prompt=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        engine._schedule_waiting()
        assert state.uid is not None

        # Run all steps
        for _ in range(10):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state.finish_reason:
                break

        # Should have tracked all non-stop token IDs
        assert state.output_token_ids == [0, 1]  # Token 2 is stop, not tracked
        assert state.completion_token_count == 2  # Stop token excluded per OpenAI convention
        assert "Hello" in state.generated_text
        assert "world" in state.generated_text

    @pytest.mark.asyncio
    async def test_length_finish_stops_at_max(self, engine):
        """When max_tokens is hit, finish_reason='length'."""
        engine._running = True

        gen = engine._batch_gen
        uid = gen._uid_counter
        gen._uid_counter += 1
        gen._pending[uid] = [
            _FakeGenResponse(uid, 0),
            _FakeGenResponse(uid, 1, finish_reason="length"),
        ]

        state = RequestState(
            request_id="test-len",
            prompt_tokens=[1, 2],
            max_tokens=2,
        )
        state.detokenizer = _FakeDetokenizer()
        state.uid = uid
        state.phase = RequestPhase.DECODING
        engine._active["test-len"] = state
        engine._uid_to_req[uid] = "test-len"

        for _ in range(5):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state.finish_reason:
                break

        assert state.finish_reason == "length"
        # Both tokens tracked (length is not stop — EOS-only gets skipped)
        assert state.output_token_ids == [0, 1]


class TestEngineStats:
    """Verify engine stats tracking (oMLX pattern)."""

    def test_initial_stats(self, engine):
        stats = engine.get_stats()
        assert stats["step_counter"] == 0
        assert stats["num_requests_processed"] == 0
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_completion_tokens"] == 0
        assert stats["uptime_seconds"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_request(self, engine):
        engine._running = True
        engine._start_time = time.monotonic()

        state = await engine.add_request(prompt="test", max_tokens=10)
        engine._schedule_waiting()

        # Run steps until finished
        for _ in range(10):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state.finish_reason:
                break

        stats = engine.get_stats()
        assert stats["num_requests_processed"] == 1
        assert stats["total_prompt_tokens"] == 4  # len("test") = 4
        assert stats["total_completion_tokens"] > 0


class TestDeferredCacheClearing:
    """Verify deferred cache clearing is scheduled (oMLX #435)."""

    @pytest.mark.asyncio
    async def test_deferred_clear_scheduled(self, engine):
        engine._running = True

        state = await engine.add_request(prompt="test", max_tokens=10)
        engine._schedule_waiting()

        # Run steps until finished
        for _ in range(10):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state.finish_reason:
                break

        # Should have scheduled a deferred clear
        assert engine._deferred_clear_at is not None
        assert engine._deferred_clear_at > 0


class TestRequestOutputUsage:
    """Verify RequestOutput.usage property (OpenAI format)."""

    def test_usage_format(self):
        output = RequestOutput(
            request_id="test",
            new_text="hello",
            new_token_ids=[1],
            prompt_tokens=10,
            completion_tokens=5,
        )
        usage = output.usage
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class TestMultipleRequests:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_two_concurrent_requests(self, engine):
        engine._running = True

        state1 = await engine.add_request(prompt="test1", max_tokens=10)
        state2 = await engine.add_request(prompt="test2", max_tokens=10)

        assert engine.get_stats()["waiting"] == 2

        engine._schedule_waiting()
        assert state1.uid is not None
        assert state2.uid is not None

        # Both should be active
        assert engine.get_stats()["active"] == 2

        # Run steps until both finish
        for _ in range(20):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state1.finish_reason and state2.finish_reason:
                break

        assert state1.finish_reason == "stop"
        assert state2.finish_reason == "stop"


class TestGatewayIntegration:
    """Test gateway with engine stats."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._tokenizer = _FakeTokenizer()
        self._engine._model_name = "test-model"
        self._engine._batch_gen = _FakeBatchGen()
        set_engine(self._engine)

    def test_health_with_stats(self):
        from fastapi.testclient import TestClient
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "step_counter" in data["engine"]
        assert "num_requests_processed" in data["engine"]

    def test_models_list(self):
        from fastapi.testclient import TestClient
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.get("/v1/models")
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
