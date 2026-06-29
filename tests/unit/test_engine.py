"""Engine integration test — full pipeline with mocked BatchGenerator.

Tests the Engine's request lifecycle with the real mlx-lm API contract:
- GenerationBatch.Response has token (int), not text
- Per-request detokenizer via add_token / last_segment
- current_state from SequenceStateMachine
- finish_reason from BatchGenerator (not our own string matching)
"""

import asyncio

import pytest

from yunshu_engine.engine import (
    Engine,
    EngineConfig,
    RequestPhase,
    RequestState,
)


class _FakeDetokenizer:
    """Mock streaming detokenizer."""

    def __init__(self):
        self._tokens = []
        self._text = ""
        self.last_segment = ""

    def reset(self):
        self._tokens = []
        self._text = ""
        self.last_segment = ""

    def add_token(self, token_id):
        text_map = {0: "Hello", 1: " world", 2: "!"}
        self.last_segment = text_map.get(token_id, f"tok{token_id}")
        self._text += self.last_segment

    def finalize(self):
        self.last_segment = ""


class _FakeTokenizer:
    """Mock tokenizer with all attributes needed by engine."""

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
    """Mimics GenerationBatch.Response from mlx-lm.

    Key: has token (int), NOT text. Has current_state, match_sequence, logprobs.
    """

    def __init__(self, uid, token, finish_reason=None, current_state="normal"):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.current_state = current_state
        self.match_sequence = None
        self.prompt_cache = None
        self.all_tokens = None
        import mlx.core as mx

        self.logprobs = mx.zeros(4)  # dummy


class _FakePromptResponse:
    """Mimics PromptProcessingBatch.Response."""

    def __init__(self, uid, progress=(0, 0), end_of_segment=False, end_of_prompt=False):
        self.uid = uid
        self.progress = progress
        self.end_of_segment = end_of_segment
        self.end_of_prompt = end_of_prompt


class _FakeBatchGen:
    """Mock BatchGenerator that returns (prompt_responses, gen_responses) tuples."""

    def __init__(self):
        self._uid_counter = 0
        self._pending = {}  # uid → list of gen responses

    def insert(self, prompts, max_tokens=None, samplers=None, state_machines=None):
        uids = []
        for _prompt, _mt in zip(prompts, max_tokens or [128], strict=False):
            uid = self._uid_counter
            self._uid_counter += 1
            uids.append(uid)
            # Simulate token generation: token IDs 0, 1, 2 then stop
            self._pending[uid] = [
                _FakeGenResponse(uid, 0),  # "Hello"
                _FakeGenResponse(uid, 1),  # " world"
                _FakeGenResponse(uid, 2, finish_reason="stop"),  # "!"
            ]
        return uids

    def next(self):
        """Return (prompt_responses, gen_responses) — the real API."""
        gen_responses = []
        finished = []
        for uid, responses in self._pending.items():
            if responses:
                gen_responses.append(responses.pop(0))
                if not responses:
                    finished.append(uid)
        for uid in finished:
            del self._pending[uid]
        return [], gen_responses  # (prompt_responses, gen_responses)

    def next_generated(self):
        """Only return gen_responses (loops internally)."""
        return self.next()[1]

    def remove(self, uids):
        for uid in uids:
            self._pending.pop(uid, None)

    def close(self):
        pass


@pytest.fixture
def engine():
    """Create an engine with a fake BatchGenerator and tokenizer."""
    eng = Engine(EngineConfig())
    eng._model = object()
    eng._tokenizer = _FakeTokenizer()
    eng._model_name = "test-model"
    eng._batch_gen = _FakeBatchGen()
    return eng


class TestEngineRequestLifecycle:
    @pytest.mark.asyncio
    async def test_generate_stream(self, engine):
        """Full streaming lifecycle: add → schedule → step → output."""
        engine._running = True

        state = await engine.add_request(
            prompt=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )

        engine._schedule_waiting()
        assert state.uid is not None

        # Run step (returns tuple now)
        prompt_resp, gen_resp = engine._batch_gen.next()
        assert len(gen_resp) >= 1
        engine._distribute_responses(gen_resp)

        # Check output — should have token_text from detokenizer
        output = state.output_queue.get_nowait()
        assert output.token_text == "Hello"
        assert output.token_id == 0
        assert output.current_state == "normal"

        # Run remaining steps
        for _ in range(5):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)

        outputs = []
        while not state.output_queue.empty():
            outputs.append(state.output_queue.get_nowait())

        token_outputs = [o for o in outputs if o is not None]
        assert len(token_outputs) >= 2
        assert any(o.token_text == " world" for o in token_outputs)
        assert None in outputs  # sentinel

    @pytest.mark.asyncio
    async def test_generate_non_stream(self, engine):
        """Non-streaming generate: wait for done_event."""
        engine._running = True

        async def _run_steps():
            for _ in range(10):
                engine._schedule_waiting()
                _, gen_resp = engine._batch_gen.next()
                if gen_resp:
                    engine._distribute_responses(gen_resp)
                if not engine._active:
                    break
                await asyncio.sleep(0.01)

        state = await engine.add_request(
            prompt=[{"role": "user", "content": "test"}],
            max_tokens=10,
        )

        task = asyncio.create_task(_run_steps())
        await state.done_event.wait()
        task.cancel()

        assert state.finish_reason == "stop"
        assert "Hello" in state.generated_text

    @pytest.mark.asyncio
    async def test_abort_request(self, engine):
        engine._running = True
        state = await engine.add_request(prompt="test", max_tokens=10)
        engine._schedule_waiting()
        assert state.uid is not None

        await engine.abort_request(state.request_id)
        engine._process_aborts()
        assert state.finish_reason == "abort"

    @pytest.mark.asyncio
    async def test_abort_waiting_request(self, engine):
        state = await engine.add_request(prompt="test", max_tokens=10)
        assert state.phase == RequestPhase.WAITING
        await engine.abort_request(state.request_id)
        assert state.finish_reason == "abort"

    @pytest.mark.asyncio
    async def test_max_tokens_via_batch(self, engine):
        """finish_reason='length' from BatchGenerator when max_tokens hit."""
        engine._running = True

        gen = engine._batch_gen
        uid = gen._uid_counter
        gen._uid_counter += 1
        gen._pending[uid] = [
            _FakeGenResponse(uid, i, finish_reason=("length" if i == 2 else None))
            for i in range(3)
        ]

        state = RequestState(
            request_id="test-max",
            prompt_tokens=[1, 2, 3],
            max_tokens=3,
        )
        state.detokenizer = _FakeDetokenizer()
        state.uid = uid
        state.phase = RequestPhase.DECODING
        engine._active["test-max"] = state
        engine._uid_to_req[uid] = "test-max"

        for _ in range(10):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state.finish_reason:
                break

        assert state.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_reasoning_state(self, engine):
        """Tokens with current_state='reasoning' get routed correctly."""
        engine._running = True

        gen = engine._batch_gen
        uid = gen._uid_counter
        gen._uid_counter += 1
        gen._pending[uid] = [
            _FakeGenResponse(uid, 0, current_state="reasoning"),
            _FakeGenResponse(uid, 1, current_state="normal", finish_reason="stop"),
        ]

        state = RequestState(
            request_id="test-reason",
            prompt_tokens=[1],
            max_tokens=10,
        )
        state.detokenizer = _FakeDetokenizer()
        state.uid = uid
        state.phase = RequestPhase.DECODING
        engine._active["test-reason"] = state
        engine._uid_to_req[uid] = "test-reason"

        for _ in range(10):
            _, gen_resp = engine._batch_gen.next()
            if gen_resp:
                engine._distribute_responses(gen_resp)
            if state.finish_reason:
                break

        # Collect outputs
        outputs = []
        while not state.output_queue.empty():
            o = state.output_queue.get_nowait()
            outputs.append(o)

        reasoning_outputs = [
            o for o in outputs if o is not None and o.current_state == "reasoning"
        ]
        assert len(reasoning_outputs) >= 1
        assert reasoning_outputs[0].token_text == "Hello"


class TestEngineStats:
    def test_stats_empty(self, engine):
        stats = engine.get_stats()
        assert stats["loaded"] is True
        assert stats["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_stats_with_request(self, engine):
        await engine.add_request(prompt="test", max_tokens=10)
        stats = engine.get_stats()
        assert stats["waiting"] == 1
        assert stats["active"] == 1


class TestEngineMessagesToText:
    def test_generic_format(self):
        eng = Engine()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        text = eng._messages_to_text(messages)
        assert "System: You are helpful." in text
        assert "User: Hello" in text
        assert "Assistant:" in text


class TestResolveModelId:
    def test_exact_display_match(self):
        eng = Engine()
        eng._model = object()
        eng._model_name = "./models/Qwen3.5-9B-MLX-4bit"
        eng._model_display = "Qwen3.5-9B-MLX-4bit"
        assert eng.resolve_model_id("Qwen3.5-9B-MLX-4bit") is True

    def test_full_path_match(self):
        eng = Engine()
        eng._model = object()
        eng._model_name = "./models/Qwen3.5-9B-MLX-4bit"
        eng._model_display = "Qwen3.5-9B-MLX-4bit"
        assert eng.resolve_model_id("./models/Qwen3.5-9B-MLX-4bit") is True

    def test_case_insensitive(self):
        eng = Engine()
        eng._model = object()
        eng._model_name = "Qwen3.5-9B-MLX-4bit"
        eng._model_display = "Qwen3.5-9B-MLX-4bit"
        assert eng.resolve_model_id("qwen3.5-9b-mlx-4bit") is True

    def test_provider_prefix_stripped(self):
        eng = Engine()
        eng._model = object()
        eng._model_name = "mlx-community/Qwen3.5-9B-MLX-4bit"
        eng._model_display = "Qwen3.5-9B-MLX-4bit"
        assert eng.resolve_model_id("yunshu/Qwen3.5-9B-MLX-4bit") is True

    def test_no_match(self):
        eng = Engine()
        eng._model = object()
        eng._model_name = "Qwen3.5-9B-MLX-4bit"
        eng._model_display = "Qwen3.5-9B-MLX-4bit"
        assert eng.resolve_model_id("llama-3b") is False

    def test_not_loaded(self):
        eng = Engine()
        assert eng.resolve_model_id("anything") is False
