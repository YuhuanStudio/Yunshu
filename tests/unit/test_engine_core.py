"""EngineCore integration tests — oMLX EngineCore + Scheduler + OutputCollector.

Tests the full request lifecycle through the new EngineCore:
- add_request → Scheduler → BatchGenerator step → OutputCollector → stream_outputs
- Non-streaming generate
- Abort handling
- Concurrent requests (continuous batching)
- Stats tracking
"""

import asyncio

import pytest

from yunshu_engine.engine_core import EngineCore
from yunshu_engine.output_collector import RequestOutputCollector, RequestStreamState
from yunshu_engine.request import RequestOutput, RequestStatus, SamplingParams
from yunshu_engine.scheduler import Scheduler

# ── Fakes ──


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
        self.logprobs = None
        self.match_sequence = None
        self.prompt_cache = None
        self.all_tokens = None


class _FakeBatchGen:
    def __init__(self):
        self._uid_counter = 0
        self._pending = {}
        self._gen_pending = {}

    def insert(self, prompts, max_tokens=None, samplers=None, state_machines=None):
        uids = []
        for _prompt, _mt in zip(prompts, max_tokens or [128], strict=False):
            uid = self._uid_counter
            self._uid_counter += 1
            uids.append(uid)
            self._gen_pending[uid] = [
                _FakeGenResponse(uid, 0),
                _FakeGenResponse(uid, 1),
                _FakeGenResponse(uid, 2, finish_reason="stop"),
            ]
        return uids

    def next(self):
        """Return prompt responses (prefill) and empty gen responses."""
        prompt_responses = []
        for uid in list(self._gen_pending.keys()):
            prompt_responses.append(_FakeGenResponse(uid, -1))
        return prompt_responses, []

    def next_generated(self):
        """Return one generation token per request (decode step)."""
        gen_responses = []
        finished = []
        for uid, responses in self._gen_pending.items():
            if responses:
                gen_responses.append(responses.pop(0))
                if not responses:
                    finished.append(uid)
        for uid in finished:
            del self._gen_pending[uid]
        return gen_responses if gen_responses else []

    def remove(self, uids):
        for uid in uids:
            self._gen_pending.pop(uid, None)

    def close(self):
        pass


# ── Output Collector Tests ──


class TestRequestOutputCollector:
    def test_put_and_get_nowait(self):
        collector = RequestOutputCollector()
        output = RequestOutput(request_id="test", new_text="Hello")
        collector.put(output)
        result = collector.get_nowait()
        assert result is not None
        assert result.new_text == "Hello"
        assert collector.get_nowait() is None

    def test_sentinel_stops_stream(self):
        collector = RequestOutputCollector()
        collector.put(None)
        assert collector.get_nowait() is None

    def test_aggregation_merges_outputs(self):
        collector = RequestOutputCollector(aggregate=True)
        collector.put(
            RequestOutput(request_id="test", new_text="Hello", new_token_ids=[0])
        )
        collector.put(
            RequestOutput(request_id="test", new_text=" world", new_token_ids=[1])
        )
        result = collector.get_nowait()
        assert result.new_text == "Hello world"
        assert result.new_token_ids == [0, 1]

    def test_no_aggregation_replaces(self):
        collector = RequestOutputCollector(aggregate=False)
        collector.put(RequestOutput(request_id="test", new_text="Hello"))
        collector.put(RequestOutput(request_id="test", new_text=" world"))
        result = collector.get_nowait()
        assert result.new_text == " world"

    @pytest.mark.asyncio
    async def test_async_get(self):
        collector = RequestOutputCollector()

        async def _producer():
            await asyncio.sleep(0.01)
            collector.put(RequestOutput(request_id="test", new_text="Hello"))
            collector.put(None)

        task = asyncio.create_task(_producer())
        result = await collector.get()
        assert result is not None
        assert result.new_text == "Hello"
        await task


class TestRequestStreamState:
    def test_first_token_always_sends(self):
        state = RequestStreamState(stream_interval=5)
        assert state.should_send(1, finished=False) is True

    def test_sends_at_interval(self):
        state = RequestStreamState(stream_interval=3)
        # First send always goes through (sent_tokens == 0)
        assert state.should_send(1, finished=False) is True
        state.mark_sent(1)
        # Not enough tokens yet
        assert state.should_send(3, finished=False) is False
        # Enough tokens accumulated
        assert state.should_send(4, finished=False) is True
        state.mark_sent(4)
        # Finished always sends regardless
        assert state.should_send(5, finished=True) is True

    def test_finished_always_sends(self):
        state = RequestStreamState(stream_interval=100)
        state.mark_sent(0)
        assert state.should_send(5, finished=True) is True

    def test_mark_sent_updates_watermark(self):
        state = RequestStreamState(stream_interval=5)
        state.mark_sent(10)
        assert state.sent_tokens == 10


# ── Scheduler Tests ──


class TestScheduler:
    def test_add_request_to_waiting(self):
        from yunshu_engine.request import Request

        scheduler = Scheduler(None, _FakeTokenizer())
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)
        assert len(scheduler.waiting) == 1
        assert req.status == RequestStatus.WAITING

    def test_abort_request(self):
        from yunshu_engine.request import Request

        scheduler = Scheduler(None, _FakeTokenizer())
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)
        scheduler.abort_request("test-1")
        assert "test-1" in scheduler._pending_abort_ids

    def test_has_requests(self):
        from yunshu_engine.request import Request

        scheduler = Scheduler(None, _FakeTokenizer())
        assert not scheduler.has_requests()
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)
        assert scheduler.has_requests()

    def test_get_stats(self):
        scheduler = Scheduler(None, _FakeTokenizer())
        stats = scheduler.get_stats()
        assert "waiting" in stats
        assert "running" in stats
        assert "step_counter" in stats

    def test_schedule_waiting_with_mock_batch_gen(self):
        from yunshu_engine.request import Request

        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()

        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)
        scheduler._schedule_waiting()

        assert len(scheduler.waiting) == 0
        assert "test-1" in scheduler.running
        assert req.status == RequestStatus.RUNNING
        assert req.batch_uid == 0

    def test_process_responses(self):
        from yunshu_engine.request import Request

        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()

        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)
        scheduler._schedule_waiting()

        # Process first response (next_generated returns decode tokens)
        gen_resp = scheduler._batch_gen.next_generated()
        outputs = scheduler._process_responses(gen_resp)
        assert len(outputs) == 1
        assert outputs[0].new_text == "Hello"
        assert outputs[0].request_id == "test-1"

    def test_full_lifecycle_through_scheduler(self):
        from yunshu_engine.request import Request

        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()

        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)

        all_outputs = []
        for _ in range(10):
            result = scheduler.step()
            all_outputs.extend(result.outputs)
            if req.is_finished():
                break

        assert req.is_finished()
        assert req.finish_reason == "stop"
        assert any(o.new_text == "Hello" for o in all_outputs)
        assert any(o.new_text == " world" for o in all_outputs)


# ── EngineCore Tests ──


class TestEngineCore:
    @pytest.mark.asyncio
    async def test_add_request_returns_id(self):
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        req_id = await core.add_request(prompt="Hello", max_tokens=10)
        assert req_id.startswith("req-")
        assert len(core._output_collectors) == 1
        assert req_id in core._finished_events

    @pytest.mark.asyncio
    async def test_abort_request(self):
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        req_id = await core.add_request(prompt="Hello", max_tokens=10)
        await core.abort_request(req_id)
        # After abort, all per-request state is cleaned up
        assert req_id not in core._output_collectors
        assert req_id not in core._finished_events

    @pytest.mark.asyncio
    async def test_get_stats(self):
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        stats = core.get_stats()
        assert "running" in stats
        assert "scheduler_waiting" in stats

    @pytest.mark.asyncio
    async def test_messages_to_text(self):
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        text = core._messages_to_text(messages)
        assert "System:" in text
        assert "User:" in text
        assert "Assistant:" in text


# ── Integration: EngineCore + Scheduler + OutputCollector ──


class TestEngineCoreIntegration:
    @pytest.mark.asyncio
    async def test_stream_with_manual_steps(self):
        """Manual step loop simulating the engine_loop."""

        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        core.scheduler = scheduler

        # Add request via EngineCore
        req_id = await core.add_request(prompt="Hello", max_tokens=10)

        # Manually drive the scheduler steps and distribute outputs
        all_outputs = []
        for _ in range(10):
            result = scheduler.step()
            for req_output in result.outputs:
                collector = core._output_collectors.get(req_id)
                if collector:
                    collector.put(req_output)
                if req_output.finished:
                    core._signal_finished(req_id)
                all_outputs.append(req_output)
            if scheduler.running.get(req_id) is None:
                break

        # Collect from the output collector
        collector = core._output_collectors[req_id]
        collected = []
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            collected.append(output)

        assert len(collected) >= 1
        # Aggregation may merge outputs — check cumulative text
        combined = "".join(o.new_text for o in collected)
        assert "Hello" in combined

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Two requests batched together (continuous batching)."""

        scheduler = Scheduler(None, _FakeTokenizer())
        batch_gen = _FakeBatchGen()
        # Override to handle two sequences independently
        batch_gen._gen_pending.clear()
        uid_counter = [0]

        def _insert(prompts, max_tokens=None, samplers=None, state_machines=None):
            uids = []
            for _prompt, _mt in zip(prompts, max_tokens or [128], strict=False):
                uid = uid_counter[0]
                uid_counter[0] += 1
                uids.append(uid)
                batch_gen._gen_pending[uid] = [
                    _FakeGenResponse(uid, 0),
                    _FakeGenResponse(uid, 1, finish_reason="stop"),
                ]
            return uids

        batch_gen.insert = _insert
        scheduler._batch_gen = batch_gen

        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        core.scheduler = scheduler

        req_id_1 = await core.add_request(prompt="Hello", max_tokens=10)
        req_id_2 = await core.add_request(prompt="World", max_tokens=10)

        # Run scheduler steps
        for _ in range(10):
            result = scheduler.step()
            for req_output in result.outputs:
                rid = req_output.request_id
                collector = core._output_collectors.get(rid)
                if collector:
                    collector.put(req_output)
                if req_output.finished:
                    core._signal_finished(rid)
            if not scheduler.has_requests():
                break

        # Both should have completed
        for req_id in [req_id_1, req_id_2]:
            event = core._finished_events.get(req_id)
            assert event is not None
            assert event.is_set()

    @pytest.mark.asyncio
    async def test_generate_non_streaming(self):
        """Non-streaming generate waits for completion and drains collector."""

        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        core.scheduler = scheduler

        # Simulate engine loop in background
        async def _drive():
            for _ in range(20):
                result = scheduler.step()
                for req_output in result.outputs:
                    rid = req_output.request_id
                    collector = core._output_collectors.get(rid)
                    if collector:
                        collector.put(req_output)
                    if req_output.finished:
                        core._signal_finished(rid)
                if not scheduler.has_requests():
                    break
                await asyncio.sleep(0.001)

        req_id = await core.add_request(prompt="Hello", max_tokens=10)

        # Drive in parallel with generate
        drive_task = asyncio.create_task(_drive())
        # Wait for completion event
        event = core._finished_events.get(req_id)
        if event:
            await event.wait()
        await drive_task

        # Drain collector
        collector = core._output_collectors.get(req_id)
        result = None
        if collector:
            while True:
                output = collector.get_nowait()
                if output is None:
                    break
                if result is None:
                    result = output
                elif hasattr(collector, "_merge"):
                    result = collector._merge(result, output)

        assert result is not None
        assert result.output_text  # Should have accumulated text


# ── Engine with EngineCore Backend ──


class TestEngineWithEngineCore:
    @pytest.mark.asyncio
    async def test_engine_core_created_on_load_flag(self):
        """Engine with use_engine_core=True flag set."""
        from yunshu_engine.engine import Engine, EngineConfig

        engine = Engine(EngineConfig(), use_engine_core=True)
        assert engine._use_engine_core is True
        # EngineCore is only created when load() is called
        assert engine._engine_core is None

    @pytest.mark.asyncio
    async def test_engine_legacy_path(self):
        """Engine with use_engine_core=False uses legacy inline loop."""
        from yunshu_engine.engine import Engine, EngineConfig

        engine = Engine(EngineConfig(), use_engine_core=False)
        engine._model = object()
        engine._tokenizer = _FakeTokenizer()
        engine._model_name = "test-model"
        engine._model_display = "test-model"
        engine._batch_gen = _FakeBatchGen()

        assert engine._engine_core is None
        assert engine._use_engine_core is False

    @pytest.mark.asyncio
    async def test_engine_manually_injected_core(self):
        """Engine with manually injected EngineCore works."""
        from yunshu_engine.engine import Engine, EngineConfig

        engine = Engine(EngineConfig(), use_engine_core=True)
        engine._model = object()
        engine._tokenizer = _FakeTokenizer()
        engine._model_name = "test-model"
        engine._model_display = "test-model"

        # Manually create EngineCore (as load() would)
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        engine._engine_core = core

        stats = engine.get_stats()
        assert stats["engine_core"] is True
        assert "scheduler_waiting" in stats

    @pytest.mark.asyncio
    async def test_engine_add_request_delegates_to_core(self):
        """Engine.add_request delegates to EngineCore when present."""
        from yunshu_engine.engine import Engine, EngineConfig

        engine = Engine(EngineConfig(), use_engine_core=True)
        engine._model = object()
        engine._tokenizer = _FakeTokenizer()
        engine._model_name = "test-model"
        engine._model_display = "test-model"

        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        engine._engine_core = core

        state = await engine.add_request(prompt="Hello", max_tokens=10)
        assert state.request_id.startswith("req-")
        assert state.prompt_token_count == 5  # len("Hello")


# ── Helper ──


def _fake_executor():
    """Create a fake executor that runs sync functions immediately."""
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=1)


class TestSchedulerPolicyPlumbing:
    """scheduling policy now flows EngineCoreConfig → SchedulerConfig
    (was hardwired FCFS, making PRIORITY/FAIR preemption + aging unreachable)."""

    def test_default_is_fcfs(self):
        from yunshu_engine.engine_core import EngineCore
        from yunshu_engine.scheduler import SchedulingPolicy

        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        assert core.scheduler.config.policy == SchedulingPolicy.FCFS

    def test_priority_policy_plumbed(self):
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.scheduler import SchedulingPolicy

        core = EngineCore(
            None,
            _FakeTokenizer(),
            config=EngineCoreConfig(scheduler_policy="priority", aging_weight=0.25),
            executor=_fake_executor(),
        )
        assert core.scheduler.config.policy == SchedulingPolicy.PRIORITY
        assert core.scheduler.config.aging_weight == 0.25

    def test_fair_policy_plumbed(self):
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.scheduler import SchedulingPolicy

        core = EngineCore(
            None,
            _FakeTokenizer(),
            config=EngineCoreConfig(scheduler_policy="fair"),
            executor=_fake_executor(),
        )
        assert core.scheduler.config.policy == SchedulingPolicy.FAIR

    def test_unknown_policy_falls_back_to_fcfs(self):
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.scheduler import SchedulingPolicy

        core = EngineCore(
            None,
            _FakeTokenizer(),
            config=EngineCoreConfig(scheduler_policy="bogus"),
            executor=_fake_executor(),
        )
        assert core.scheduler.config.policy == SchedulingPolicy.FCFS
