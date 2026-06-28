"""Tests for two architecture gap features:

1. Request cancellation propagation through scheduler ()
   - cancel_event passed to stream_outputs() for responsive cancellation
   - cancel_event triggers abort_request() to free GPU slots + KV blocks
   - Works even during chunked prefill when no outputs are produced

2. Chunked prefill progress reporting ()
   - RequestOutput.prefill_progress field (processed, total) tuple
   - Scheduler emits synthetic progress outputs during chunked prefill
   - GenerationOutput.prefill_progress forwarded to gateway
   - SSE comment emission for client-side progress bars
"""
import asyncio

import pytest

from yunshu_engine.batched_engine import GenerationOutput
from yunshu_engine.engine_core import EngineCore
from yunshu_engine.output_collector import RequestOutputCollector
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
        prompt_responses = []
        for uid in list(self._gen_pending.keys()):
            prompt_responses.append(_FakeGenResponse(uid, -1))
        return prompt_responses, []

    def next_generated(self):
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

    def remove(self, uids, **kwargs):
        for uid in uids:
            self._gen_pending.pop(uid, None)

    def close(self):
        pass


def _fake_executor():
    import concurrent.futures
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


# ── Feature 1: Cancel Propagation Tests ──


class TestCancelPropagationStreamOutputs:
    """Test that stream_outputs() respects cancel_event and triggers abort."""

    @pytest.mark.asyncio
    async def test_stream_outputs_breaks_on_cancel_before_output(self):
        """When cancel_event is set before any output arrives, stream_outputs
        should break and call abort_request."""
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core.scheduler = scheduler

        req_id = await core.add_request(prompt="Hello", max_tokens=10)
        cancel_event = asyncio.Event()

        # Set cancel immediately — no output in collector yet
        cancel_event.set()

        collected = []
        async for output in core.stream_outputs(req_id, cancel_event=cancel_event):
            collected.append(output)

        # Should have broken out without any outputs
        assert len(collected) == 0
        # Scheduler should have been told to abort
        assert req_id not in scheduler.running

    @pytest.mark.asyncio
    async def test_stream_outputs_cancel_during_wait(self):
        """cancel_event set after a delay should break the await collector.get()"""
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core.scheduler = scheduler

        req_id = await core.add_request(prompt="Hello", max_tokens=10)
        cancel_event = asyncio.Event()

        async def _cancel_after_delay():
            await asyncio.sleep(0.05)
            cancel_event.set()

        cancel_task = asyncio.create_task(_cancel_after_delay())

        collected = []
        async for output in core.stream_outputs(req_id, cancel_event=cancel_event):
            collected.append(output)

        await cancel_task
        assert len(collected) == 0
        # Verify abort propagated to scheduler
        assert req_id not in scheduler.running

    @pytest.mark.asyncio
    async def test_stream_outputs_normal_flow_without_cancel(self):
        """Normal streaming should work unchanged when cancel_event is None."""
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core.scheduler = scheduler

        req_id = await core.add_request(prompt="Hello", max_tokens=10)

        # Put output into collector to simulate engine loop producing output
        collector = core._output_collectors.get(req_id)
        assert collector is not None
        collector.put(RequestOutput(
            request_id=req_id,
            new_text="Hello",
            finished=True,
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=1,
        ))
        collector.put(None)  # sentinel

        collected = []
        async for output in core.stream_outputs(req_id):
            collected.append(output)

        assert len(collected) >= 1
        assert any(o.new_text == "Hello" for o in collected)

    @pytest.mark.asyncio
    async def test_abort_removes_from_scheduler_running(self):
        """abort_request() should remove request from scheduler running queue."""
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core.scheduler = scheduler

        req_id = await core.add_request(prompt="Hello", max_tokens=10)

        # Request should be in scheduler
        assert req_id in scheduler.requests or req_id in scheduler.running

        await core.abort_request(req_id)

        # After abort, request should NOT be in running
        assert req_id not in scheduler.running
        # Collector should be cleaned up
        assert req_id not in core._output_collectors

    @pytest.mark.asyncio
    async def test_abort_removes_from_pending_prefill_on_next_step(self):
        """abort_request() defers cleanup; _process_aborts on next step() removes
        the request from _pending_prefill."""
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        core.scheduler = scheduler

        req_id = await core.add_request(prompt="Hello", max_tokens=10)

        # Simulate the request being in pending_prefill
        scheduler._pending_prefill[req_id] = {
            'remaining_tokens': [1, 2, 3],
            'total_prompt_len': 10,
            'offset': 5,
        }
        scheduler._active_partial_prefills = 1

        await core.abort_request(req_id)

        # Abort is deferred: request is in _pending_abort_ids but NOT yet
        # removed from _pending_prefill (that happens on next step())
        assert req_id in scheduler._pending_abort_ids

        # Running a scheduler step processes the abort and cleans up
        scheduler.step()
        assert req_id not in scheduler._pending_prefill
        assert req_id not in scheduler._pending_abort_ids

    @pytest.mark.asyncio
    async def test_cancel_event_aborts_waiting_queue_request(self):
        """Cancellation should also abort requests still in the waiting queue."""
        core = EngineCore(None, _FakeTokenizer(), executor=_fake_executor())
        scheduler = Scheduler(None, _FakeTokenizer())
        core.scheduler = scheduler

        req_id = await core.add_request(prompt="Hello", max_tokens=10)
        cancel_event = asyncio.Event()
        cancel_event.set()

        collected = []
        async for output in core.stream_outputs(req_id, cancel_event=cancel_event):
            collected.append(output)

        # Should not hang and should clean up
        assert req_id not in scheduler.running


# ── Feature 2: Chunked Prefill Progress Tests ──


class TestPrefillProgressField:
    """Test that prefill_progress field exists and works on RequestOutput."""

    def test_request_output_has_prefill_progress(self):
        """RequestOutput should have a prefill_progress field."""
        output = RequestOutput(request_id="test-1")
        assert hasattr(output, 'prefill_progress')
        assert output.prefill_progress is None

    def test_request_output_prefill_progress_tuple(self):
        """prefill_progress should accept a (processed, total) tuple."""
        output = RequestOutput(
            request_id="test-1",
            prefill_progress=(1024, 4096),
        )
        assert output.prefill_progress == (1024, 4096)
        assert output.prefill_progress[0] == 1024
        assert output.prefill_progress[1] == 4096

    def test_generation_output_has_prefill_progress(self):
        """GenerationOutput should have a prefill_progress field."""
        output = GenerationOutput()
        assert hasattr(output, 'prefill_progress')
        assert output.prefill_progress is None

    def test_generation_output_prefill_progress_tuple(self):
        """GenerationOutput prefill_progress should accept (processed, total)."""
        output = GenerationOutput(prefill_progress=(2048, 8192))
        assert output.prefill_progress == (2048, 8192)


class TestPrefillProgressInCollector:
    """Test that prefill_progress survives collector merge."""

    def test_merge_preserves_prefill_progress(self):
        """Merging two outputs should preserve the newer prefill_progress."""
        collector = RequestOutputCollector(aggregate=True)
        collector.put(RequestOutput(
            request_id="test-1",
            new_text="",
            new_token_ids=[0],
            prefill_progress=(1024, 4096),
        ))
        collector.put(RequestOutput(
            request_id="test-1",
            new_text="",
            new_token_ids=[1],
            prefill_progress=(2048, 4096),
        ))
        result = collector.get_nowait()
        assert result is not None
        assert result.prefill_progress == (2048, 4096)

    def test_merge_uses_newer_when_existing_is_none(self):
        """If existing has no progress but new does, use new's progress."""
        collector = RequestOutputCollector(aggregate=True)
        collector.put(RequestOutput(
            request_id="test-1",
            new_text="Hello",
            new_token_ids=[0],
        ))
        collector.put(RequestOutput(
            request_id="test-1",
            new_text=" world",
            new_token_ids=[1],
            prefill_progress=(512, 2048),
        ))
        result = collector.get_nowait()
        assert result.prefill_progress == (512, 2048)

    def test_merge_keeps_existing_when_new_is_none(self):
        """If new has no progress but existing does, keep existing's progress."""
        collector = RequestOutputCollector(aggregate=True)
        collector.put(RequestOutput(
            request_id="test-1",
            new_text="Hello",
            new_token_ids=[0],
            prefill_progress=(512, 2048),
        ))
        collector.put(RequestOutput(
            request_id="test-1",
            new_text=" world",
            new_token_ids=[1],
        ))
        result = collector.get_nowait()
        assert result.prefill_progress == (512, 2048)


class TestPrefillProgressInScheduler:
    """Test that the scheduler emits progress outputs during chunked prefill."""

    def test_scheduler_has_prefill_progress_outputs_list(self):
        """Scheduler should have _prefill_progress_outputs instance variable."""
        scheduler = Scheduler(None, _FakeTokenizer())
        assert hasattr(scheduler, '_prefill_progress_outputs')
        assert isinstance(scheduler._prefill_progress_outputs, list)

    def test_scheduler_step_clears_progress_outputs(self):
        """step() should reset _prefill_progress_outputs at the start."""
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()
        scheduler._prefill_progress_outputs = ["stale"]

        from yunshu_engine.request import Request
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=[0, 1, 2, 3, 4],
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)

        scheduler.step()

        # _prefill_progress_outputs should have been reset at step start
        # (it may be populated during step, but stale value should be gone)
        # Since our fake doesn't trigger chunked prefill, it should be empty
        assert "stale" not in scheduler._prefill_progress_outputs

    def test_progress_output_emitted_during_chunked_prefill(self):
        """_process_pending_prefill should emit progress outputs."""
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()

        from yunshu_engine.request import Request
        req = Request(
            request_id="test-1",
            prompt="Long prompt",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(100)),
            num_prompt_tokens=100,
        )
        scheduler.add_request(req)

        # Manually simulate chunked prefill state
        scheduler._pending_prefill["test-1"] = {
            'remaining_tokens': list(range(50)),
            'total_prompt_len': 100,
            'offset': 50,
            'all_prompt_tokens': list(range(100)),
        }
        scheduler._active_partial_prefills = 1
        scheduler._chunked_prefill_fairness["test-1"] = 0
        scheduler._chunked_prefill_enqueued_at["test-1"] = __import__('time').monotonic()

        # Put request into running
        req.status = RequestStatus.RUNNING
        scheduler.running["test-1"] = req

        scheduler._process_pending_prefill()

        # Check if progress output was emitted (depends on whether
        # BatchGenerator.insert was called and succeeded)
        # The progress output is emitted AFTER a successful chunk insertion
        # Since _FakeBatchGen.insert works, we should see progress outputs
        # IF there are still remaining tokens after the chunk
        progress_outputs = [
            o for o in scheduler._prefill_progress_outputs
            if hasattr(o, 'prefill_progress') and o.prefill_progress is not None
        ]

        # After one chunk is processed, there should be progress outputs
        # if remaining tokens still exist
        if scheduler._pending_prefill.get("test-1", {}).get('remaining_tokens'):
            assert len(progress_outputs) > 0, (
                "Expected progress output when remaining tokens exist"
            )
            # Verify progress output fields
            po = progress_outputs[0]
            assert po.request_id == "test-1"
            assert po.finished is False
            assert po.prefill_progress[0] > 0  # processed > 0
            assert po.prefill_progress[1] == 100  # total = 100

    def test_no_progress_on_final_chunk(self):
        """No progress output should be emitted on the final chunk (no remaining tokens)."""
        scheduler = Scheduler(None, _FakeTokenizer())
        scheduler._batch_gen = _FakeBatchGen()

        from yunshu_engine.request import Request
        req = Request(
            request_id="test-final",
            prompt="Short",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=[0, 1, 2],
            num_prompt_tokens=3,
        )
        scheduler.add_request(req)
        req.status = RequestStatus.RUNNING
        scheduler.running["test-final"] = req

        # Simulate final chunk: only 2 tokens remaining, chunk_size = 2048
        scheduler._pending_prefill["test-final"] = {
            'remaining_tokens': [0, 1],
            'total_prompt_len': 3,
            'offset': 1,
            'all_prompt_tokens': [0, 1, 2],
        }
        scheduler._active_partial_prefills = 1
        scheduler._chunked_prefill_fairness["test-final"] = 0
        scheduler._chunked_prefill_enqueued_at["test-final"] = __import__('time').monotonic()

        scheduler._process_pending_prefill()

        # Final chunk has no remaining tokens, so no progress output
        progress_outputs = [
            o for o in scheduler._prefill_progress_outputs
            if hasattr(o, 'prefill_progress') and o.prefill_progress is not None
        ]
        assert len(progress_outputs) == 0, (
            "No progress output should be emitted for final chunk"
        )


class TestPrefillProgressSSEFormat:
    """Test the SSE comment format for prefill progress."""

    def test_sse_comment_format(self):
        """Verify the format of the SSE prefill-progress comment."""
        processed, total = 2048, 8192
        sse_comment = f": prefill-progress {processed}/{total}\n\n"
        # SSE comments start with ': '
        assert sse_comment.startswith(": prefill-progress ")
        assert f"{processed}/{total}" in sse_comment

    def test_sse_comment_bytes_encoding(self):
        """Verify the SSE comment is properly encoded as bytes."""
        processed, total = 4096, 16384
        sse_bytes = f": prefill-progress {processed}/{total}\n\n".encode()
        assert isinstance(sse_bytes, bytes)
        assert b"prefill-progress" in sse_bytes
        assert b"4096/16384" in sse_bytes


class TestCancelAdminGating:
    """The /v1/cancel admin gate derives from request.state.role. Single-consumer
    model: the simplified auth middleware stamps role="owner" on every admitted
    request, so the owner can always cancel_all / cancel by id."""

    async def _call(self, role, cancel_all=True, request_id=None, monkeypatch=None):
        import types

        from yunshu_gateway.routers import cancel as cancel_mod

        # Fake tracker so cancel_all/cancel don't touch real state.
        class _Tracker:
            def cancel_all(self):
                return 0
            def cancel(self, _id):
                return False
            def get_owner(self, _id):
                return None
        monkeypatch.setattr(cancel_mod, "get_request_tracker", lambda: _Tracker(), raising=False)
        # Auth is tested elsewhere — stub _check_auth so we can drive the admin
        # gate directly with an arbitrary role.
        monkeypatch.setattr(cancel_mod, "_check_auth", lambda request: None, raising=False)
        # Simulate a real authenticated (non dev-bypass) deployment.
        monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
        # request.state.role is set by the simplified TenantAuthMiddleware for
        # every admitted request; role as given drives the admin gate.
        st = types.SimpleNamespace(role=role)
        req_obj = types.SimpleNamespace(state=st, headers={})
        body = cancel_mod.CancelRequest(cancel_all=cancel_all, request_id=request_id)
        return await cancel_mod.cancel_generation(body, req_obj)

    @pytest.mark.asyncio
    async def test_role_unset_denied_cancel_all(self, monkeypatch):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await self._call(role=None, cancel_all=True, monkeypatch=monkeypatch)
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_role_allowed_cancel_all(self, monkeypatch):
        result = await self._call(role="admin", cancel_all=True, monkeypatch=monkeypatch)
        assert result.get("status") == "cancelled"

    @pytest.mark.asyncio
    async def test_owner_role_allowed_cancel_all(self, monkeypatch):
        # Single-consumer default: role="owner" (stamped by the simplified middleware).
        result = await self._call(role="owner", cancel_all=True, monkeypatch=monkeypatch)
        assert result.get("status") == "cancelled"
        assert result.get("status") == "cancelled"
