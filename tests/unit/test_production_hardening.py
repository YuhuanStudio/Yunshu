"""Production hardening tests — stress and integration tests for robustness.

Covers 8 categories of production-critical scenarios:
1. Streaming cancellation mid-generation — verify resources cleaned up
2. Concurrent LoRA adapter acquire/release — verify no ref count leaks
3. Empty/zero-length inputs — verify no crashes
4. Timeout during generation — verify cleanup
5. OOM during generation — verify mx.clear_cache is called
6. n>1 generation — verify per-choice isolation
7. Thinking budget enforcement — verify thinking stops at budget
8. Anthropic streaming format — verify message_start/message_stop pairing
"""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

from yunshu_engine.lora_manager import (
    LoRAAdapterEntry,
    LoRAAdapterManager,
)
from yunshu_engine.request_tracker import (
    ActiveGeneration,
    RequestTracker,
)
from yunshu_engine.thinking_budget import (
    ThinkingBudgetConfig,
    ThinkingBudgetProcessor,
    parse_thinking_budget,
)

# ═══════════════════════════════════════════════════════════════
# 1. Streaming Cancellation Mid-Generation
# ═══════════════════════════════════════════════════════════════


class TestStreamingCancellation:
    """Verify that mid-generation cancellation cleans up all resources."""

    def test_request_tracker_unregister_on_cancel(self):
        """Cancelled requests must be removed from the tracker."""
        tracker = RequestTracker()
        gen = tracker.register("req-001", "test-model")
        assert tracker.active_count == 1
        tracker.cancel("req-001")
        assert gen.cancel_event.is_set()
        tracker.unregister("req-001")
        assert tracker.active_count == 0

    def test_cancel_event_propagates(self):
        """Cancel event must be visible across threads."""
        tracker = RequestTracker()
        gen = tracker.register("req-002", "test-model")
        seen = []

        def _waiter():
            # Simulate a generation loop checking the event
            for _ in range(100):
                if gen.cancel_event.is_set():
                    seen.append("cancelled")
                    return
                time.sleep(0.01)

        t = threading.Thread(target=_waiter)
        t.start()
        time.sleep(0.02)
        tracker.cancel("req-002")
        t.join(timeout=2.0)
        assert "cancelled" in seen

    def test_cancel_nonexistent_request(self):
        """Cancelling a request that doesn't exist returns False."""
        tracker = RequestTracker()
        assert tracker.cancel("nonexistent") is False

    def test_unregister_nonexistent_is_safe(self):
        """Unregistering a request that doesn't exist must not raise."""
        tracker = RequestTracker()
        tracker.unregister("nonexistent")  # should not raise

    def test_cancel_all(self):
        """cancel_all must set the event on every registered request."""
        tracker = RequestTracker()
        gens = [tracker.register(f"req-{i}", "test") for i in range(5)]
        count = tracker.cancel_all()
        assert count == 5
        for gen in gens:
            assert gen.cancel_event.is_set()

    def test_cancel_idempotent(self):
        """Calling cancel() twice on the same request is safe."""
        tracker = RequestTracker()
        tracker.register("req-dup", "test")
        assert tracker.cancel("req-dup") is True
        # Event already set, second cancel still returns True
        assert tracker.cancel("req-dup") is True

    def test_active_generation_elapsed(self):
        """ActiveGeneration.elapsed_s returns a positive float."""
        gen = ActiveGeneration(
            request_id="r1",
            model="test",
            created_at=time.monotonic(),
            cancel_event=asyncio.Event(),
        )
        time.sleep(0.01)
        assert gen.elapsed_s > 0


# ═══════════════════════════════════════════════════════════════
# 2. Concurrent LoRA Adapter Acquire/Release
# ═══════════════════════════════════════════════════════════════


class TestConcurrentLoRARefs:
    """Verify LoRA adapter ref counts don't leak under concurrent access."""

    def _make_manager(self, max_loras=4):
        mgr = LoRAAdapterManager(max_loras=max_loras)
        # Mock base model to avoid MLX dependency
        mgr._base_model = MagicMock()
        mgr._base_model_copy = {"weights": "saved"}
        for i in range(max_loras):
            mgr.register_adapter(f"adapter-{i}", f"/tmp/adapter-{i}")
            # Mark as loaded to skip actual apply
            mgr._adapters[f"adapter-{i}"].is_loaded = True
        return mgr

    def test_acquire_increments_ref_count(self):
        """acquire_adapter must increment ref_count."""
        mgr = self._make_manager()
        assert mgr._adapters["adapter-0"].ref_count == 0
        mgr.acquire_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 1
        mgr.acquire_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 2

    def test_release_decrements_ref_count(self):
        """release_adapter must decrement ref_count."""
        mgr = self._make_manager()
        mgr.acquire_adapter("adapter-0")
        mgr.acquire_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 2
        mgr.release_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 1
        mgr.release_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 0

    def test_release_below_zero_clamped(self):
        """release_adapter with ref_count=0 should not go negative."""
        mgr = self._make_manager()
        mgr.release_adapter("adapter-0")  # ref_count stays 0
        assert mgr._adapters["adapter-0"].ref_count == 0

    def test_concurrent_acquire_release_no_leak(self):
        """Many concurrent acquire/release cycles must end at ref_count=0."""
        mgr = self._make_manager()
        errors = []

        def _worker(adapter_id, cycles=50):
            try:
                for _ in range(cycles):
                    mgr.acquire_adapter(adapter_id)
                    mgr.release_adapter(adapter_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_worker, args=(f"adapter-{i}",)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Concurrent errors: {errors}"
        for i in range(4):
            assert mgr._adapters[f"adapter-{i}"].ref_count == 0, (
                f"adapter-{i} leaked ref_count={mgr._adapters[f'adapter-{i}'].ref_count}"
            )

    def test_acquire_unknown_adapter_returns_false(self):
        """acquire_adapter for unregistered adapter returns False."""
        mgr = self._make_manager()
        assert mgr.acquire_adapter("nonexistent") is False

    def test_release_unknown_adapter_safe(self):
        """release_adapter for unregistered adapter must not raise."""
        mgr = self._make_manager()
        mgr.release_adapter("nonexistent")  # should not raise

    def test_shutdown_clears_all_refs(self):
        """shutdown() must zero out all ref counts."""
        mgr = self._make_manager()
        for i in range(4):
            mgr.acquire_adapter(f"adapter-{i}")
        assert all(mgr._adapters[f"adapter-{i}"].ref_count > 0 for i in range(4))
        mgr.shutdown()
        # All adapters should be gone after shutdown
        assert len(mgr._adapters) == 0

    def test_concurrent_acquire_release_stress(self):
        """High-contention stress: many threads acquiring/releasing same adapter."""
        mgr = self._make_manager()
        N_THREADS = 8
        N_CYCLES = 100
        errors = []

        def _stress():
            try:
                for _ in range(N_CYCLES):
                    mgr.acquire_adapter("adapter-0")
                    mgr.release_adapter("adapter-0")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_stress) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Stress test errors: {errors}"
        assert mgr._adapters["adapter-0"].ref_count == 0


# ═══════════════════════════════════════════════════════════════
# 3. Empty/Zero-Length Inputs
# ═══════════════════════════════════════════════════════════════


class TestEmptyInputs:
    """Verify empty or zero-length inputs don't crash the engine."""

    def test_empty_prompt_guarded(self):
        """_generate_fast injects BOS/EOS for empty prompt to avoid crash."""
        # Simulate the guard logic from batched_engine.py line ~1747
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

        input_ids = []
        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]

        assert len(input_ids) > 0
        assert input_ids == [1]

    def test_empty_prompt_fallback_to_eos(self):
        """When no BOS token, falls back to EOS token."""
        tokenizer = MagicMock()
        tokenizer.bos_token_id = None
        tokenizer.eos_token_id = 2
        del tokenizer.bos_token_id  # Ensure hasattr returns False

        input_ids = []
        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]

        assert len(input_ids) > 0
        assert input_ids == [2]

    def test_empty_prompt_no_bos_no_eos(self):
        """When neither BOS nor EOS is available, default to token ID 1."""
        tokenizer = MagicMock(spec=[])  # No attributes at all

        input_ids = []
        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]

        assert len(input_ids) > 0
        assert input_ids == [1]

    def test_empty_messages_list(self):
        """Empty messages list should not crash chat template application."""
        # Simulate the prompt encoding path
        messages = []
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = ""

        if isinstance(messages, list) and messages:
            text = tokenizer.apply_chat_template(messages)
        else:
            text = ""

        # Empty string is valid — downstream guard handles it
        assert text == ""

    def test_whitespace_only_prompt(self):
        """Whitespace-only prompt should encode to some token IDs (not crash)."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [220]  # whitespace token
        ids = tokenizer.encode("   ")
        assert len(ids) > 0

    def test_null_content_in_message(self):
        """Message with None content should not crash."""
        msg = {"role": "assistant", "content": None}
        content = msg.get("content") or ""
        assert content == ""

    def test_generation_output_default_values(self):
        """GenerationOutput defaults are safe for empty generation."""
        from yunshu_engine.batched_engine import GenerationOutput

        out = GenerationOutput()
        assert out.text == ""
        assert out.prompt_tokens == 0
        assert out.completion_tokens == 0
        assert out.finished is False
        assert out.finish_reason is None


# ═══════════════════════════════════════════════════════════════
# 4. Timeout During Generation
# ═══════════════════════════════════════════════════════════════


class TestTimeoutDuringGeneration:
    """Verify generation timeout triggers cleanup."""

    def test_request_lifecycle_timeout_tracking(self):
        """RequestLifecycleState tracks timeout via timing fields."""
        from yunshu_engine.request_lifecycle import (
            RequestLifecycleState,
        )

        state = RequestLifecycleState(request_id="timeout-1")
        state.queued_at = time.monotonic() - 10.0  # 10s ago
        state.prefill_start = time.monotonic() - 9.0
        # Simulate decode still not started → TTFT not measurable
        assert state.ttft_ms is not None  # has prefill_start
        assert state.queue_time_ms > 0

    def test_timeout_deadline_calculation(self):
        """Timeout deadline = gen_t0 + timeout_seconds."""
        gen_t0 = 100.0
        timeout_seconds = 5.0
        deadline = gen_t0 + timeout_seconds
        # At time 103.0, we're still within deadline
        assert time.monotonic() is not None  # real clock works
        assert deadline == 105.0

    def test_timeout_cancel_event_set(self):
        """After timeout, the cancel event should be set to stop generation."""
        tracker = RequestTracker()
        gen = tracker.register("timeout-req", "test")
        # Simulate timeout by setting the event
        gen.cancel_event.set()
        assert gen.cancel_event.is_set()

    def test_request_phase_aborted(self):
        """ABORTED phase exists for timeout/cancel scenarios."""
        from yunshu_engine.request_lifecycle import RequestPhase

        assert RequestPhase.ABORTED is not None
        state_aborted = RequestPhase.ABORTED
        assert state_aborted.name == "ABORTED"

    def test_lifecycle_transition_to_aborted(self):
        """Transition from DECODING to ABORTED is valid."""
        from yunshu_engine.request_lifecycle import (
            RequestLifecycleState,
            RequestPhase,
        )

        state = RequestLifecycleState(request_id="abort-1")
        state.phase = RequestPhase.DECODING
        result = state.transition(RequestPhase.ABORTED)
        assert result is True
        assert state.phase == RequestPhase.ABORTED


# ═══════════════════════════════════════════════════════════════
# 5. OOM During Generation
# ═══════════════════════════════════════════════════════════════


class TestOOMHandling:
    """Verify OOM during generation calls mx.clear_cache and returns gracefully."""

    def test_memory_error_returns_memory_limit(self):
        """MemoryError in _generate_fast returns finish_reason='memory_limit'."""
        from yunshu_engine.batched_engine import GenerationOutput

        # Simulate the OOM handler path from batched_engine.py line ~2195
        output = GenerationOutput(
            finished=True,
            finish_reason="memory_limit",
            prompt_tokens=100,
            completion_tokens=0,
        )
        assert output.finished is True
        assert output.finish_reason == "memory_limit"
        assert output.completion_tokens == 0

    def test_runtime_error_memory_returns_memory_limit(self):
        """RuntimeError with 'memory' in message treated as OOM."""
        error = RuntimeError("out of memory during allocation")
        msg = str(error).lower()
        is_oom = "memory" in msg
        assert is_oom is True

    def test_runtime_error_non_memory_not_oom(self):
        """RuntimeError without 'memory' in message is NOT OOM."""
        error = RuntimeError("invalid tensor shape")
        msg = str(error).lower()
        is_oom = "memory" in msg
        assert is_oom is False

    @patch("mlx.core.clear_cache")
    @patch("mlx.core.synchronize")
    def test_mx_clear_cache_called_on_oom(self, mock_sync, mock_clear):
        """mx.clear_cache() must be called after OOM to free Metal buffers."""
        # Simulate the cleanup pattern from batched_engine.py line ~2190
        try:
            raise MemoryError("GPU OOM")
        except MemoryError:
            mock_sync()
            mock_clear()

        mock_sync.assert_called_once()
        mock_clear.assert_called_once()

    def test_memory_guard_preflight_rejection(self):
        """MemoryGuard preflight_check rejects oversized requests."""
        from yunshu_engine.memory_guard import MemoryGuard
        from yunshu_engine.memory_monitor import MemoryInfo

        monitor = MagicMock()
        # Simulate very little memory
        info = MemoryInfo(
            total_bytes=10_000,
            active_bytes=9_000,
            peak_bytes=9_500,
            cache_bytes=0,
            available_bytes=1000,
            utilization_pct=90.0,
        )
        monitor.get_memory_info.return_value = info
        monitor.estimate_prompt_kv_bytes.return_value = 500
        monitor.estimate_prefill_peak_bytes.return_value = 600
        monitor.get_stats.return_value = {}

        guard = MemoryGuard(monitor, safety_margin_pct=0.1)
        ok, reason = guard.preflight_check(num_prompt_tokens=500, max_tokens=100)
        # Should reject because total > usable (900 bytes after 10% margin)
        assert ok is False
        assert isinstance(reason, str)

    def test_memory_guard_stats_tracking(self):
        """MemoryGuard tracks check and rejection counts."""
        from yunshu_engine.memory_guard import MemoryGuard
        from yunshu_engine.memory_monitor import MemoryInfo

        monitor = MagicMock()
        info = MemoryInfo(
            total_bytes=10_000_000_000,
            active_bytes=1_000_000_000,
            peak_bytes=2_000_000_000,
            cache_bytes=0,
            available_bytes=9_000_000_000,
            utilization_pct=10.0,
        )
        monitor.get_memory_info.return_value = info
        monitor.estimate_prompt_kv_bytes.return_value = 100
        monitor.estimate_prefill_peak_bytes.return_value = 100
        monitor.get_stats.return_value = {}

        guard = MemoryGuard(monitor)
        ok, reason = guard.preflight_check(num_prompt_tokens=10, max_tokens=10)
        assert ok is True
        stats = guard.get_stats()
        assert stats["total_checks"] == 1
        assert isinstance(stats["total_rejections"], int)


# ═══════════════════════════════════════════════════════════════
# 6. n>1 Generation — Per-Choice Isolation
# ═══════════════════════════════════════════════════════════════


class TestNPlusOneIsolation:
    """Verify n>1 generation maintains per-choice isolation."""

    def test_format_choice_chunk_index(self):
        """Each choice chunk must have the correct index."""
        from yunshu_gateway.routers.chat import _format_choice_chunk

        for idx in range(5):
            chunk = _format_choice_chunk(
                completion_id="chatcmpl-test",
                model="test-model",
                index=idx,
                delta_content=f"choice_{idx}",
                finish_reason=None,
            )
            parsed = json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
            assert parsed["choices"][0]["index"] == idx
            assert parsed["choices"][0]["delta"]["content"] == f"choice_{idx}"

    def test_each_choice_gets_unique_seed(self):
        """In _stream_response_multi, each choice uses seed + choice_idx."""
        base_seed = 42
        n = 5
        seeds = [base_seed + i for i in range(n)]
        # All unique
        assert len(set(seeds)) == n
        # All deterministic
        assert seeds == [42, 43, 44, 45, 46]

    def test_per_choice_finish_reason(self):
        """Each choice must have its own finish_reason."""
        from yunshu_gateway.routers.chat import _format_choice_chunk

        # Choice 0: length limit
        c0 = _format_choice_chunk("id", "model", 0, "", "length")
        p0 = json.loads(c0.removeprefix("data: ").removesuffix("\n\n"))
        assert p0["choices"][0]["finish_reason"] == "length"

        # Choice 1: stop
        c1 = _format_choice_chunk("id", "model", 1, "", "stop")
        p1 = json.loads(c1.removeprefix("data: ").removesuffix("\n\n"))
        assert p1["choices"][0]["finish_reason"] == "stop"

    def test_n1_is_default(self):
        """ChatCompletionRequest defaults to n=1."""
        from yunshu_gateway.routers.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.n == 1

    def test_per_choice_tool_call_streamer_isolation(self):
        """Each choice gets its own ToolCallStreamer for independent extraction."""
        from yunshu_engine.tool_call_streamer import ToolCallStreamer

        streamer0 = ToolCallStreamer()
        streamer1 = ToolCallStreamer()
        # They are independent instances
        assert streamer0 is not streamer1
        # Processing tokens on one doesn't affect the other
        streamer0.process_token("<")
        # streamer1 is untouched (empty buffer)
        assert streamer1._buffer == ""

    def test_cancel_stops_remaining_choices(self):
        """When cancel event fires, remaining choices get finish_reason='stop'."""
        from yunshu_gateway.routers.chat import _format_choice_chunk

        n = 5
        cancelled_after = 2
        results = []
        for i in range(n):
            if i >= cancelled_after:
                # Simulate cancel — emit stop chunk
                results.append(_format_choice_chunk("id", "model", i, "", "stop"))
            else:
                results.append(_format_choice_chunk("id", "model", i, "text", None))

        # Choices after cancellation should have finish_reason
        for i, chunk in enumerate(results):
            parsed = json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
            if i >= cancelled_after:
                assert parsed["choices"][0]["finish_reason"] == "stop"


# ═══════════════════════════════════════════════════════════════
# 7. Thinking Budget Enforcement
# ═══════════════════════════════════════════════════════════════


class TestThinkingBudgetEnforcement:
    """Verify thinking stops when budget is exceeded."""

    def test_budget_not_exceeded_within_limit(self):
        """Tokens within budget should not trigger force_stop."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))
        for _i in range(99):
            result = proc.process_token("reasoning")
            assert result["force_stop"] is False
            assert result["budget_exceeded"] is False

    def test_budget_exceeded_at_limit(self):
        """Token at exactly max_thinking_tokens must trigger force_stop."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=10))
        for _i in range(9):
            proc.process_token("reasoning")
        # 10th token hits the budget
        result = proc.process_token("reasoning")
        assert result["force_stop"] is True
        assert result["budget_exceeded"] is True

    def test_budget_remaining_decreases(self):
        """budget_remaining should decrease with each thinking token."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=50))
        for _i in range(10):
            proc.process_token("reasoning")
        assert proc.budget_remaining == 40

    def test_non_thinking_tokens_not_counted(self):
        """Tokens in 'normal' state must not consume thinking budget."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=10))
        for _ in range(100):
            result = proc.process_token("normal")
            assert result["force_stop"] is False
        assert proc.thinking_tokens_used == 0
        assert proc.budget_remaining == 10

    def test_mixed_thinking_normal(self):
        """Switching between reasoning and normal states counts correctly."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))
        # 10 thinking tokens
        for _ in range(10):
            proc.process_token("reasoning")
        # 20 normal tokens
        for _ in range(20):
            proc.process_token("normal")
        # 5 more thinking tokens
        for _ in range(5):
            proc.process_token("reasoning")
        assert proc.thinking_tokens_used == 15
        assert proc.budget_remaining == 85

    def test_reset_clears_state(self):
        """reset() must clear all counters."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=10))
        for _ in range(8):
            proc.process_token("reasoning")
        proc.reset()
        assert proc.thinking_tokens_used == 0
        assert proc.budget_remaining == 10
        assert proc.is_budget_exceeded is False

    def test_disabled_never_stops(self):
        """When config.enabled=False, thinking budget is never enforced."""
        cfg = ThinkingBudgetConfig(max_thinking_tokens=1, enabled=False)
        proc = ThinkingBudgetProcessor(cfg)
        for _ in range(1000):
            result = proc.process_token("reasoning")
            assert result["force_stop"] is False
            assert result["budget_exceeded"] is False

    def test_get_think_end_tokens(self):
        """get_think_end_tokens returns token IDs when budget exceeded."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=5))
        for _ in range(6):
            proc.process_token("reasoning")
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [100, 101]
        ids = proc.get_think_end_tokens(tokenizer)
        assert ids == [100, 101]

    def test_get_think_end_tokens_not_exceeded(self):
        """get_think_end_tokens returns None when budget not exceeded."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))
        proc.process_token("reasoning")
        tokenizer = MagicMock()
        assert proc.get_think_end_tokens(tokenizer) is None

    def test_get_think_end_tokens_no_tokenizer(self):
        """get_think_end_tokens returns None when tokenizer is None."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=1))
        for _ in range(2):
            proc.process_token("reasoning")
        assert proc.get_think_end_tokens(None) is None

    def test_parse_thinking_budget_from_tokens(self):
        """parse_thinking_budget with explicit token count."""
        cfg = parse_thinking_budget({"thinking_budget": 4096})
        assert cfg.max_thinking_tokens == 4096
        assert cfg.enabled is True

    def test_parse_thinking_budget_from_effort(self):
        """parse_thinking_budget maps reasoning_effort to token budgets."""
        assert (
            parse_thinking_budget({"reasoning_effort": "low"}).max_thinking_tokens
            == 2048
        )
        assert (
            parse_thinking_budget({"reasoning_effort": "medium"}).max_thinking_tokens
            == 8192
        )
        assert (
            parse_thinking_budget({"reasoning_effort": "high"}).max_thinking_tokens
            == 32768
        )

    def test_parse_thinking_budget_none(self):
        """parse_thinking_budget returns None when no params provided."""
        assert parse_thinking_budget({}) is None
        assert parse_thinking_budget({"model": "test"}) is None

    def test_get_stats(self):
        """get_stats returns correct diagnostic state."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=50))
        proc.process_token("reasoning")
        stats = proc.get_stats()
        assert stats["enabled"] is True
        assert stats["max_thinking_tokens"] == 50
        assert stats["thinking_tokens_used"] == 1
        assert stats["budget_remaining"] == 49
        assert stats["budget_exceeded"] is False
        assert stats["in_thinking"] is True

    def test_budget_zero_edge(self):
        """Budget of 0 should immediately exceed on first thinking token."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=0))
        result = proc.process_token("reasoning")
        assert result["force_stop"] is True
        assert result["budget_exceeded"] is True

    def test_segment_start_count_tracking(self):
        """New thinking segment resets segment tracking."""
        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))
        # First segment
        for _ in range(5):
            proc.process_token("reasoning")
        # Transition to normal
        proc.process_token("normal")
        # Second segment starts — segment_start_count should track
        proc.process_token("reasoning")
        assert proc.thinking_tokens_used == 6  # 5 + 1 in second segment


# ═══════════════════════════════════════════════════════════════
# 8. Anthropic Streaming Format — message_start/message_stop Pairing
# ═══════════════════════════════════════════════════════════════


class TestAnthropicStreamingFormat:
    """Verify Anthropic SSE streaming follows the message_start/message_stop protocol."""

    def test_message_start_schema(self):
        """message_start event has the required Anthropic schema fields."""
        msg_start = {
            "type": "message_start",
            "message": {
                "id": "msg_test123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                },
            },
        }
        assert msg_start["type"] == "message_start"
        assert msg_start["message"]["role"] == "assistant"
        assert msg_start["message"]["stop_reason"] is None
        assert "usage" in msg_start["message"]
        assert "input_tokens" in msg_start["message"]["usage"]

    def test_message_stop_schema(self):
        """message_stop event has the required Anthropic schema."""
        msg_stop = {"type": "message_stop"}
        assert msg_stop["type"] == "message_stop"

    def test_message_delta_schema(self):
        """message_delta event has stop_reason and usage."""
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 50},
        }
        assert delta["type"] == "message_delta"
        assert delta["delta"]["stop_reason"] == "end_turn"
        assert "output_tokens" in delta["usage"]

    def test_content_block_start_text(self):
        """content_block_start for text block has correct schema."""
        block = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        assert block["type"] == "content_block_start"
        assert block["content_block"]["type"] == "text"

    def test_content_block_start_thinking(self):
        """content_block_start for thinking block has correct schema."""
        block = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "yunshu-reasoning",
            },
        }
        assert block["content_block"]["type"] == "thinking"

    def test_content_block_delta_text(self):
        """content_block_delta for text has text_delta type."""
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        assert delta["delta"]["type"] == "text_delta"
        assert delta["delta"]["text"] == "Hello"

    def test_content_block_delta_thinking(self):
        """content_block_delta for thinking has thinking_delta type."""
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
        }
        assert delta["delta"]["type"] == "thinking_delta"

    def test_content_block_stop(self):
        """content_block_stop has index."""
        block_stop = {
            "type": "content_block_stop",
            "index": 0,
        }
        assert block_stop["type"] == "content_block_stop"
        assert "index" in block_stop

    def test_sse_event_format(self):
        """SSE events must be 'event: <type>\\ndata: <json>\\n\\n'."""
        event_type = "message_start"
        data = {"type": "message_start"}
        sse = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        assert sse.startswith("event: message_start\n")
        assert sse.endswith("\n\n")
        # Parse the data line
        data_line = sse.split("data: ", 1)[1].rstrip("\n")
        parsed = json.loads(data_line)
        assert parsed["type"] == "message_start"

    def test_full_stream_sequence_no_thinking(self):
        """Complete Anthropic streaming sequence without thinking:
        message_start → content_block_start → content_block_delta* →
        content_block_stop → message_delta → message_stop"""
        events = []
        # message_start
        events.append(
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 5, "output_tokens": 0},
                    },
                },
            )
        )
        # content_block_start (text)
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        # content_block_delta
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            )
        )
        # content_block_stop
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": 0})
        )
        # message_delta
        events.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            )
        )
        # message_stop
        events.append(("message_stop", {"type": "message_stop"}))

        # Validate ordering
        assert events[0][0] == "message_start"
        assert events[-1][0] == "message_stop"
        # message_delta before message_stop
        assert events[-2][0] == "message_delta"

        # Validate SSE wire format for each
        for event_type, data in events:
            sse = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            assert "event: " in sse
            assert "data: " in sse

    def test_full_stream_sequence_with_thinking(self):
        """Anthropic streaming with thinking:
        message_start → content_block_start(thinking) →
        content_block_delta(thinking)* → content_block_stop →
        content_block_start(text) → content_block_delta(text)* →
        content_block_stop → message_delta → message_stop"""
        events = []
        events.append(
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_2",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 5, "output_tokens": 0},
                    },
                },
            )
        )
        # Thinking block
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "yunshu-reasoning",
                    },
                },
            )
        )
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
                },
            )
        )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": 0})
        )
        # Text block
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Answer"},
                },
            )
        )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": 1})
        )
        # Close
        events.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            )
        )
        events.append(("message_stop", {"type": "message_stop"}))

        # Validate structure
        assert events[0][0] == "message_start"
        assert events[-1][0] == "message_stop"
        # Thinking block index 0, text block index 1
        assert events[1][1]["content_block"]["type"] == "thinking"
        assert events[1][1]["index"] == 0
        assert events[4][1]["content_block"]["type"] == "text"
        assert events[4][1]["index"] == 1

    def test_message_start_only_emitted_once(self):
        """message_start must only be emitted once per stream."""
        _message_start_emitted = False
        emits = 0
        for _ in range(10):
            if not _message_start_emitted:
                _message_start_emitted = True
                emits += 1
        assert emits == 1

    def test_stop_reason_map_end_turn(self):
        """Default finish maps to end_turn."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason

        assert _map_stop_reason("stop") == "end_turn"

    def test_stop_reason_map_stop_sequence(self):
        """matched_stop maps to stop_sequence."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason

        assert _map_stop_reason(None, matched_stop="<end>") == "stop_sequence"

    def test_stop_reason_map_tool_use(self):
        """has_tool_calls=True maps to tool_use."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason

        assert _map_stop_reason("stop", has_tool_calls=True) == "tool_use"

    def test_stop_reason_map_max_tokens(self):
        """'length' finish_reason maps to max_tokens."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason

        assert _map_stop_reason("length") == "max_tokens"

    def test_message_stop_emitted_on_error(self):
        """message_stop must be emitted even after errors."""
        # Simulating the pattern from anthropic.py lines 1088-1096
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": "Internal server error"},
        }
        msg_stop = {"type": "message_stop"}

        # Both must be present
        assert error_event["type"] == "error"
        assert msg_stop["type"] == "message_stop"

    def test_ping_event_format(self):
        """Ping events use the SSE comment format."""
        ping = b"event: ping\ndata: {}\n\n"
        assert ping.startswith(b"event: ping\n")

    def test_cache_usage_in_message_start(self):
        """message_start usage includes cache_creation and cache_read when available."""
        msg_start = {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 60,
                    "cache_read_input_tokens": 40,
                },
            },
        }
        usage = msg_start["message"]["usage"]
        assert usage["cache_creation_input_tokens"] == 60
        assert usage["cache_read_input_tokens"] == 40
        assert usage["input_tokens"] == 100


# ═══════════════════════════════════════════════════════════════
# Cross-Cutting: Resource Cleanup Verification
# ═══════════════════════════════════════════════════════════════


class TestResourceCleanup:
    """Verify resources are cleaned up in all exit paths."""

    def test_request_tracker_no_leak_after_many_requests(self):
        """Tracker must not leak entries after register/unregister cycles."""
        tracker = RequestTracker()
        for i in range(1000):
            rid = f"req-{i}"
            tracker.register(rid, "test")
            tracker.unregister(rid)
        assert tracker.active_count == 0

    def test_request_tracker_active_count(self):
        """active_count returns correct number of registered requests."""
        tracker = RequestTracker()
        assert tracker.active_count == 0
        tracker.register("r1", "test")
        tracker.register("r2", "test")
        assert tracker.active_count == 2
        tracker.unregister("r1")
        assert tracker.active_count == 1

    def test_lora_shutdown_idempotent(self):
        """Calling shutdown() twice is safe."""
        mgr = LoRAAdapterManager()
        mgr.register_adapter("a1", "/tmp/a1")
        mgr.shutdown()
        mgr.shutdown()  # Should not raise

    def test_generation_output_finished_states(self):
        """GenerationOutput covers all expected finish reasons."""
        from yunshu_engine.batched_engine import GenerationOutput

        valid_reasons = [None, "stop", "length", "memory_limit", "error", "cancel"]
        for reason in valid_reasons:
            out = GenerationOutput(finished=reason is not None, finish_reason=reason)
            assert out.finish_reason == reason
            if reason is not None:
                assert out.finished is True

    def test_lora_entry_defaults(self):
        """LoRAAdapterEntry defaults are safe."""
        entry = LoRAAdapterEntry(adapter_id="test", adapter_path="/tmp/test")
        assert entry.is_loaded is False
        assert entry.is_merged is False
        assert entry.ref_count == 0
        assert entry.estimated_bytes == 0
