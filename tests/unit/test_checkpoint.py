"""Tests for inference checkpoint/restore and fault recovery.

Tests:
- InferenceCheckpoint: save/load roundtrip, auto-checkpoint, eviction, stats
- FaultRecoveryManager: all 4 strategies, priority config, stats
- ProgressEstimator: speed estimation, progress tracking, accuracy stats
"""
import threading
import time

from yunshu_engine.checkpoint import (
    AutoCheckpointPolicy,
    FaultRecoveryManager,
    InferenceCheckpoint,
    InferenceState,
    ProgressEstimator,
    RecoveryStrategy,
)

# ── InferenceState Tests ──


class TestInferenceState:
    """Tests for InferenceState serialization roundtrip."""

    def test_default_state(self):
        state = InferenceState(request_id="req-1")
        assert state.request_id == "req-1"
        assert state.generated_tokens == []
        assert state.output_text == ""
        assert state.position == 0
        assert state.temperature == 0.7
        assert state.max_tokens == 256
        assert state.thinking_active is False
        assert state.last_error is None

    def test_to_dict_from_dict_roundtrip(self):
        state = InferenceState(
            request_id="req-rt",
            generated_tokens=[10, 20, 30],
            output_text="Hello",
            position=100,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_tokens=512,
            repetition_penalty=1.2,
            seed=42,
            grammar_state={"allowed_tokens": [1, 2, 3]},
            thinking_active=True,
            thinking_budget=1024,
            reasoning_state={"depth": 5},
            model_name="qwen-2.5-7b",
            last_error="OOM",
        )
        d = state.to_dict()
        restored = InferenceState.from_dict(d)

        assert restored.request_id == "req-rt"
        assert restored.generated_tokens == [10, 20, 30]
        assert restored.output_text == "Hello"
        assert restored.position == 100
        assert restored.temperature == 0.5
        assert restored.top_k == 50
        assert restored.seed == 42
        assert restored.grammar_state == {"allowed_tokens": [1, 2, 3]}
        assert restored.thinking_active is True
        assert restored.thinking_budget == 1024
        assert restored.reasoning_state == {"depth": 5}
        assert restored.model_name == "qwen-2.5-7b"
        assert restored.last_error == "OOM"

    def test_to_dict_deep_copy(self):
        """Mutating to_dict result does not affect original state."""
        state = InferenceState(
            request_id="req-dc",
            generated_tokens=[1, 2, 3],
            grammar_state={"key": "value"},
        )
        d = state.to_dict()
        d["generated_tokens"].append(4)
        d["grammar_state"]["key"] = "modified"

        assert state.generated_tokens == [1, 2, 3]
        assert state.grammar_state["key"] == "value"

    def test_from_dict_ignores_unknown_fields(self):
        """from_dict should ignore fields not in the dataclass."""
        state = InferenceState.from_dict({
            "request_id": "req-extra",
            "unknown_field": "ignored",
            "temperature": 0.3,
        })
        assert state.request_id == "req-extra"
        assert state.temperature == 0.3

    def test_timestamp_auto_set(self):
        before = time.monotonic()
        state = InferenceState(request_id="req-ts")
        after = time.monotonic()
        assert before <= state.timestamp <= after


# ── InferenceCheckpoint Tests ──


class TestInferenceCheckpoint:
    """Tests for InferenceCheckpoint save/load/delete/eviction."""

    def test_save_and_load(self):
        cp = InferenceCheckpoint()
        state = InferenceState(request_id="req-1", position=42)
        cp.save("req-1", state)

        loaded = cp.load("req-1")
        assert loaded is not None
        assert loaded.request_id == "req-1"
        assert loaded.position == 42

    def test_load_returns_deep_copy(self):
        """Modifying loaded state does not affect stored checkpoint."""
        cp = InferenceCheckpoint()
        state = InferenceState(request_id="req-copy", generated_tokens=[1, 2, 3])
        cp.save("req-copy", state)

        loaded = cp.load("req-copy")
        loaded.generated_tokens.append(4)
        loaded.position = 999

        reloaded = cp.load("req-copy")
        assert reloaded.generated_tokens == [1, 2, 3]
        assert reloaded.position == 0

    def test_load_nonexistent_returns_none(self):
        cp = InferenceCheckpoint()
        assert cp.load("nonexistent") is None

    def test_delete(self):
        cp = InferenceCheckpoint()
        cp.save("req-del", InferenceState(request_id="req-del"))
        assert cp.delete("req-del") is True
        assert cp.load("req-del") is None

    def test_delete_nonexistent(self):
        cp = InferenceCheckpoint()
        assert cp.delete("nonexistent") is False

    def test_list_checkpoints(self):
        cp = InferenceCheckpoint()
        cp.save("req-a", InferenceState(request_id="req-a"))
        cp.save("req-b", InferenceState(request_id="req-b"))
        cp.save("req-c", InferenceState(request_id="req-c"))

        keys = cp.list_checkpoints()
        assert keys == ["req-a", "req-b", "req-c"]

    def test_overwrite_existing(self):
        """Saving same request_id overwrites the previous checkpoint."""
        cp = InferenceCheckpoint()
        cp.save("req-ov", InferenceState(request_id="req-ov", position=10))
        cp.save("req-ov", InferenceState(request_id="req-ov", position=20))

        loaded = cp.load("req-ov")
        assert loaded.position == 20
        assert len(cp.list_checkpoints()) == 1

    def test_eviction_at_capacity(self):
        """Oldest checkpoint evicted when max_checkpoints reached."""
        cp = InferenceCheckpoint(max_checkpoints=3)
        cp.save("r1", InferenceState(request_id="r1"))
        cp.save("r2", InferenceState(request_id="r2"))
        cp.save("r3", InferenceState(request_id="r3"))
        # At capacity — next save evicts r1
        cp.save("r4", InferenceState(request_id="r4"))

        assert cp.load("r1") is None  # evicted
        assert cp.load("r2") is not None
        assert cp.load("r3") is not None
        assert cp.load("r4") is not None

    def test_eviction_not_triggered_on_overwrite(self):
        """Overwriting existing entry does not trigger eviction."""
        cp = InferenceCheckpoint(max_checkpoints=2)
        cp.save("r1", InferenceState(request_id="r1"))
        cp.save("r2", InferenceState(request_id="r2"))
        # Overwrite r1 — should NOT evict
        cp.save("r1", InferenceState(request_id="r1", position=99))

        assert cp.load("r1") is not None
        assert cp.load("r2") is not None
        assert len(cp.list_checkpoints()) == 2

    def test_stats(self):
        cp = InferenceCheckpoint()
        cp.save("s1", InferenceState(request_id="s1"))
        cp.save("s2", InferenceState(request_id="s2"))
        cp.load("s1")
        cp.delete("s2")

        stats = cp.get_stats()
        assert stats["checkpoints_saved"] == 2
        assert stats["checkpoints_loaded"] == 1
        assert stats["checkpoints_deleted"] == 1
        assert stats["current_count"] == 1

    def test_clear(self):
        cp = InferenceCheckpoint()
        cp.save("c1", InferenceState(request_id="c1"))
        cp.save("c2", InferenceState(request_id="c2"))
        count = cp.clear()
        assert count == 2
        assert len(cp.list_checkpoints()) == 0


class TestAutoCheckpoint:
    """Tests for auto-checkpoint policy and triggering."""

    def test_every_n_tokens_policy(self):
        cp = InferenceCheckpoint(
            auto_checkpoint_interval=10,
            auto_checkpoint_policy=AutoCheckpointPolicy.EVERY_N_TOKENS,
        )
        state = InferenceState(request_id="auto-1", generated_tokens=list(range(5)))
        assert cp.auto_checkpoint(state) is False  # 5 tokens, no prev cp, interval 0 == 0

        # 10 tokens — no prev checkpoint, prev_interval=0, curr=1 → triggers
        state = InferenceState(request_id="auto-1", generated_tokens=list(range(10)))
        assert cp.auto_checkpoint(state) is True  # crosses 10 boundary

        # 15 tokens — prev cp has 10 tokens (interval 1), curr=1 → no trigger
        state = InferenceState(request_id="auto-1", generated_tokens=list(range(15)))
        assert cp.auto_checkpoint(state) is False  # same interval (10-19)

        # 20 tokens — prev cp has 10 tokens (interval 1), curr=2 → triggers
        state = InferenceState(request_id="auto-1", generated_tokens=list(range(20)))
        assert cp.auto_checkpoint(state) is True  # crosses 20 boundary

    def test_disabled_policy(self):
        cp = InferenceCheckpoint(
            auto_checkpoint_policy=AutoCheckpointPolicy.DISABLED,
        )
        state = InferenceState(request_id="disabled", generated_tokens=list(range(500)))
        assert cp.auto_checkpoint(state) is False

    def test_every_n_seconds_policy(self):
        cp = InferenceCheckpoint(
            auto_checkpoint_interval=1,  # 1 second
            auto_checkpoint_policy=AutoCheckpointPolicy.EVERY_N_SECONDS,
        )
        state = InferenceState(request_id="sec-1", generated_tokens=[1, 2])
        # First call: no existing checkpoint, should save
        assert cp.auto_checkpoint(state) is True

        # Manipulate stored checkpoint timestamp to simulate elapsed time
        loaded = cp._checkpoints["sec-1"]
        loaded.timestamp = time.monotonic() - 0.5  # 0.5s ago
        # 0.5s < 1s interval → should NOT trigger
        assert cp.should_auto_checkpoint("sec-1", 10) is False

        loaded.timestamp = time.monotonic() - 1.5  # 1.5s ago
        # 1.5s >= 1s interval → should trigger
        assert cp.should_auto_checkpoint("sec-1", 10) is True

    def test_auto_checkpoint_stats(self):
        cp = InferenceCheckpoint(
            auto_checkpoint_interval=5,
            auto_checkpoint_policy=AutoCheckpointPolicy.EVERY_N_TOKENS,
        )
        state = InferenceState(request_id="auto-stats", generated_tokens=list(range(5)))
        cp.auto_checkpoint(state)  # triggers (crosses 5)
        stats = cp.get_stats()
        assert stats["auto_checkpoints"] == 1

    def test_thread_safety(self):
        """Concurrent save/load does not corrupt state."""
        cp = InferenceCheckpoint()
        errors = []

        def writer(rid, count):
            try:
                for i in range(count):
                    cp.save(
                        rid,
                        InferenceState(request_id=rid, position=i),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t-{i}", 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        stats = cp.get_stats()
        assert stats["checkpoints_saved"] == 200
        assert stats["current_count"] == 4


# ── FaultRecoveryManager Tests ──


class TestFaultRecoveryRetry:
    """Tests for retry recovery strategy."""

    def test_retry_from_checkpoint(self):
        cp = InferenceCheckpoint()
        state = InferenceState(
            request_id="retry-1",
            generated_tokens=[1, 2, 3],
            output_text="Hel",
            position=10,
        )
        cp.save("retry-1", state)
        frm = FaultRecoveryManager(cp)
        # Pass current_state so handle_error stamps last_error before retry
        result = frm.handle_error(
            "retry-1", RuntimeError("GPU fault"), current_state=state
        )

        assert result.success is True
        assert result.strategy == RecoveryStrategy.RETRY
        assert result.restored_state is not None
        assert result.restored_state.generated_tokens == [1, 2, 3]
        assert result.restored_state.last_error == "GPU fault"

    def test_retry_exhausted(self):
        cp = InferenceCheckpoint()
        cp.save("retry-max", InferenceState(request_id="retry-max"))
        frm = FaultRecoveryManager(cp, max_retries=2)

        # Exhaust retries
        frm.handle_error("retry-max", RuntimeError("err1"))
        frm.handle_error("retry-max", RuntimeError("err2"))
        result = frm.handle_error("retry-max", RuntimeError("err3"))

        # 3rd call: retry strategy fails, falls through to next
        assert result.success is True  # graceful_error is always success

    def test_retry_no_checkpoint(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_ERROR],
        )
        result = frm.handle_error("no-cp", RuntimeError("err"))

        # Retry fails (no checkpoint), graceful_error succeeds
        assert result.success is True
        assert result.strategy == RecoveryStrategy.GRACEFUL_ERROR

    def test_reset_retry_count(self):
        cp = InferenceCheckpoint()
        cp.save("reset-1", InferenceState(request_id="reset-1"))
        frm = FaultRecoveryManager(cp, max_retries=1)

        frm.handle_error("reset-1", RuntimeError("err"))
        frm.reset_retry_count("reset-1")
        # Should be able to retry again
        result = frm.handle_error("reset-1", RuntimeError("err2"))
        assert result.success is True
        assert result.strategy == RecoveryStrategy.RETRY


class TestFaultRecoveryTruncate:
    """Tests for truncate recovery strategy."""

    def test_truncate_reduces_max_tokens(self):
        cp = InferenceCheckpoint()
        state = InferenceState(
            request_id="trunc-1",
            generated_tokens=list(range(50)),
            max_tokens=200,
        )
        cp.save("trunc-1", state)
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.TRUNCATE],
            truncate_ratio=0.5,
        )
        result = frm.handle_error("trunc-1", MemoryError("OOM"))

        assert result.success is True
        assert result.strategy == RecoveryStrategy.TRUNCATE
        assert result.restored_state is not None
        remaining = 200 - 50  # 150
        expected_max = 50 + int(remaining * 0.5)  # 50 + 75 = 125
        assert result.restored_state.max_tokens == expected_max

    def test_truncate_no_checkpoint(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.TRUNCATE, RecoveryStrategy.GRACEFUL_ERROR],
        )
        result = frm.handle_error("no-cp", MemoryError("OOM"))
        # Truncate fails, graceful_error succeeds
        assert result.strategy == RecoveryStrategy.GRACEFUL_ERROR

    def test_truncate_with_zero_remaining(self):
        """When all tokens are generated, truncate still returns at least 1."""
        cp = InferenceCheckpoint()
        state = InferenceState(
            request_id="trunc-done",
            generated_tokens=list(range(200)),
            max_tokens=200,
        )
        cp.save("trunc-done", state)
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.TRUNCATE],
            truncate_ratio=0.5,
        )
        result = frm.handle_error("trunc-done", MemoryError("OOM"))
        assert result.success is True
        # remaining=0, new_remaining=max(1, 0)=1
        assert result.restored_state.max_tokens == 201


class TestFaultRecoveryFallback:
    """Tests for fallback model recovery strategy."""

    def test_fallback_to_smaller_model(self):
        cp = InferenceCheckpoint()
        cp.save("fb-1", InferenceState(
            request_id="fb-1",
            model_name="qwen-2.5-7b",
        ))
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.FALLBACK_MODEL],
            fallback_models={"qwen-2.5-7b": "qwen-2.5-0.5b"},
        )
        result = frm.handle_error("fb-1", RuntimeError("model crash"))

        assert result.success is True
        assert result.strategy == RecoveryStrategy.FALLBACK_MODEL
        assert result.restored_state.model_name == "qwen-2.5-0.5b"
        assert result.metadata["original_model"] == "qwen-2.5-7b"
        assert result.metadata["fallback_model"] == "qwen-2.5-0.5b"

    def test_fallback_no_mapping(self):
        cp = InferenceCheckpoint()
        cp.save("fb-nm", InferenceState(
            request_id="fb-nm",
            model_name="unknown-model",
        ))
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.FALLBACK_MODEL, RecoveryStrategy.GRACEFUL_ERROR],
            fallback_models={"qwen-2.5-7b": "qwen-2.5-0.5b"},
        )
        result = frm.handle_error("fb-nm", RuntimeError("err"))
        # Fallback fails (no mapping), graceful_error succeeds
        assert result.strategy == RecoveryStrategy.GRACEFUL_ERROR

    def test_fallback_no_checkpoint(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.FALLBACK_MODEL, RecoveryStrategy.GRACEFUL_ERROR],
        )
        result = frm.handle_error("no-cp", RuntimeError("err"))
        assert result.strategy == RecoveryStrategy.GRACEFUL_ERROR


class TestFaultRecoveryGraceful:
    """Tests for graceful error recovery strategy."""

    def test_graceful_error_returns_partial(self):
        cp = InferenceCheckpoint()
        cp.save("gr-1", InferenceState(
            request_id="gr-1",
            output_text="Partial generation text",
            generated_tokens=list(range(10)),
            position=20,
        ))
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.GRACEFUL_ERROR],
        )
        result = frm.handle_error("gr-1", TimeoutError("timeout"))

        assert result.success is True
        assert result.strategy == RecoveryStrategy.GRACEFUL_ERROR
        assert result.partial_output == "Partial generation text"
        assert result.error_message == "timeout"
        assert result.metadata["tokens_generated"] == 10
        assert result.metadata["position"] == 20

    def test_graceful_error_no_checkpoint(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.GRACEFUL_ERROR],
        )
        result = frm.handle_error("no-cp", RuntimeError("err"))

        assert result.success is True
        assert result.partial_output == ""


class TestFaultRecoveryConfig:
    """Tests for FaultRecoveryManager configuration."""

    def test_configure_strategy_priority(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(cp)
        frm.configure(
            strategy_priority=[
                RecoveryStrategy.TRUNCATE,
                RecoveryStrategy.RETRY,
            ]
        )
        assert frm._strategy_priority[0] == RecoveryStrategy.TRUNCATE

    def test_configure_max_retries(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(cp)
        frm.configure(max_retries=5)
        assert frm._max_retries == 5

    def test_configure_truncate_ratio(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(cp)
        frm.configure(truncate_ratio=0.25)
        assert frm._truncate_ratio == 0.25

    def test_configure_fallback_models(self):
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(cp)
        frm.configure(fallback_models={"big": "small"})
        assert frm._fallback_models == {"big": "small"}


class TestFaultRecoveryStats:
    """Tests for FaultRecoveryManager statistics."""

    def test_stats_after_errors(self):
        cp = InferenceCheckpoint()
        cp.save("st-1", InferenceState(request_id="st-1"))
        frm = FaultRecoveryManager(cp)

        frm.handle_error("st-1", RuntimeError("err1"))
        frm.handle_error("st-1", RuntimeError("err2"))

        stats = frm.get_stats()
        assert stats["errors_handled"] == 2
        assert stats["recoveries_success"] == 2
        assert stats["recoveries_failed"] == 0
        assert stats["success_rate"] == 1.0

    def test_strategy_stats(self):
        cp = InferenceCheckpoint()
        cp.save("ss-1", InferenceState(request_id="ss-1"))
        frm = FaultRecoveryManager(
            cp,
            strategy_priority=[RecoveryStrategy.RETRY],
        )
        frm.handle_error("ss-1", RuntimeError("err"))
        stats = frm.get_stats()
        assert stats["strategy_stats"]["retry"]["attempts"] == 1
        assert stats["strategy_stats"]["retry"]["successes"] == 1

    def test_current_state_saved_on_error(self):
        """handle_error saves current_state as checkpoint before recovery."""
        cp = InferenceCheckpoint()
        frm = FaultRecoveryManager(cp)
        current = InferenceState(
            request_id="save-on-err",
            generated_tokens=[1, 2, 3],
            position=50,
        )
        frm.handle_error("save-on-err", RuntimeError("crash"), current_state=current)

        # Should have been saved
        loaded = cp.load("save-on-err")
        assert loaded is not None
        assert loaded.position == 50
        assert loaded.last_error == "crash"


# ── ProgressEstimator Tests ──


class TestProgressEstimator:
    """Tests for progress estimation and speed tracking."""

    def test_register_and_update(self):
        pe = ProgressEstimator()
        pe.register("p1", max_tokens=100)
        pe.update("p1", tokens_generated=10, elapsed_ms=100.0)

        info = pe.estimate_remaining("p1")
        assert info is not None
        assert info.tokens_generated == 10
        assert info.tokens_remaining == 90
        assert info.progress_pct == 0.1

    def test_progress_percentage(self):
        pe = ProgressEstimator()
        pe.register("p2", max_tokens=200)
        pe.update("p2", tokens_generated=100, elapsed_ms=500.0)

        assert pe.get_progress("p2") == 0.5

    def test_progress_at_zero(self):
        pe = ProgressEstimator()
        pe.register("p3", max_tokens=100)
        assert pe.get_progress("p3") == 0.0

    def test_progress_untracked(self):
        pe = ProgressEstimator()
        assert pe.get_progress("unknown") == 0.0

    def test_estimate_remaining_untracked(self):
        pe = ProgressEstimator()
        assert pe.estimate_remaining("unknown") is None

    def test_speed_estimation(self):
        pe = ProgressEstimator(speed_window_s=10.0)
        pe.register("speed-1", max_tokens=1000)

        # First update to establish baseline
        pe.update("speed-1", tokens_generated=50, elapsed_ms=500.0)
        # Wait long enough for time.monotonic() to show measurable delta
        time.sleep(0.1)
        pe.update("speed-1", tokens_generated=60, elapsed_ms=600.0)

        info = pe.estimate_remaining("speed-1")
        assert info is not None
        assert info.speed_tps > 0
        # 10 tokens in ~0.1s = ~100 tok/s (allow very wide margin for CI jitter)
        assert info.speed_tps > 10  # at least some measurable speed
        assert info.estimated_remaining_ms > 0

    def test_estimate_remaining_unknown_speed(self):
        pe = ProgressEstimator()
        pe.register("no-speed", max_tokens=100)
        # No update, so speed is 0
        info = pe.estimate_remaining("no-speed")
        assert info is not None
        assert info.estimated_remaining_ms == -1.0  # unknown

    def test_estimate_remaining_completed(self):
        pe = ProgressEstimator()
        pe.register("done", max_tokens=10)
        pe.update("done", tokens_generated=10, elapsed_ms=100.0)

        info = pe.estimate_remaining("done")
        assert info is not None
        assert info.tokens_remaining == 0
        assert info.estimated_remaining_ms == 0.0

    def test_progress_capped_at_1(self):
        pe = ProgressEstimator()
        pe.register("over", max_tokens=50)
        pe.update("over", tokens_generated=60, elapsed_ms=100.0)

        assert pe.get_progress("over") == 1.0
        info = pe.estimate_remaining("over")
        assert info.tokens_remaining == 0

    def test_unregister(self):
        pe = ProgressEstimator()
        pe.register("unreg", max_tokens=100)
        pe.unregister("unreg")
        assert pe.estimate_remaining("unreg") is None

    def test_max_history_eviction(self):
        pe = ProgressEstimator(max_history=3)
        pe.register("h1", max_tokens=10)
        pe.register("h2", max_tokens=10)
        pe.register("h3", max_tokens=10)
        pe.register("h4", max_tokens=10)  # should evict h1

        assert pe.get_progress("h1") == 0.0  # not tracked anymore
        assert pe.get_progress("h4") == 0.0  # registered but no update

    def test_record_completion_accuracy(self):
        pe = ProgressEstimator()
        pe.record_completion("done", actual_remaining_ms=5000.0, estimated_remaining_ms=4800.0)
        pe.record_completion("done2", actual_remaining_ms=3000.0, estimated_remaining_ms=3500.0)

        stats = pe.get_stats()
        assert stats["completed_estimates"] == 2
        # error1=200, error2=500, avg=350
        assert stats["average_error_ms"] == 350.0

    def test_variable_speed_ema(self):
        """Speed adapts to changes via EMA (exponential moving average).

        Instead of relying on wall-clock timing (which is unreliable in CI),
        we verify that the EMA formula is applied correctly by checking
        that the record's speed is updated after each call.
        """
        pe = ProgressEstimator(speed_window_s=10.0)
        pe.register("var-speed", max_tokens=1000)

        # Phase 1: establish initial speed via update
        rec = pe._records["var-speed"]
        rec.last_update_time = time.monotonic() - 1.0  # 1 second ago
        rec.last_update_tokens = 0
        pe.update("var-speed", tokens_generated=10, elapsed_ms=1000.0)
        # 10 tokens in ~1s = ~10 tok/s (exact value depends on clock)
        speed_after_slow = rec.current_speed_tps
        assert speed_after_slow > 0

        # Phase 2: fast burst — 90 tokens in ~1s
        rec.last_update_time = time.monotonic() - 1.0  # 1 second ago
        rec.last_update_tokens = 10
        pe.update("var-speed", tokens_generated=100, elapsed_ms=2000.0)
        speed_after_fast = rec.current_speed_tps

        # EMA should have moved toward the faster speed
        assert speed_after_fast > speed_after_slow

    def test_stats(self):
        pe = ProgressEstimator()
        pe.register("s1", max_tokens=100)
        pe.update("s1", tokens_generated=10, elapsed_ms=100.0)
        pe.estimate_remaining("s1")
        pe.estimate_remaining("s1")

        stats = pe.get_stats()
        assert stats["estimates_made"] == 2
        assert stats["active_requests"] == 1

    def test_clear(self):
        pe = ProgressEstimator()
        pe.register("cl", max_tokens=100)
        pe.update("cl", tokens_generated=50, elapsed_ms=500.0)
        pe.estimate_remaining("cl")
        pe.clear()

        stats = pe.get_stats()
        assert stats["active_requests"] == 0
        assert stats["estimates_made"] == 0
        assert pe.estimate_remaining("cl") is None
