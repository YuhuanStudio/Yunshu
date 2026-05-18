"""Tests for request_lifecycle.py — request lifecycle orchestrator."""

import time
from unittest.mock import patch

import pytest

from yunshu_engine.request_lifecycle import (
    AdaptiveConcurrencyController,
    RequestLifecycleOrchestrator,
    RequestLifecycleState,
    RequestPhase,
)


class TestRequestPhase:
    def test_all_phases_defined(self):
        assert RequestPhase.QUEUED
        assert RequestPhase.PREFILLING
        assert RequestPhase.DECODING
        assert RequestPhase.FINISHED
        assert RequestPhase.REJECTED
        assert RequestPhase.RETRYING
        assert RequestPhase.ABORTED


class TestRequestLifecycleState:
    def test_initial_state(self):
        state = RequestLifecycleState(request_id="r1")
        assert state.phase == RequestPhase.QUEUED
        assert state.prompt_tokens == 0
        assert state.completion_tokens == 0
        assert state.retry_count == 0

    def test_transition_queued_to_prefilling(self):
        state = RequestLifecycleState(request_id="r1")
        assert state.transition(RequestPhase.PREFILLING)
        assert state.phase == RequestPhase.PREFILLING
        assert state.prefill_start > 0

    def test_transition_prefilling_to_decoding(self):
        state = RequestLifecycleState(request_id="r1")
        state.transition(RequestPhase.PREFILLING)
        assert state.transition(RequestPhase.DECODING)
        assert state.phase == RequestPhase.DECODING
        assert state.decode_start > 0

    def test_transition_decoding_to_finished(self):
        state = RequestLifecycleState(request_id="r1")
        state.transition(RequestPhase.PREFILLING)
        state.transition(RequestPhase.DECODING)
        assert state.transition(RequestPhase.FINISHED)
        assert state.finished_at > 0

    def test_invalid_transition(self):
        state = RequestLifecycleState(request_id="r1")
        assert not state.transition(RequestPhase.FINISHED)  # can't go QUEUED → FINISHED

    def test_queued_to_rejected(self):
        state = RequestLifecycleState(request_id="r1")
        assert state.transition(RequestPhase.REJECTED)

    def test_rejected_to_retrying(self):
        state = RequestLifecycleState(request_id="r1")
        state.transition(RequestPhase.REJECTED)
        assert state.transition(RequestPhase.RETRYING)
        assert state.retry_count == 1

    def test_retrying_to_queued(self):
        state = RequestLifecycleState(request_id="r1")
        state.transition(RequestPhase.REJECTED)
        state.transition(RequestPhase.RETRYING)
        assert state.transition(RequestPhase.QUEUED)

    def test_queued_to_aborted(self):
        state = RequestLifecycleState(request_id="r1")
        assert state.transition(RequestPhase.ABORTED)

    def test_finished_is_terminal(self):
        state = RequestLifecycleState(request_id="r1")
        state.transition(RequestPhase.PREFILLING)
        state.transition(RequestPhase.DECODING)
        state.transition(RequestPhase.FINISHED)
        assert not state.transition(RequestPhase.QUEUED)

    def test_aborted_is_terminal(self):
        state = RequestLifecycleState(request_id="r1")
        state.transition(RequestPhase.ABORTED)
        assert not state.transition(RequestPhase.QUEUED)

    def test_queue_time_ms(self):
        state = RequestLifecycleState(request_id="r1")
        state.queued_at = 100.0
        state.prefill_start = 100.5
        assert state.queue_time_ms == 500.0

    def test_queue_time_ms_in_progress(self):
        state = RequestLifecycleState(request_id="r1")
        state.queued_at = time.monotonic() - 0.1
        assert state.queue_time_ms is not None
        assert state.queue_time_ms > 50

    def test_ttft_ms(self):
        state = RequestLifecycleState(request_id="r1")
        state.prefill_start = 100.0
        state.decode_start = 100.2
        assert state.ttft_ms == pytest.approx(200.0)

    def test_total_time_ms(self):
        state = RequestLifecycleState(request_id="r1")
        state.created_at = 100.0
        state.finished_at = 105.0
        assert state.total_time_ms == 5000.0

    def test_decode_time_ms(self):
        state = RequestLifecycleState(request_id="r1")
        state.decode_start = 100.0
        state.decode_end = 103.0
        assert state.decode_time_ms == 3000.0

    def test_throughput_tps(self):
        state = RequestLifecycleState(request_id="r1")
        state.decode_start = 100.0
        state.decode_end = 101.0
        state.completion_tokens = 50
        assert state.throughput_tps == 50.0

    def test_throughput_tps_none(self):
        state = RequestLifecycleState(request_id="r1")
        assert state.throughput_tps is None


class TestAdaptiveConcurrencyController:
    def test_initial_limit(self):
        ctrl = AdaptiveConcurrencyController(initial=8)
        assert ctrl.current_limit == 8

    def test_from_env(self):
        with patch.dict("os.environ", {
            "YUNSHU_CONCURRENCY_INITIAL": "16",
            "YUNSHU_CONCURRENCY_MIN": "2",
            "YUNSHU_CONCURRENCY_MAX": "256",
            "YUNSHU_SLO_TTFT_MS": "300.0",
            "YUNSHU_SLO_TOTAL_MS": "5000.0",
        }):
            ctrl = AdaptiveConcurrencyController.from_env()
            assert ctrl.current_limit == 16
            assert ctrl._minimum == 2
            assert ctrl._maximum == 256

    def test_increase_on_success(self):
        ctrl = AdaptiveConcurrencyController(initial=8, increase_window=0.0)
        state = RequestLifecycleState(request_id="r1")
        state.prefill_start = time.monotonic() - 0.1
        state.decode_start = time.monotonic() - 0.05
        state.finished_at = time.monotonic()
        ctrl.report_success(state)
        assert ctrl.current_limit == 9

    def test_decrease_on_slo_violation_ttft(self):
        ctrl = AdaptiveConcurrencyController(initial=8, slo_ttft_ms=10.0)
        state = RequestLifecycleState(request_id="r1")
        state.prefill_start = time.monotonic() - 1.0
        state.decode_start = time.monotonic() - 0.5
        ctrl.report_success(state)
        assert ctrl.current_limit == 4  # halved

    def test_decrease_on_failure(self):
        ctrl = AdaptiveConcurrencyController(initial=8)
        ctrl.report_failure("oom")
        assert ctrl.current_limit == 4

    def test_minimum_bound(self):
        ctrl = AdaptiveConcurrencyController(initial=2, minimum=2)
        ctrl.report_failure("error")
        assert ctrl.current_limit == 2

    def test_maximum_bound(self):
        ctrl = AdaptiveConcurrencyController(initial=8, maximum=8, increase_window=0.0)
        state = RequestLifecycleState(request_id="r1")
        state.prefill_start = time.monotonic() - 0.01
        state.decode_start = time.monotonic()
        state.finished_at = time.monotonic()
        ctrl.report_success(state)
        assert ctrl.current_limit == 8

    def test_get_stats(self):
        ctrl = AdaptiveConcurrencyController(initial=8)
        stats = ctrl.get_stats()
        assert stats["current_limit"] == 8
        assert "slo_ttft_ms" in stats


class TestRequestLifecycleOrchestrator:
    def test_add_request(self):
        orch = RequestLifecycleOrchestrator()
        state = orch.on_request_added("r1", prompt_tokens=10)
        assert state.request_id == "r1"
        assert state.phase == RequestPhase.QUEUED  # added but not yet prefilling
        assert orch.active_count == 0
        orch.on_prefill_start("r1")  # scheduler starts prefill
        assert orch.active_count == 1

    def test_full_lifecycle(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1")
        orch.on_prefill_start("r1")  # scheduler starts prefill
        assert orch.get_state("r1") is not None
        assert orch.active_count == 1
        assert orch.on_decode_start("r1")
        result = orch.on_request_finished("r1", completion_tokens=50)
        assert result is not None
        assert result.phase == RequestPhase.FINISHED
        assert orch.active_count == 0

    def test_reject_when_concurrency_full(self):
        orch = RequestLifecycleOrchestrator(
            concurrency_controller=AdaptiveConcurrencyController(initial=1),
            max_pending=0,
        )
        orch.on_request_added("r1")
        orch.on_prefill_start("r1")  # active=1, fills concurrency
        state = orch.on_request_added("r2")  # limit=1, pending=0 → rejected
        assert state.phase == RequestPhase.REJECTED

    def test_pending_queue_promotion(self):
        orch = RequestLifecycleOrchestrator(
            concurrency_controller=AdaptiveConcurrencyController(initial=1),
            max_pending=5,
        )
        orch.on_request_added("r1")
        orch.on_prefill_start("r1")  # fills concurrency slot
        assert orch.active_count == 1
        orch.on_request_added("r2")  # goes to pending (limit=1)
        assert orch.pending_count == 1
        orch.on_request_finished("r1")
        # r2 stays in pending — promotion is handled by the scheduler,
        # not the lifecycle manager (avoids inflating _active_count
        # before the request is actually scheduled).
        assert orch.pending_count == 1
        assert orch.active_count == 0

    def test_retry_on_failure(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1", max_retries=3)
        orch.on_prefill_start("r1")  # scheduler starts prefill
        assert orch.get_state("r1").phase == RequestPhase.PREFILLING
        result = orch.on_request_failed("r1", error="timeout", retryable=True)
        assert result.phase == RequestPhase.QUEUED
        assert result.retry_count == 1

    def test_no_retry_when_exhausted(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1", max_retries=0)
        orch.on_prefill_start("r1")  # scheduler starts prefill
        result = orch.on_request_failed("r1", error="fatal", retryable=True)
        assert result.phase == RequestPhase.FINISHED

    def test_abort(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1")
        orch.on_prefill_start("r1")  # scheduler starts prefill
        assert orch.active_count == 1
        orch.on_request_aborted("r1")
        assert orch.active_count == 0

    def test_nonexistent_request(self):
        orch = RequestLifecycleOrchestrator()
        assert not orch.on_prefill_start("nonexistent")
        assert orch.on_request_finished("nonexistent") is None
        assert orch.get_state("nonexistent") is None

    def test_check_timeouts(self):
        orch = RequestLifecycleOrchestrator(default_timeout_ms=0.001)
        orch.on_request_added("r1")
        time.sleep(0.01)
        timed_out = orch.check_timeouts()
        assert "r1" in timed_out
        assert orch._total_timeouts == 1

    def test_stats(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1", model="test-model")
        orch.on_prefill_start("r1")  # scheduler starts prefill
        orch.on_request_finished("r1", completion_tokens=10)
        stats = orch.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_completed"] == 1
        assert stats["active_requests"] == 0
        assert "concurrency" in stats
        assert "test-model" in stats["per_model"]

    def test_latency_percentiles(self):
        orch = RequestLifecycleOrchestrator()
        percentiles = orch.get_latency_percentiles()
        assert "slo_ttft_ms" in percentiles
        assert "completion_rate" in percentiles

    def test_model_counts(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1", model="model-a")
        orch.on_request_added("r2", model="model-b")
        orch.on_prefill_start("r1")
        orch.on_request_finished("r1", completion_tokens=5)
        stats = orch.get_stats()
        assert stats["per_model"]["model-a"]["completed"] == 1
        assert stats["per_model"]["model-b"]["total"] == 1

    def test_failure_decreases_concurrency(self):
        cc = AdaptiveConcurrencyController(initial=8)
        orch = RequestLifecycleOrchestrator(concurrency_controller=cc)
        orch.on_request_added("r1")  # auto-starts prefill
        orch.on_request_failed("r1", error="oom", retryable=False)
        assert cc.current_limit < 8

    def test_double_prefill_start(self):
        orch = RequestLifecycleOrchestrator()
        orch.on_request_added("r1")
        orch.on_prefill_start("r1")  # first start succeeds
        # Second prefill start should fail (already in PREFILLING)
        assert not orch.on_prefill_start("r1")

    def test_phase_distribution(self):
        orch = RequestLifecycleOrchestrator(
            concurrency_controller=AdaptiveConcurrencyController(initial=2),
        )
        orch.on_request_added("r1")
        orch.on_prefill_start("r1")  # scheduler starts prefill
        orch.on_request_added("r2")
        orch.on_prefill_start("r2")  # scheduler starts prefill
        stats = orch.get_stats()
        phases = stats["phase_distribution"]
        assert phases.get("PREFILLING", 0) == 2
