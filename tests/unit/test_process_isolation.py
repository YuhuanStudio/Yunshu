"""Tests for process_isolation — worker lifecycle, supervisor, circuit breaker.

All multiprocessing is mocked — we test the logic, not OS-level process management.
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.process_isolation import (
    InferenceWorker,
    IsolatedResult,
    IsolationMode,
    WorkerCrashError,
    WorkerProcessConfig,
    WorkerState,
    WorkerStats,
    WorkerSupervisor,
    get_supervisor,
    is_isolation_enabled,
    maybe_isolate_inference,
    reset_supervisor,
)

# ── WorkerProcessConfig ──


class TestWorkerProcessConfig:
    def test_defaults(self):
        cfg = WorkerProcessConfig()
        assert cfg.model_id == ""
        assert cfg.isolation_mode == IsolationMode.IN_PROCESS
        assert cfg.max_restarts == 5
        assert cfg.restart_window_seconds == 60.0
        assert cfg.heartbeat_interval_seconds == 2.0
        assert cfg.request_timeout_seconds == 30.0
        assert cfg.memory_limit_mb is None

    def test_custom_values(self):
        cfg = WorkerProcessConfig(
            model_id="qwen-7b",
            isolation_mode=IsolationMode.SUBPROCESS,
            max_restarts=3,
            restart_window_seconds=120.0,
            heartbeat_interval_seconds=1.0,
            request_timeout_seconds=60.0,
            memory_limit_mb=4096,
        )
        assert cfg.model_id == "qwen-7b"
        assert cfg.isolation_mode == IsolationMode.SUBPROCESS
        assert cfg.max_restarts == 3
        assert cfg.restart_window_seconds == 120.0
        assert cfg.memory_limit_mb == 4096


# ── IsolationMode ──


class TestIsolationMode:
    def test_has_all_modes(self):
        assert IsolationMode.IN_PROCESS is not None
        assert IsolationMode.SUBPROCESS is not None
        assert IsolationMode.HYBRID is not None

    def test_modes_are_distinct(self):
        modes = [
            IsolationMode.IN_PROCESS,
            IsolationMode.SUBPROCESS,
            IsolationMode.HYBRID,
        ]
        assert len(set(modes)) == 3


# ── WorkerStats ──


class TestWorkerStats:
    def test_defaults(self):
        stats = WorkerStats()
        assert stats.request_count == 0
        assert stats.crash_count == 0
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        stats = WorkerStats(request_count=10, total_latency_ms=500.0)
        assert stats.avg_latency_ms == 50.0

    def test_avg_latency_zero_requests(self):
        stats = WorkerStats()
        assert stats.avg_latency_ms == 0.0

    def test_to_dict(self):
        stats = WorkerStats(
            request_count=5,
            crash_count=1,
            total_latency_ms=200.0,
            memory_usage_mb=1024.0,
        )
        d = stats.to_dict()
        assert d["request_count"] == 5
        assert d["crash_count"] == 1
        assert d["avg_latency_ms"] == 40.0
        assert d["memory_usage_mb"] == 1024.0


# ── IsolatedResult ──


class TestIsolatedResult:
    def test_set_result(self):
        r = IsolatedResult(request_id="abc")
        r.set_result({"text": "hello"})
        assert r.completed
        assert r.result() == {"text": "hello"}

    def test_set_error(self):
        r = IsolatedResult(request_id="abc")
        r.set_error(ValueError("bad"))
        assert r.completed
        with pytest.raises(WorkerCrashError, match="abc"):
            r.result()

    def test_timeout(self):
        r = IsolatedResult(request_id="abc")
        with pytest.raises(TimeoutError, match="timed out"):
            r.result(timeout=0.01)

    def test_result_blocks_until_set(self):
        r = IsolatedResult(request_id="abc")

        def _set():
            time.sleep(0.05)
            r.set_result("done")

        t = threading.Thread(target=_set)
        t.start()
        result = r.result(timeout=1.0)
        t.join()
        assert result == "done"


# ── InferenceWorker Lifecycle ──


class TestInferenceWorkerLifecycle:
    def test_initial_state_is_idle(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        assert worker.state == WorkerState.IDLE
        assert worker.model_id == "test"

    def test_start_transitions_to_running(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            # Each Pipe() returns (parent_conn, child_conn)
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            assert worker.state == WorkerState.RUNNING
            worker.stop()

    def test_stop_transitions_to_stopped(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            assert worker.state == WorkerState.RUNNING
            worker.stop()
            assert worker.state == WorkerState.STOPPED

    def test_stop_when_already_stopped_is_noop(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        worker.stop()  # should not raise
        assert worker.state == WorkerState.STOPPED

    def test_restart_calls_stop_then_start(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            assert worker.state == WorkerState.RUNNING
            worker.restart()
            assert worker.state == WorkerState.RUNNING
            worker.stop()

    def test_start_when_already_running_is_noop(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            # Second start should be a no-op
            worker.start()
            assert worker.state == WorkerState.RUNNING
            worker.stop()


# ── InferenceWorker Health ──


class TestInferenceWorkerHealth:
    def test_not_healthy_when_idle(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        assert not worker.is_healthy()

    def test_not_healthy_when_stopped(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        worker.stop()
        assert not worker.is_healthy()

    def test_not_healthy_when_circuit_open(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        worker._state = WorkerState.CIRCUIT_OPEN
        assert not worker.is_healthy()

    def test_healthy_when_running(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            # Heartbeat was just set, should be healthy
            assert worker.is_healthy()
            worker.stop()

    def test_not_healthy_when_heartbeat_stale(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=0.01)
            )
            worker.start()
            # Simulate stale heartbeat
            worker._last_heartbeat = time.monotonic() - 10.0
            assert not worker.is_healthy()
            worker.stop()


# ── Circuit Breaker ──


class TestCircuitBreaker:
    def test_circuit_not_open_initially(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        assert not worker._is_circuit_open()

    def test_circuit_opens_after_max_crashes(self):
        cfg = WorkerProcessConfig(
            model_id="test", max_restarts=3, restart_window_seconds=60.0
        )
        worker = InferenceWorker(cfg)
        for _ in range(3):
            worker._record_crash()
        assert worker._is_circuit_open()
        assert worker.state == WorkerState.CIRCUIT_OPEN

    def test_circuit_stays_closed_under_limit(self):
        cfg = WorkerProcessConfig(
            model_id="test", max_restarts=5, restart_window_seconds=60.0
        )
        worker = InferenceWorker(cfg)
        for _ in range(4):
            worker._record_crash()
        assert not worker._is_circuit_open()

    def test_crash_times_pruned_by_window(self):
        cfg = WorkerProcessConfig(
            model_id="test", max_restarts=3, restart_window_seconds=0.1
        )
        worker = InferenceWorker(cfg)
        for _ in range(3):
            worker._record_crash()
        assert worker._is_circuit_open()

        # Wait for window to expire
        time.sleep(0.15)
        worker._prune_crash_times()
        assert not worker._is_circuit_open()

    def test_crash_count_increments(self):
        cfg = WorkerProcessConfig(model_id="test", max_restarts=100)
        worker = InferenceWorker(cfg)
        worker._record_crash()
        worker._record_crash()
        assert worker.stats.crash_count == 2


# ── InferenceWorker Submit ──


class TestInferenceWorkerSubmit:
    def test_submit_raises_when_not_running(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        with pytest.raises(WorkerCrashError, match="not running"):
            worker.submit_request({"prompt": "hello"})

    def test_submit_returns_isolated_result(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            result = worker.submit_request({"prompt": "hello"})
            assert isinstance(result, IsolatedResult)
            assert not result.completed
            worker.stop()

    def test_submit_records_crash_on_broken_pipe(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            # Parent connection that raises on send
            parent_conn = MagicMock()
            parent_conn.send.side_effect = BrokenPipeError("broken")
            child_conn = MagicMock()
            mock_pipe.return_value = (parent_conn, child_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            with pytest.raises(WorkerCrashError, match="pipe broken"):
                worker.submit_request({"prompt": "hello"})
            assert worker.stats.crash_count == 1
            worker.stop()


# ── Pending Requests on Stop ──


class TestPendingRequestsOnStop:
    def test_pending_requests_failed_on_stop(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            result = worker.submit_request({"prompt": "hello"})
            assert not result.completed

            worker.stop()
            assert result.completed
            with pytest.raises(WorkerCrashError, match="stopped"):
                result.result()


# ── WorkerSupervisor ──


class TestWorkerSupervisor:
    def test_register_worker(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="model-a"))
        supervisor.register_worker("model-a", worker)
        assert supervisor.worker_count == 1

    def test_register_worker_with_fallback(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="model-a"))

        def fallback(req):
            return {"result": "ok"}

        supervisor.register_worker("model-a", worker, fallback_fn=fallback)
        assert supervisor.worker_count == 1

    def test_deregister_worker(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="model-a"))
        supervisor.register_worker("model-a", worker)
        supervisor.deregister_worker("model-a")
        assert supervisor.worker_count == 0

    def test_get_healthy_worker_returns_none_when_not_healthy(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="model-a"))
        supervisor.register_worker("model-a", worker)
        # Worker not started, so not healthy
        assert supervisor.get_healthy_worker("model-a") is None

    def test_get_healthy_worker_returns_none_for_unknown_model(self):
        supervisor = WorkerSupervisor()
        assert supervisor.get_healthy_worker("nonexistent") is None

    def test_get_healthy_worker_returns_worker_when_healthy(self):
        supervisor = WorkerSupervisor()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(
                    model_id="model-a", heartbeat_interval_seconds=100.0
                )
            )
            worker.start()
            supervisor.register_worker("model-a", worker)
            result = supervisor.get_healthy_worker("model-a")
            assert result is worker
            worker.stop()

    def test_health_check_reports_all_workers(self):
        supervisor = WorkerSupervisor()
        worker_a = InferenceWorker(WorkerProcessConfig(model_id="a"))
        worker_b = InferenceWorker(WorkerProcessConfig(model_id="b"))
        supervisor.register_worker("a", worker_a)
        supervisor.register_worker("b", worker_b)

        status = supervisor.health_check()
        assert "a" in status
        assert "b" in status
        assert status["a"]["healthy"] is False
        assert status["a"]["state"] == "IDLE"

    def test_health_check_shows_fallback_availability(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="a"))
        supervisor.register_worker("a", worker, fallback_fn=lambda r: None)
        status = supervisor.health_check()
        assert status["a"]["has_fallback"] is True

    def test_health_check_shows_no_fallback(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="a"))
        supervisor.register_worker("a", worker)
        status = supervisor.health_check()
        assert status["a"]["has_fallback"] is False


# ── Fallback ──


class TestFallback:
    def test_fallback_to_inprocess(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="a"))
        supervisor.register_worker("a", worker, fallback_fn=lambda req: {"echo": req})
        result = supervisor.fallback_to_inprocess("a", {"prompt": "hi"})
        assert result == {"echo": {"prompt": "hi"}}

    def test_fallback_raises_when_no_fallback_registered(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="a"))
        supervisor.register_worker("a", worker)
        with pytest.raises(WorkerCrashError, match="No in-process fallback"):
            supervisor.fallback_to_inprocess("a", {"prompt": "hi"})


# ── Supervisor Submit with Fallback ──


class TestSupervisorSubmit:
    def test_submit_uses_fallback_when_no_healthy_worker(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="a"))
        supervisor.register_worker(
            "a", worker, fallback_fn=lambda req: {"result": "fallback"}
        )
        result = supervisor.submit("a", {"prompt": "hi"})
        assert result.result(timeout=1.0) == {"result": "fallback"}

    def test_submit_returns_error_when_no_worker_no_fallback(self):
        supervisor = WorkerSupervisor()
        worker = InferenceWorker(WorkerProcessConfig(model_id="a"))
        supervisor.register_worker("a", worker)
        result = supervisor.submit("a", {"prompt": "hi"})
        with pytest.raises(WorkerCrashError):
            result.result(timeout=1.0)


# ── Stats ──


class TestStats:
    def test_worker_stats_snapshot(self):
        worker = InferenceWorker(WorkerProcessConfig(model_id="test"))
        worker._stats.request_count = 10
        worker._stats.crash_count = 2
        worker._stats.total_latency_ms = 500.0
        snapshot = worker.stats
        assert snapshot.request_count == 10
        assert snapshot.crash_count == 2
        assert snapshot.avg_latency_ms == 50.0

    def test_supervisor_get_all_stats(self):
        supervisor = WorkerSupervisor()
        worker_a = InferenceWorker(WorkerProcessConfig(model_id="a"))
        worker_a._stats.request_count = 5
        supervisor.register_worker("a", worker_a)

        all_stats = supervisor.get_all_stats()
        assert "a" in all_stats
        assert all_stats["a"]["request_count"] == 5

    def test_stats_uptime_tracking(self):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            stats = worker.stats
            assert stats.uptime_seconds >= 0
            worker.stop()


# ── Singleton ──


class TestSingleton:
    def setup_method(self):
        reset_supervisor()

    def teardown_method(self):
        reset_supervisor()

    def test_get_supervisor_returns_same_instance(self):
        s1 = get_supervisor()
        s2 = get_supervisor()
        assert s1 is s2

    def test_reset_supervisor_creates_new_instance(self):
        s1 = get_supervisor()
        reset_supervisor()
        s2 = get_supervisor()
        assert s1 is not s2


# ── Environment Variable ──


class TestEnvironmentVariable:
    def test_isolation_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("YUNSHU_PROCESS_ISOLATION", None)
            assert not is_isolation_enabled()

    def test_isolation_enabled_when_set(self):
        with patch.dict("os.environ", {"YUNSHU_PROCESS_ISOLATION": "1"}):
            assert is_isolation_enabled()

    def test_isolation_disabled_for_other_values(self):
        with patch.dict("os.environ", {"YUNSHU_PROCESS_ISOLATION": "0"}):
            assert not is_isolation_enabled()


# ── Integration Hook ──


class TestMaybeIsolateInference:
    def setup_method(self):
        reset_supervisor()

    def teardown_method(self):
        reset_supervisor()

    def test_calls_fallback_when_isolation_disabled(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("YUNSHU_PROCESS_ISOLATION", None)
            called = {"v": False}

            def fallback(req):
                called["v"] = True
                return {"result": "ok"}

            result = maybe_isolate_inference("model-a", {"prompt": "hi"}, fallback)
            assert called["v"]
            assert result == {"result": "ok"}

    def test_uses_supervisor_when_isolation_enabled(self):
        supervisor = WorkerSupervisor()
        supervisor.register_worker(
            "model-a",
            InferenceWorker(WorkerProcessConfig(model_id="model-a")),
            fallback_fn=lambda req: {"result": "fallback"},
        )

        with (
            patch.dict("os.environ", {"YUNSHU_PROCESS_ISOLATION": "1"}),
            patch(
                "yunshu_engine.process_isolation.get_supervisor",
                return_value=supervisor,
            ),
        ):
            result = maybe_isolate_inference(
                "model-a", {"prompt": "hi"}, lambda r: None
            )
            assert result == {"result": "fallback"}


# ── Auto-restart ──


class TestAutoRestart:
    def test_auto_restart_triggers_for_crashed_worker(self):
        supervisor = WorkerSupervisor()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        with (
            patch(
                "yunshu_engine.process_isolation.multiprocessing.Process",
                return_value=mock_process,
            ),
            patch("yunshu_engine.process_isolation.multiprocessing.Pipe") as mock_pipe,
        ):
            mock_conn = MagicMock()
            mock_pipe.return_value = (mock_conn, mock_conn)

            worker = InferenceWorker(
                WorkerProcessConfig(model_id="test", heartbeat_interval_seconds=100.0)
            )
            worker.start()
            supervisor.register_worker("test", worker)

            # Simulate crash
            worker._state = WorkerState.CRASHED
            worker._process = mock_process

            # Auto-restart should bring it back
            supervisor._auto_restart_unhealthy()
            assert worker.state == WorkerState.RUNNING
            worker.stop()

    def test_no_restart_when_circuit_open(self):
        supervisor = WorkerSupervisor()
        cfg = WorkerProcessConfig(
            model_id="test", max_restarts=2, restart_window_seconds=60.0
        )
        worker = InferenceWorker(cfg)
        # Trip the circuit breaker
        worker._record_crash()
        worker._record_crash()
        worker._state = WorkerState.CIRCUIT_OPEN
        supervisor.register_worker("test", worker)

        # Should NOT attempt restart
        supervisor._auto_restart_unhealthy()
        assert worker.state == WorkerState.CIRCUIT_OPEN


# ── Shutdown ──


class TestShutdown:
    def test_shutdown_stops_all_workers(self):
        supervisor = WorkerSupervisor()
        worker_a = InferenceWorker(WorkerProcessConfig(model_id="a"))
        worker_b = InferenceWorker(WorkerProcessConfig(model_id="b"))
        supervisor.register_worker("a", worker_a)
        supervisor.register_worker("b", worker_b)
        assert supervisor.worker_count == 2

        supervisor.shutdown()
        assert supervisor.worker_count == 0
        assert worker_a.state == WorkerState.STOPPED
        assert worker_b.state == WorkerState.STOPPED

    def test_supervisor_monitoring_start_stop(self):
        supervisor = WorkerSupervisor()
        supervisor.start_monitoring()
        assert supervisor.is_started
        supervisor.stop_monitoring()
        # is_started remains True (was started at some point)
        assert supervisor.is_started
