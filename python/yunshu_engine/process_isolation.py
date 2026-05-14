"""Yunshu Process Isolation — exo-style fault tolerance for inference.

exo uses independent processes per inference task with a supervisor for fault
isolation: a crash in one model's inference does not take down the whole server.
This module replicates that pattern for Yunshu.

Architecture:
  WorkerSupervisor (singleton)
    ├── InferenceWorker[model-A] → subprocess via multiprocessing.Process
    ├── InferenceWorker[model-B] → subprocess via multiprocessing.Process
    └── fallback: in-process inference when worker is unhealthy

Each InferenceWorker:
  - Runs inference in an isolated subprocess
  - Communicates via request/result pipes + heartbeat pipe
  - Auto-restarted by the supervisor on crash
  - Circuit breaker prevents restart loops

Integration:
  - YUNSHU_PROCESS_ISOLATION=1 env var enables isolation
  - BatchedEngine wraps inference calls via get_supervisor().submit()
  - Graceful degradation: worker crash → clear error, not server crash
  - Stats exposed via monitoring endpoints
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .exceptions import YunshuError

logger = logging.getLogger(__name__)


# ── Enums ──


class IsolationMode(Enum):
    """Process isolation strategy.

    - IN_PROCESS: current behavior, everything in one process
    - SUBPROCESS: each model runs in its own subprocess
    - HYBRID: critical paths (speculative decode, MoE) isolated
    """

    IN_PROCESS = auto()
    SUBPROCESS = auto()
    HYBRID = auto()


class WorkerState(Enum):
    """Lifecycle state of an InferenceWorker."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    CRASHED = auto()
    CIRCUIT_OPEN = auto()  # circuit breaker tripped, no restart attempts


# ── Data Classes ──


@dataclass
class WorkerProcessConfig:
    """Configuration for a single isolated worker process.

    Attributes:
        model_id: Unique identifier for the model this worker serves.
        isolation_mode: Which isolation strategy to use.
        max_restarts: Maximum restarts allowed within restart_window_seconds.
        restart_window_seconds: Sliding time window for restart counting.
        heartbeat_interval_seconds: How often the worker sends a heartbeat.
        request_timeout_seconds: Max seconds to wait for a request result.
        memory_limit_mb: Optional hard memory limit for the subprocess.
    """

    model_id: str = ""
    isolation_mode: IsolationMode = IsolationMode.IN_PROCESS
    max_restarts: int = 5
    restart_window_seconds: float = 60.0
    heartbeat_interval_seconds: float = 2.0
    request_timeout_seconds: float = 30.0
    memory_limit_mb: Optional[int] = None


@dataclass
class WorkerStats:
    """Runtime statistics for an InferenceWorker."""

    request_count: int = 0
    crash_count: int = 0
    total_latency_ms: float = 0.0
    last_request_time: float = 0.0
    last_crash_time: float = 0.0
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "crash_count": self.crash_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "last_request_time": self.last_request_time,
            "last_crash_time": self.last_crash_time,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


@dataclass
class IsolatedResult:
    """Future-like result from an isolated worker request.

    Blocks on .result() until the worker responds or timeout.
    """

    request_id: str
    _result: Any = None
    _error: Optional[Exception] = None
    _completed: bool = False
    _event: threading.Event = field(default_factory=threading.Event)

    def set_result(self, value: Any) -> None:
        self._result = value
        self._completed = True
        self._event.set()

    def set_error(self, exc: Exception) -> None:
        self._error = exc
        self._completed = True
        self._event.set()

    def result(self, timeout: Optional[float] = None) -> Any:
        """Wait for and return the result. Raises on error or timeout."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(
                f"Worker request {self.request_id} timed out after {timeout}s"
            )
        if self._error is not None:
            raise WorkerCrashError(
                f"Worker request {self.request_id} failed",
                details={"error": str(self._error)},
            )
        return self._result

    @property
    def completed(self) -> bool:
        return self._completed


# ── Exceptions ──


class WorkerCrashError(YunshuError):
    """Raised when an isolated worker crashes during request processing."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.model_id = model_id


class CircuitBreakerOpenError(YunshuError):
    """Raised when the circuit breaker prevents a request."""

    def __init__(self, model_id: str, crash_count: int, window_seconds: float):
        super().__init__(
            f"Circuit breaker open for model '{model_id}': "
            f"{crash_count} crashes in {window_seconds}s",
            details={
                "model_id": model_id,
                "crash_count": crash_count,
                "window_seconds": window_seconds,
            },
        )
        self.model_id = model_id


# ── Worker Process Target ──


def _worker_main(
    model_id: str,
    request_pipe: multiprocessing.connection.Connection,
    result_pipe: multiprocessing.connection.Connection,
    heartbeat_pipe: multiprocessing.connection.Connection,
    heartbeat_interval: float,
    memory_limit_mb: Optional[int],
) -> None:
    """Entry point for an isolated worker subprocess.

    Loops on request_pipe, sends results on result_pipe,
    sends heartbeat pings on heartbeat_pipe.
    """
    import resource

    if memory_limit_mb is not None:
        try:
            limit_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            pass  # non-fatal; best-effort limit

    # Model loading would happen here in production
    # For now, we just process requests
    last_heartbeat = time.monotonic()
    try:
        while True:
            # Check for heartbeat
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                try:
                    heartbeat_pipe.send(("ping", time.time()))
                except (BrokenPipeError, OSError):
                    break
                last_heartbeat = now

            # Poll for requests (non-blocking with small timeout)
            if request_pipe.poll(timeout=heartbeat_interval / 2):
                try:
                    msg = request_pipe.recv()
                except EOFError:
                    break

                if msg is None:
                    # Shutdown signal
                    break

                request_id, payload = msg
                start = time.monotonic()
                try:
                    # In production, this would call the actual model inference
                    # For now, just echo back
                    result = {"echo": payload}
                    elapsed_ms = (time.monotonic() - start) * 1000
                    result_pipe.send(
                        {
                            "request_id": request_id,
                            "result": result,
                            "latency_ms": elapsed_ms,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    result_pipe.send(
                        {
                            "request_id": request_id,
                            "result": None,
                            "latency_ms": 0,
                            "error": str(exc),
                        }
                    )
    except Exception:
        logger.exception("Worker %s crashed", model_id)
    finally:
        try:
            heartbeat_pipe.send(("shutdown", time.time()))
        except (BrokenPipeError, OSError):
            pass


# ── InferenceWorker ──


class InferenceWorker:
    """Wraps a single model's inference in an isolated process.

    Manages the subprocess lifecycle, heartbeat monitoring, and
    request/result routing via pipes.

    Usage:
        worker = InferenceWorker(config)
        worker.start()
        result = worker.submit_request({"prompt": "hello"})
        worker.stop()
    """

    def __init__(
        self,
        config: WorkerProcessConfig,
        worker_fn: Optional[Callable] = None,
    ) -> None:
        self._config = config
        self._state = WorkerState.IDLE
        self._stats = WorkerStats()
        self._start_time: float = 0.0

        # Crash tracking for circuit breaker
        self._crash_times: deque[float] = deque()

        # Pipes: parent ↔ child communication
        self._request_pipe: Optional[multiprocessing.Pipe] = None
        self._result_pipe: Optional[multiprocessing.Pipe] = None
        self._heartbeat_pipe: Optional[multiprocessing.Pipe] = None

        # Subprocess handle
        self._process: Optional[multiprocessing.Process] = None

        # Heartbeat monitoring thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_heartbeat: float = 0.0
        self._heartbeat_thread_running: bool = False

        # Pending requests: request_id → IsolatedResult
        self._pending: dict[str, IsolatedResult] = {}
        self._pending_lock = threading.Lock()

        # Result reader thread
        self._result_thread: Optional[threading.Thread] = None
        self._result_thread_running: bool = False

        # Worker entry point (injectable for testing)
        self._worker_fn = worker_fn or _worker_main

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def stats(self) -> WorkerStats:
        """Return a snapshot of worker stats."""
        s = WorkerStats(
            request_count=self._stats.request_count,
            crash_count=self._stats.crash_count,
            total_latency_ms=self._stats.total_latency_ms,
            last_request_time=self._stats.last_request_time,
            last_crash_time=self._stats.last_crash_time,
            memory_usage_mb=self._stats.memory_usage_mb,
        )
        if self._state == WorkerState.RUNNING and self._start_time > 0:
            s.uptime_seconds = time.monotonic() - self._start_time
        return s

    @property
    def config(self) -> WorkerProcessConfig:
        return self._config

    def start(self) -> None:
        """Start the worker subprocess."""
        if self._state == WorkerState.RUNNING:
            return

        self._state = WorkerState.STARTING

        try:
            # Create pipes (each returns (parent_conn, child_conn))
            req_parent, req_child = multiprocessing.Pipe(duplex=True)
            res_parent, res_child = multiprocessing.Pipe(duplex=True)
            hb_parent, hb_child = multiprocessing.Pipe(duplex=True)

            self._request_pipe = (req_parent, req_child)
            self._result_pipe = (res_parent, res_child)
            self._heartbeat_pipe = (hb_parent, hb_child)

            # Launch subprocess
            self._process = multiprocessing.Process(
                target=self._worker_fn,
                args=(
                    self._config.model_id,
                    req_child,
                    res_child,
                    hb_child,
                    self._config.heartbeat_interval_seconds,
                    self._config.memory_limit_mb,
                ),
                daemon=True,
                name=f"yunshu-worker-{self._config.model_id}",
            )
            self._process.start()
            self._start_time = time.monotonic()
            self._last_heartbeat = time.monotonic()

            # Start result reader thread
            self._result_thread_running = True
            self._result_thread = threading.Thread(
                target=self._result_reader_loop,
                name=f"yunshu-result-{self._config.model_id}",
                daemon=True,
            )
            self._result_thread.start()

            # Start heartbeat monitor thread
            self._heartbeat_thread_running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_monitor_loop,
                name=f"yunshu-hb-{self._config.model_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()

            self._state = WorkerState.RUNNING
            logger.info(
                "Worker %s started (pid=%d)",
                self._config.model_id,
                self._process.pid,
            )
        except Exception:
            self._state = WorkerState.CRASHED
            logger.exception("Failed to start worker %s", self._config.model_id)
            raise

    def stop(self) -> None:
        """Gracefully stop the worker subprocess."""
        if self._state == WorkerState.STOPPED:
            return
        if self._state == WorkerState.IDLE:
            self._state = WorkerState.STOPPED
            return

        self._state = WorkerState.STOPPING

        # Signal shutdown via request pipe
        try:
            if self._request_pipe is not None:
                parent_conn = self._request_pipe[0]
                parent_conn.send(None)  # shutdown signal
        except (BrokenPipeError, OSError):
            pass

        # Stop threads
        self._heartbeat_thread_running = False
        self._result_thread_running = False

        # Wait for threads
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        if self._result_thread is not None:
            self._result_thread.join(timeout=2.0)

        # Terminate process
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.0)

        # Close pipes
        self._close_pipes()

        # Fail all pending requests
        with self._pending_lock:
            for req_id, result in self._pending.items():
                result.set_error(
                    WorkerCrashError(
                        f"Worker stopped while request {req_id} pending",
                        model_id=self._config.model_id,
                    )
                )
            self._pending.clear()

        self._state = WorkerState.STOPPED
        logger.info("Worker %s stopped", self._config.model_id)

    def restart(self) -> None:
        """Stop and re-start the worker (e.g. after a crash)."""
        self.stop()
        self.start()

    def submit_request(self, request: dict) -> IsolatedResult:
        """Submit a request to the worker subprocess.

        Returns an IsolatedResult that can be awaited for the response.
        """
        if self._state != WorkerState.RUNNING:
            raise WorkerCrashError(
                f"Worker {self._config.model_id} is not running "
                f"(state={self._state.name})",
                model_id=self._config.model_id,
            )

        request_id = str(uuid.uuid4())
        result = IsolatedResult(request_id=request_id)

        with self._pending_lock:
            self._pending[request_id] = result

        try:
            parent_conn = self._request_pipe[0]
            parent_conn.send((request_id, request))
        except (BrokenPipeError, OSError) as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._record_crash()
            raise WorkerCrashError(
                f"Worker {self._config.model_id} pipe broken during submit",
                model_id=self._config.model_id,
            ) from exc

        return result

    def is_healthy(self) -> bool:
        """Check if the worker is healthy based on heartbeat + crash count."""
        if self._state != WorkerState.RUNNING:
            return False

        # Check heartbeat freshness (allow 3x interval before declaring unhealthy)
        heartbeat_age = time.monotonic() - self._last_heartbeat
        if heartbeat_age > self._config.heartbeat_interval_seconds * 3:
            return False

        # Check if circuit breaker is tripped
        if self._is_circuit_open():
            return False

        # Check subprocess is alive
        if self._process is not None and not self._process.is_alive():
            return False

        return True

    def _is_circuit_open(self) -> bool:
        """Check if the circuit breaker should prevent operations."""
        self._prune_crash_times()
        return len(self._crash_times) >= self._config.max_restarts

    def _prune_crash_times(self) -> None:
        """Remove crash timestamps outside the sliding window."""
        cutoff = time.monotonic() - self._config.restart_window_seconds
        while self._crash_times and self._crash_times[0] < cutoff:
            self._crash_times.popleft()

    def _record_crash(self) -> None:
        """Record a crash event and update stats."""
        now = time.monotonic()
        self._crash_times.append(now)
        self._stats.crash_count += 1
        self._stats.last_crash_time = now

        if self._is_circuit_open():
            self._state = WorkerState.CIRCUIT_OPEN
            logger.warning(
                "Circuit breaker tripped for worker %s "
                "(%d crashes in %.0fs)",
                self._config.model_id,
                len(self._crash_times),
                self._config.restart_window_seconds,
            )

    def _close_pipes(self) -> None:
        """Close all pipe connections."""
        for pipe in (self._request_pipe, self._result_pipe, self._heartbeat_pipe):
            if pipe is not None:
                for conn in pipe:
                    try:
                        conn.close()
                    except OSError:
                        pass
        self._request_pipe = None
        self._result_pipe = None
        self._heartbeat_pipe = None

    def _result_reader_loop(self) -> None:
        """Background thread: reads results from the subprocess pipe."""
        if self._result_pipe is None:
            return
        parent_conn = self._result_pipe[0]
        try:
            while self._result_thread_running:
                if parent_conn.poll(timeout=0.5):
                    try:
                        msg = parent_conn.recv()
                    except EOFError:
                        break

                    if isinstance(msg, dict):
                        request_id = msg.get("request_id")
                        with self._pending_lock:
                            result = self._pending.pop(request_id, None)

                        if result is not None:
                            error = msg.get("error")
                            if error is not None:
                                result.set_error(
                                    WorkerCrashError(
                                        str(error),
                                        model_id=self._config.model_id,
                                    )
                                )
                            else:
                                result.set_result(msg.get("result"))
                                self._stats.request_count += 1
                                latency = msg.get("latency_ms", 0)
                                self._stats.total_latency_ms += latency
                                self._stats.last_request_time = time.time()
        except (OSError, BrokenPipeError):
            pass
        finally:
            # If we exit due to pipe closure, the worker has crashed
            if self._result_thread_running and self._state == WorkerState.RUNNING:
                self._record_crash()
                self._state = WorkerState.CRASHED

    def _heartbeat_monitor_loop(self) -> None:
        """Background thread: reads heartbeats from the subprocess pipe."""
        if self._heartbeat_pipe is None:
            return
        parent_conn = self._heartbeat_pipe[0]
        try:
            while self._heartbeat_thread_running:
                if parent_conn.poll(timeout=self._config.heartbeat_interval_seconds):
                    try:
                        msg = parent_conn.recv()
                    except EOFError:
                        break

                    if isinstance(msg, tuple) and msg[0] == "ping":
                        self._last_heartbeat = time.monotonic()
                    elif isinstance(msg, tuple) and msg[0] == "shutdown":
                        break
        except (OSError, BrokenPipeError):
            pass


# ── WorkerSupervisor ──


class WorkerSupervisor:
    """Manages multiple InferenceWorkers with health monitoring and auto-restart.

    Provides:
    - Worker registration / deregistration
    - Periodic health checks with auto-restart
    - Circuit breaker per worker (stops restart loops)
    - Fallback to in-process mode when workers are unhealthy
    - Stats aggregation for monitoring

    Usage:
        supervisor = WorkerSupervisor()
        supervisor.register_worker("model-a", worker)
        healthy = supervisor.get_healthy_worker("model-a")
    """

    def __init__(
        self,
        heartbeat_check_interval: float = 5.0,
    ) -> None:
        self._workers: dict[str, InferenceWorker] = {}
        self._configs: dict[str, WorkerProcessConfig] = {}
        self._fallbacks: dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._heartbeat_check_interval = heartbeat_check_interval
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running: bool = False
        self._started: bool = False

    def start_monitoring(self) -> None:
        """Start the background health monitoring thread."""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._started = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="yunshu-supervisor-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Supervisor monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._monitor_running = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def register_worker(
        self,
        model_id: str,
        worker: InferenceWorker,
        fallback_fn: Optional[Callable] = None,
    ) -> None:
        """Register a worker for supervision.

        Args:
            model_id: Model identifier.
            worker: The InferenceWorker instance.
            fallback_fn: Optional callable for in-process fallback when
                the worker is unhealthy.
        """
        with self._lock:
            self._workers[model_id] = worker
            self._configs[model_id] = worker.config
            if fallback_fn is not None:
                self._fallbacks[model_id] = fallback_fn
        logger.info("Registered worker for model '%s'", model_id)

    def deregister_worker(self, model_id: str) -> None:
        """Remove a worker from supervision and stop it."""
        with self._lock:
            worker = self._workers.pop(model_id, None)
            self._configs.pop(model_id, None)
            self._fallbacks.pop(model_id, None)

        if worker is not None:
            try:
                worker.stop()
            except Exception:
                logger.exception(
                    "Error stopping worker during deregister for '%s'",
                    model_id,
                )
        logger.info("Deregistered worker for model '%s'", model_id)

    def get_healthy_worker(self, model_id: str) -> Optional[InferenceWorker]:
        """Return a healthy worker for the given model, or None."""
        with self._lock:
            worker = self._workers.get(model_id)
        if worker is not None and worker.is_healthy():
            return worker
        return None

    def fallback_to_inprocess(self, model_id: str, request: dict) -> Any:
        """Execute a request using the in-process fallback.

        Raises WorkerCrashError if no fallback is registered.
        """
        with self._lock:
            fallback_fn = self._fallbacks.get(model_id)

        if fallback_fn is None:
            raise WorkerCrashError(
                f"No in-process fallback registered for model '{model_id}'",
                model_id=model_id,
            )

        return fallback_fn(request)

    def health_check(self) -> dict[str, dict[str, Any]]:
        """Check health of all registered workers.

        Returns:
            Dict mapping model_id → health status dict.
        """
        result: dict[str, dict[str, Any]] = {}
        with self._lock:
            model_ids = list(self._workers.keys())

        for model_id in model_ids:
            with self._lock:
                worker = self._workers.get(model_id)

            if worker is None:
                continue

            healthy = worker.is_healthy()
            result[model_id] = {
                "healthy": healthy,
                "state": worker.state.name,
                "model_id": model_id,
                "stats": worker.stats.to_dict(),
                "circuit_open": worker._is_circuit_open(),
                "has_fallback": model_id in self._fallbacks,
            }

        return result

    def _auto_restart_unhealthy(self) -> None:
        """Check all workers and restart any that are unhealthy."""
        with self._lock:
            items = list(self._workers.items())

        for model_id, worker in items:
            if worker.state == WorkerState.CIRCUIT_OPEN:
                continue

            if not worker.is_healthy() and worker.state not in (
                WorkerState.STOPPING,
                WorkerState.STARTING,
            ):
                logger.warning(
                    "Worker '%s' unhealthy (state=%s), attempting restart",
                    model_id,
                    worker.state.name,
                )
                try:
                    worker.restart()
                    logger.info("Worker '%s' restarted successfully", model_id)
                except Exception:
                    logger.exception(
                        "Failed to restart worker '%s'", model_id
                    )
                    worker._record_crash()

    def _monitor_loop(self) -> None:
        """Background monitoring loop: periodic health checks + auto-restart."""
        while self._monitor_running:
            try:
                self._auto_restart_unhealthy()
            except Exception:
                logger.exception("Error in supervisor monitor loop")
            time.sleep(self._heartbeat_check_interval)

    def submit(self, model_id: str, request: dict) -> IsolatedResult:
        """Submit a request to the isolated worker, with fallback.

        If the worker is unhealthy and a fallback is registered, uses
        in-process fallback instead.
        """
        worker = self.get_healthy_worker(model_id)
        if worker is not None:
            return worker.submit_request(request)

        # Try fallback
        logger.warning(
            "No healthy worker for '%s', falling back to in-process",
            model_id,
        )
        result = IsolatedResult(request_id=str(uuid.uuid4()))
        try:
            fallback_result = self.fallback_to_inprocess(model_id, request)
            result.set_result(fallback_result)
        except Exception as exc:
            result.set_error(exc)
        return result

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all registered workers."""
        stats: dict[str, dict[str, Any]] = {}
        with self._lock:
            for model_id, worker in self._workers.items():
                stats[model_id] = worker.stats.to_dict()
        return stats

    def shutdown(self) -> None:
        """Stop all workers and the monitoring thread."""
        self.stop_monitoring()
        with self._lock:
            model_ids = list(self._workers.keys())

        for model_id in model_ids:
            self.deregister_worker(model_id)

    @property
    def worker_count(self) -> int:
        with self._lock:
            return len(self._workers)

    @property
    def is_started(self) -> bool:
        return self._started


# ── Module-level Singleton ──

_supervisor: Optional[WorkerSupervisor] = None
_supervisor_lock = threading.Lock()


def get_supervisor() -> WorkerSupervisor:
    """Get or create the module-level WorkerSupervisor singleton."""
    global _supervisor
    if _supervisor is None:
        with _supervisor_lock:
            if _supervisor is None:
                _supervisor = WorkerSupervisor()
    return _supervisor


def reset_supervisor() -> None:
    """Reset the singleton supervisor (for testing)."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is not None:
            _supervisor.shutdown()
        _supervisor = None


def is_isolation_enabled() -> bool:
    """Check if process isolation is enabled via environment variable."""
    return os.environ.get("YUNSHU_PROCESS_ISOLATION", "0") == "1"


# ── Integration Hook for BatchedEngine ──


def maybe_isolate_inference(
    model_id: str,
    request: dict,
    fallback_fn: Callable,
) -> Any:
    """Integration point for BatchedEngine.

    When YUNSHU_PROCESS_ISOLATION=1, submits the request to an isolated worker.
    Otherwise, runs the fallback (in-process) function directly.

    Returns the inference result or raises on error.
    """
    if not is_isolation_enabled():
        return fallback_fn(request)

    supervisor = get_supervisor()
    worker = supervisor.get_healthy_worker(model_id)
    if worker is not None:
        result_future = worker.submit_request(request)
        return result_future.result(
            timeout=worker.config.request_timeout_seconds
        )

    # No healthy worker — try in-process fallback
    return supervisor.fallback_to_inprocess(model_id, request)
