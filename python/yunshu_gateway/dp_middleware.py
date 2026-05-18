from __future__ import annotations
"""Yunshu Gateway — DataParallel middleware.

Bridges the DataParallelRouter (yunshu_mesh.data_parallel) into the FastAPI
request path so that inference requests are load-balanced across engine
replicas and the request lifecycle is properly tracked.

Activation: YUNSHU_DATA_PARALLEL=1

The middleware is intentionally thin — it:
1. Intercepts inference requests (chat/completions, text completions, etc.)
2. Stores DP node selection in request.state so downstream routers can use it
3. Records request start/end lifecycle on the DataParallelRouter
4. Adds tracing headers (X-DP-Node, X-DP-Latency) to responses
5. Performs health checking on nodes that return errors

The actual node selection happens in engine/__init__.py:get_engine_for_model()
which calls _dp_router.select_node(). This middleware wraps the full request
lifecycle so that record_request_end() is always called.
"""

import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

# Inference paths that should be tracked by the DP load balancer
_INFERENCE_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/audio/speech",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/responses",
    "/v1/messages",
    "/v1/batch/inference",
    "/v1/tokenize",
})


def _is_inference_path(path: str) -> bool:
    """Check if a path should be tracked for DP load balancing."""
    return path in _INFERENCE_PATHS or path.startswith("/v1/chat/") or path.startswith("/v1/batch/")


# ---------------------------------------------------------------------------
# DPLoadBalancer — tracks per-node request counts, latencies, health
# ---------------------------------------------------------------------------

@dataclass
class NodeHealth:
    """Track health state for a single DP node."""
    node_id: str
    consecutive_errors: int = 0
    last_error_time: float = 0.0
    last_success_time: float = 0.0
    marked_unhealthy: bool = False
    # Health check: if a node returns this many consecutive errors, mark unhealthy
    error_threshold: int = 5
    # Recovery: after this many seconds, allow retry
    recovery_timeout_seconds: float = 30.0


class DPLoadBalancer:
    """Load balancer layer on top of DataParallelRouter.

    Adds:
    - Request lifecycle tracking (record_start / record_end)
    - Per-node latency histogram (EMA)
    - Health checking: auto-removes nodes with consecutive errors
    - Stats reporting for monitoring endpoint
    """

    def __init__(self, dp_router=None):
        self._dp_router = dp_router
        self._lock = threading.Lock()
        self._node_health: dict[str, NodeHealth] = {}
        # Per-node latency tracking
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._max_latency_samples = 100  # keep last N samples per node
        # Global counters
        self._total_requests = 0
        self._total_routed = 0

    @property
    def dp_router(self):
        return self._dp_router

    @dp_router.setter
    def dp_router(self, value):
        self._dp_router = value

    def register_node(self, node_id: str) -> None:
        """Register a node for health tracking."""
        with self._lock:
            if node_id not in self._node_health:
                self._node_health[node_id] = NodeHealth(node_id=node_id)

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from health tracking."""
        with self._lock:
            self._node_health.pop(node_id, None)
            self._latencies.pop(node_id, None)

    def select_node(self, model_id: str | None = None) -> Optional[str]:
        """Select the best node, considering health state.

        Delegates to DataParallelRouter.select_node() but checks health
        state and recovers nodes whose recovery timeout has elapsed.
        """
        if self._dp_router is None:
            return None

        # Proactively check for node recovery before selection
        self._check_recoveries()

        node_id = self._dp_router.select_node()

        with self._lock:
            self._total_requests += 1
            if node_id is not None:
                self._total_routed += 1

        return node_id

    def _check_recoveries(self) -> None:
        """Proactively check all unhealthy nodes for recovery eligibility.

        Nodes are only recovered after (a) the recovery timeout has elapsed
        AND (b) a lightweight health probe confirms the node is reachable.
        The probe runs outside the lock to avoid blocking other operations.
        """
        # Collect candidates while holding the lock, then probe outside
        candidates: list[str] = []
        with self._lock:
            for node_id, health in self._node_health.items():
                if health.marked_unhealthy:
                    elapsed = time.monotonic() - health.last_error_time
                    if elapsed > health.recovery_timeout_seconds:
                        candidates.append(node_id)

        # Probe each candidate without holding the lock
        for node_id in candidates:
            if self._probe_node_health(node_id):
                with self._lock:
                    health = self._node_health.get(node_id)
                    if health is not None and health.marked_unhealthy:
                        health.marked_unhealthy = False
                        health.consecutive_errors = 0
                        if self._dp_router is not None:
                            self._dp_router.mark_available(node_id)
                        logger.info(f"DP node {node_id} recovered after health probe succeeded")
            else:
                logger.debug(f"DP node {node_id} recovery deferred — health probe failed")

    def _probe_node_health(self, node_id: str) -> bool:
        """Lightweight health check: try HTTP GET /health on the node.

        Returns True if the node responds 200, False otherwise.
        Falls back to always-True (time-based recovery) if httpx is
        unavailable or the node ID does not look like a network address.
        """
        try:
            import httpx
        except ImportError:
            # httpx not available — fall back to time-based recovery
            logger.debug("httpx not available, skipping health probe for %s", node_id)
            return True

        # Parse host from node_id — formats: "host:port", "host:rank", etc.
        host = node_id.split(":")[0] if ":" in node_id else node_id

        # Skip probe for local-only or synthetic node IDs that are not
        # network-reachable.  Synthetic IDs like "node-0", "gpu-1", "alpha"
        # default to True (time-based recovery) because they have no real
        # HTTP endpoint to probe.
        if host in ("local", "localhost", "127.0.0.1", "::1"):
            return True
        # Heuristic: if the host looks like a real hostname or IP (contains
        # dots or is an IP-like pattern), probe it.  Otherwise assume it's
        # a synthetic ID and return True.
        import re
        if not re.match(r'^[\d]+\.[\d]+\.[\d]+\.[\d]+$|^[\w][\w.-]*\.[\w]+$', host):
            return True

        # Derive a health URL
        parts = node_id.rsplit(":", 1)
        if len(parts) == 2:
            try:
                port = int(parts[1])
                url = f"http://{host}:{port}/health"
            except ValueError:
                url = f"http://{host}:8000/health"
        else:
            url = f"http://{host}:8000/health"

        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    def record_start(self, node_id: str) -> None:
        """Record that a request started on a node."""
        if self._dp_router is not None:
            self._dp_router.record_request_start(node_id)

    def record_end(self, node_id: str, latency_ms: float, success: bool = True) -> None:
        """Record that a request finished on a node.

        Updates latency tracking and health state.
        """
        if self._dp_router is not None:
            self._dp_router.record_request_end(node_id, latency_ms)

        with self._lock:
            health = self._node_health.get(node_id)
            if health is None:
                return

            # Track latency
            self._latencies[node_id].append(latency_ms)
            if len(self._latencies[node_id]) > self._max_latency_samples:
                self._latencies[node_id] = self._latencies[node_id][-self._max_latency_samples // 2:]

            if success:
                health.consecutive_errors = 0
                health.last_success_time = time.monotonic()
            else:
                health.consecutive_errors += 1
                health.last_error_time = time.monotonic()

                # Check if node should be marked unhealthy
                if health.consecutive_errors >= health.error_threshold:
                    health.marked_unhealthy = True
                    if self._dp_router is not None:
                        self._dp_router.mark_unavailable(node_id)
                    logger.warning(
                        f"DP node {node_id} marked unhealthy after "
                        f"{health.consecutive_errors} consecutive errors"
                    )

    def mark_unhealthy(self, node_id: str) -> None:
        """Manually mark a node as unhealthy."""
        with self._lock:
            health = self._node_health.get(node_id)
            if health:
                health.marked_unhealthy = True
                health.last_error_time = time.monotonic()
        if self._dp_router is not None:
            self._dp_router.mark_unavailable(node_id)

    def mark_healthy(self, node_id: str) -> None:
        """Manually mark a node as healthy (re-enable it)."""
        with self._lock:
            health = self._node_health.get(node_id)
            if health:
                health.marked_unhealthy = False
                health.consecutive_errors = 0
        if self._dp_router is not None:
            self._dp_router.mark_available(node_id)

    def get_stats(self) -> dict:
        """Return comprehensive DP load balancer stats."""
        with self._lock:
            nodes = {}
            for node_id, health in self._node_health.items():
                lats = self._latencies.get(node_id, [])
                nodes[node_id] = {
                    "healthy": not health.marked_unhealthy,
                    "consecutive_errors": health.consecutive_errors,
                    "last_error_time": health.last_error_time,
                    "last_success_time": health.last_success_time,
                    "avg_latency_ms": sum(lats) / len(lats) if lats else 0.0,
                    "p50_latency_ms": sorted(lats)[len(lats) // 2] if lats else 0.0,
                    "p99_latency_ms": sorted(lats)[min(int(len(lats) * 0.99), len(lats) - 1)] if lats else 0.0,
                    "sample_count": len(lats),
                }
            total_req = self._total_requests
            total_routed = self._total_routed

        # Merge with DP router stats
        router_stats = {}
        if self._dp_router is not None:
            router_stats = self._dp_router.get_stats()

        return {
            "active": self._dp_router is not None,
            "total_requests": total_req,
            "total_routed": total_routed,
            "node_health": nodes,
            "router": router_stats,
        }


# ---------------------------------------------------------------------------
# DPRouterMiddleware — FastAPI middleware
# ---------------------------------------------------------------------------

# Module-level singleton
_dp_load_balancer: DPLoadBalancer | None = None


def get_dp_load_balancer() -> DPLoadBalancer | None:
    """Get the global DP load balancer (or None if not initialized)."""
    return _dp_load_balancer


def init_dp_load_balancer(dp_router=None) -> DPLoadBalancer:
    """Initialize the global DP load balancer.

    Called during gateway startup when YUNSHU_DATA_PARALLEL=1.
    """
    global _dp_load_balancer
    _dp_load_balancer = DPLoadBalancer(dp_router=dp_router)
    return _dp_load_balancer


def setup_data_parallel(
    strategy: str | None = None,
    nodes: list[tuple[str, int]] | None = None,
) -> DPLoadBalancer:
    """Set up the full data-parallel stack.

    Creates DataParallelRouter, DPLoadBalancer, and registers nodes.
    Called from main.py during startup when YUNSHU_DATA_PARALLEL=1.

    Args:
        strategy: Routing strategy (round_robin, least_loaded, latency_aware, capacity_aware).
                  Defaults to "least_loaded".
        nodes: List of (node_id, rank) tuples to register.
               If None, nodes are discovered from YUNSHU_DP_NODES env var
               or default to a single local node.
    """
    from .engine import init_data_parallel

    if strategy is None:
        strategy = os.environ.get("YUNSHU_DP_STRATEGY", "least_loaded")

    dp_router = init_data_parallel(strategy=strategy)

    lb = init_dp_load_balancer(dp_router=dp_router)

    # Register nodes
    if nodes is None:
        nodes_str = os.environ.get("YUNSHU_DP_NODES", "")
        if nodes_str:
            # Format: "node0:0,node1:1,node2:2"
            for entry in nodes_str.split(","):
                entry = entry.strip()
                if ":" in entry:
                    nid, rank_str = entry.rsplit(":", 1)
                    try:
                        rank = int(rank_str)
                    except ValueError:
                        rank = 0
                else:
                    nid = entry
                    rank = 0
                dp_router.add_node(nid, rank)
                lb.register_node(nid)
        else:
            # Single local node
            dp_router.add_node("local:0", rank=0)
            lb.register_node("local:0")
    else:
        for node_id, rank in nodes:
            dp_router.add_node(node_id, rank)
            lb.register_node(node_id)

    logger.info(
        f"DataParallel middleware initialized: strategy={strategy}, "
        f"nodes={dp_router.num_nodes}"
    )
    return lb


class DPRouterMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that wraps inference requests with DP lifecycle tracking.

    For each inference request:
    1. Selects a DP node (via DPLoadBalancer.select_node)
    2. Stores node_id in request.state.dp_node_id
    3. Records request start
    4. After response: records request end with latency
    5. Adds X-DP-Node and X-DP-Latency tracing headers
    6. On error: updates node health tracking

    The actual engine routing uses the same node_id stored in request.state
    (via engine/__init__.py's get_engine_for_model which reads _dp_load_balancer).
    """

    async def dispatch(self, request: Request, call_next):
        lb = _dp_load_balancer

        # Only track inference paths
        if lb is None or not _is_inference_path(request.url.path):
            return await call_next(request)

        t0 = time.monotonic()

        # Select node for this request
        node_id = lb.select_node(
            model_id=request.query_params.get("model")
        )

        # Store in request state so downstream code can access it
        request.state.dp_node_id = node_id
        request.state.dp_start_time = t0

        # Process request
        try:
            response = await call_next(request)
        except Exception:
            logger.debug("operation failed", exc_info=True)
            # Request failed at the framework level
            latency_ms = (time.monotonic() - t0) * 1000
            if node_id is not None:
                lb.record_end(node_id, latency_ms, success=False)
            raise

        # Record completion
        latency_ms = (time.monotonic() - t0) * 1000
        success = response.status_code < 500

        if node_id is not None:
            lb.record_end(node_id, latency_ms, success=success)

        # Add tracing headers
        if node_id is not None:
            response.headers["X-DP-Node"] = node_id
            response.headers["X-DP-Latency"] = f"{latency_ms:.2f}"

        return response
