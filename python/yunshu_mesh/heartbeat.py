"""Yunshu Mesh — Heartbeat monitoring for cluster health.

UDP-based heartbeat protocol for detecting node failures:
- Nodes send heartbeat packets every `interval` seconds
- Missing heartbeat for `timeout` seconds → node marked unhealthy
- Recovery detected when timed-out node sends heartbeat again
"""

import json
import logging
import socket
import threading
import time
from typing import Any, Callable, Optional

from .node import MeshNode, MeshNodeState

logger = logging.getLogger(__name__)

_HEARTBEAT_PORT = 7999


class HeartbeatMonitor:
    """Monitors mesh node health via periodic UDP heartbeats.

    Each node sends a small JSON packet with its status.
    The monitor tracks last_heartbeat per peer and fires callbacks
    on timeout or recovery.
    """

    def __init__(
        self,
        interval: float = 5.0,
        timeout: float = 30.0,
        port: int = _HEARTBEAT_PORT,
        failure_threshold: int = 3,
    ):
        self.interval = interval
        self.timeout = timeout
        self.port = port
        self.failure_threshold = failure_threshold
        self._nodes: dict[str, MeshNode] = {}
        self._last_heartbeat: dict[str, float] = {}
        self._missed_counts: dict[str, int] = {}
        self._timed_out: set[str] = set()
        self._nodes_lock = threading.Lock()
        self._on_timeout_callbacks: list[Callable] = []
        self._on_recovery_callbacks: list[Callable] = []
        self._running = False
        self._local_node: Optional[MeshNode] = None
        self._socket: Optional[socket.socket] = None
        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._check_thread: Optional[threading.Thread] = None

    def start(self, local_node: MeshNode, peers: list[MeshNode]) -> None:
        self._local_node = local_node
        with self._nodes_lock:
            for peer in peers:
                self._nodes[peer.node_id] = peer
                self._last_heartbeat[peer.node_id] = time.monotonic()
                self._missed_counts[peer.node_id] = 0

        self._running = True
        self._setup_socket()
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()
        self._check_thread.start()
        logger.info(f"Heartbeat monitor started (interval={self.interval}s, timeout={self.timeout}s)")

    def _setup_socket(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket.bind(("", self.port))
        except OSError as e:
            logger.warning(f"Failed to bind heartbeat port {self.port}: {e}")
            return
        self._socket.settimeout(2.0)

    def stop(self) -> None:
        self._running = False
        # Wait for threads to exit BEFORE closing the socket to prevent
        # OSError: Bad file descriptor during concurrent send/recv.
        for t in (self._send_thread, self._recv_thread, self._check_thread):
            if t:
                t.join(timeout=3)
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                logger.debug("failed to close heartbeat socket", exc_info=True)
        logger.info("Heartbeat monitor stopped")

    def add_node(self, node: Any) -> None:
        """Add a node to heartbeat monitoring (for dynamically discovered peers)."""
        with self._nodes_lock:
            self._nodes[node.node_id] = node
            self._last_heartbeat[node.node_id] = time.monotonic()
            self._missed_counts[node.node_id] = 0
            self._timed_out.discard(node.node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from monitoring (called on permanent removal)."""
        with self._nodes_lock:
            self._nodes.pop(node_id, None)
            self._last_heartbeat.pop(node_id, None)
            self._missed_counts.pop(node_id, None)
            self._timed_out.discard(node_id)

    def _send_loop(self) -> None:
        while self._running:
            if self._local_node and self._socket:
                # Use to_dict() for a thread-safe snapshot instead of
                # reading individual fields without the node lock.
                node_snapshot = self._local_node.to_dict()
                msg = json.dumps({
                    "node_id": node_snapshot["node_id"],
                    "timestamp": time.monotonic(),
                    "state": node_snapshot["state"],
                    "active_requests": node_snapshot["active_requests"],
                }).encode()
                with self._nodes_lock:
                    peers = list(self._nodes.values())
                for peer in peers:
                    try:
                        self._socket.sendto(msg, (peer.ip, self.port))
                    except Exception:
                        logger.debug("failed to send heartbeat to peer", exc_info=True)
            time.sleep(self.interval)

    def _recv_loop(self) -> None:
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(4096)
                msg = json.loads(data.decode())
                node_id = msg.get("node_id")
                recovered_node = None
                with self._nodes_lock:
                    if node_id and node_id in self._nodes:
                        self._last_heartbeat[node_id] = time.monotonic()
                        self._missed_counts[node_id] = 0
                        node = self._nodes[node_id]
                        node.heartbeat()
                        try:
                            reported_state = MeshNodeState[msg.get("state", "READY")]
                        except (KeyError, ValueError):
                            logger.debug("Invalid node state from heartbeat: %s",
                                         msg.get("state"))
                            reported_state = None
                        if reported_state is not None:
                            # Use mark_healthy for recovery path
                            # (OFFLINE → RECOVERING → READY).
                            # For other state transitions, use set_state.
                            if reported_state == MeshNodeState.READY:
                                node.mark_healthy()
                            else:
                                node.set_state(reported_state, force=True)
                        node.update_load(msg.get("active_requests", 0))
                        # Recovery check — capture callback data, fire outside lock
                        if node_id in self._timed_out:
                            self._timed_out.discard(node_id)
                            logger.info(f"Node recovered: {node.hostname} ({node_id})")
                            recovered_node = node
                # Fire recovery callbacks outside lock to prevent deadlock
                if recovered_node is not None:
                    for cb in list(self._on_recovery_callbacks):
                        try:
                            cb(recovered_node)
                        except Exception:
                            logger.debug("on_recovery callback failed", exc_info=True)
            except socket.timeout:
                continue
            except Exception:
                logger.debug("failed to receive heartbeat packet", exc_info=True)
                continue

    def _check_loop(self) -> None:
        while self._running:
            now = time.monotonic()
            timed_out_nodes = []
            with self._nodes_lock:
                for node_id, last_hb in list(self._last_heartbeat.items()):
                    if node_id not in self._missed_counts:
                        self._missed_counts[node_id] = 0
                    if now - last_hb > self.timeout:
                        self._missed_counts[node_id] = self._missed_counts.get(node_id, 0) + 1
                        if (self._missed_counts[node_id] >= self.failure_threshold
                                and node_id not in self._timed_out):
                            self._timed_out.add(node_id)
                            node = self._nodes.get(node_id)
                            if node:
                                node.mark_unhealthy(reason="heartbeat_timeout")
                                logger.warning(f"Node timeout: {node.hostname} ({node_id})")
                                timed_out_nodes.append(node)
                    else:
                        self._missed_counts[node_id] = 0
            # Fire timeout callbacks outside lock to prevent deadlock
            for node in timed_out_nodes:
                for cb in list(self._on_timeout_callbacks):
                    try:
                        cb(node)
                    except Exception:
                        logger.debug("on_timeout callback failed", exc_info=True)
            time.sleep(self.interval)

    def check_health(self) -> dict[str, bool]:
        now = time.monotonic()
        with self._nodes_lock:
            return {
                node_id: (now - self._last_heartbeat.get(node_id, now)) < self.timeout
                for node_id in self._nodes
            }

    def on_node_timeout(self, callback: Callable) -> None:
        self._on_timeout_callbacks.append(callback)

    def on_node_recovered(self, callback: Callable) -> None:
        self._on_recovery_callbacks.append(callback)
