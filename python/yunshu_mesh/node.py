from __future__ import annotations
"""Yunshu Mesh — Node representation and discovery.

Each node is a Mac (Mac Studio, MacBook Pro, Mac Mini) in the cluster.
Nodes discover each other via mDNS on the local network.

Thread safety:
  MeshNode.state, last_heartbeat, _active_requests, and _loaded_models are
  accessed from multiple threads (heartbeat sender/receiver/checker, discovery,
  API handlers).  All mutations go through a per-node Lock so that readers
  always see a consistent snapshot.
"""


import hashlib
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class MeshNodeState(Enum):
    INITIALIZING = auto()
    READY = auto()
    BUSY = auto()
    DRAINING = auto()
    OFFLINE = auto()
    RECOVERING = auto()


# ── Valid state transitions ──────────────────────────────────────────
# The state machine guarantees that nodes cannot skip the RECOVERING
# phase when coming back from OFFLINE.
#
#   INITIALIZING → READY
#   READY → BUSY | DRAINING | OFFLINE
#   BUSY → READY | DRAINING | OFFLINE
#   DRAINING → OFFLINE
#   OFFLINE → RECOVERING          ← must go through RECOVERING first
#   RECOVERING → READY            ← health verification succeeded
#   RECOVERING → OFFLINE          ← health verification failed
_VALID_TRANSITIONS: dict[MeshNodeState, set[MeshNodeState]] = {
    MeshNodeState.INITIALIZING: {
        MeshNodeState.READY, MeshNodeState.OFFLINE,
    },
    MeshNodeState.READY: {
        MeshNodeState.BUSY, MeshNodeState.DRAINING,
        MeshNodeState.OFFLINE,
    },
    MeshNodeState.BUSY: {
        MeshNodeState.READY, MeshNodeState.DRAINING,
        MeshNodeState.OFFLINE,
    },
    MeshNodeState.DRAINING: {
        MeshNodeState.OFFLINE,
    },
    MeshNodeState.OFFLINE: {
        MeshNodeState.RECOVERING,  # cannot go directly to READY
    },
    MeshNodeState.RECOVERING: {
        MeshNodeState.READY, MeshNodeState.OFFLINE,
    },
}


@dataclass
class NodeCapabilities:
    """Hardware capabilities of a mesh node."""
    total_memory_gb: float = 0.0
    gpu_cores: int = 0
    cpu_cores: int = 0
    chip: str = ""
    thunderbolt_ports: int = 0
    supports_jaccl: bool = False

    @staticmethod
    def detect() -> NodeCapabilities:
        """Auto-detect local node capabilities."""
        import subprocess

        gpu_cores = 0
        chip = ""
        tb_ports = 0
        total_mem_gb = 0.0

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                if "Chipset Model" in line:
                    chip = line.split(":")[-1].strip()
                elif "Total Number of Cores" in line:
                    gpu_cores = int(line.split(":")[-1].strip())
        except Exception:
            logger.debug("failed to detect GPU via system_profiler", exc_info=True)

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize", "hw.ncpu"],
                capture_output=True, text=True,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                total_mem_gb = int(lines[0]) / (1024 ** 3)
                cpu_cores = int(lines[1])
        except Exception:
            logger.debug("failed to detect memory/CPU via sysctl", exc_info=True)
            cpu_cores = 0

        # Thunderbolt detection
        try:
            result = subprocess.run(
                ["system_profiler", "SPThunderboltDataType"],
                capture_output=True, text=True, timeout=5,
            )
            tb_ports = result.stdout.count("Thunderbolt")
        except Exception:
            logger.debug("failed to detect Thunderbolt ports", exc_info=True)

        return NodeCapabilities(
            total_memory_gb=total_mem_gb,
            gpu_cores=gpu_cores,
            cpu_cores=cpu_cores,
            chip=chip,
            thunderbolt_ports=tb_ports,
            supports_jaccl=tb_ports > 0 and total_mem_gb >= 64,
        )


@dataclass
class MeshNode:
    """A node in the compute mesh.

    Follows oMLX's node representation but with JACCL-aware capabilities
    and proper state management.

    Thread safety:
      The ``state`` field is guarded by a per-node ``threading.Lock``.
      Direct assignment (``node.state = X``) still works for backward
      compatibility but callers in multi-threaded contexts should prefer
      ``set_state()``, ``mark_healthy()``, and ``mark_unhealthy()`` which
      are atomic and validate state-machine transitions.
    """
    node_id: str = ""
    hostname: str = ""
    ip: str = ""
    port: int = 8000
    state: MeshNodeState = MeshNodeState.INITIALIZING
    capabilities: NodeCapabilities = field(default_factory=NodeCapabilities)
    joined_at: float = field(default_factory=time.monotonic)
    last_heartbeat: float = field(default_factory=time.monotonic)
    rank: int = -1  # Assigned rank in the distributed group

    # Runtime
    _loaded_models: list[str] = field(default_factory=list)
    _active_requests: int = 0

    # Internal lock — not serialized, created fresh by __post_init__.
    _lock: threading.Lock = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # ── Thread-safe state transitions ───────────────────────────

    def set_state(self, new_state: MeshNodeState, *, force: bool = False) -> bool:
        """Transition to *new_state* atomically.

        Args:
            new_state: Target state.
            force: If True, skip the state-machine validation (used for
                   initial construction and deserialization).

        Returns:
            True if the transition was applied, False if it was rejected
            by the state machine.
        """
        with self._lock:
            old = self.state
            if old == new_state:
                return True  # idempotent
            if not force:
                allowed = _VALID_TRANSITIONS.get(old, set())
                if new_state not in allowed:
                    logger.warning(
                        "Invalid state transition on node %s: %s → %s "
                        "(rejected)",
                        self.node_id, old.name, new_state.name,
                    )
                    return False
            object.__setattr__(self, 'state', new_state)
            return True

    # ── Thread-safe heartbeat ───────────────────────────────────

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        with self._lock:
            object.__setattr__(self, 'last_heartbeat', time.monotonic())

    # ── High-level thread-safe mutations ────────────────────────

    def mark_healthy(self) -> bool:
        """Mark the node as healthy (RECOVERING → READY).

        If the node is currently OFFLINE, transitions to RECOVERING first.
        If already RECOVERING, transitions to READY.
        If already READY, no-op.

        Returns True if the node ended up in READY state.
        """
        with self._lock:
            cur = self.state
            if cur == MeshNodeState.READY:
                return True
            if cur == MeshNodeState.OFFLINE:
                object.__setattr__(self, 'state', MeshNodeState.RECOVERING)
                cur = MeshNodeState.RECOVERING
                logger.info(
                    "Node %s: OFFLINE → RECOVERING (health verification "
                    "pending)", self.node_id,
                )
            if cur == MeshNodeState.RECOVERING:
                object.__setattr__(self, 'state', MeshNodeState.READY)
                logger.info(
                    "Node %s: RECOVERING → READY", self.node_id,
                )
                return True
            # BUSY, DRAINING, INITIALIZING — these can also go to READY
            # via the state machine.
            allowed = _VALID_TRANSITIONS.get(cur, set())
            if MeshNodeState.READY in allowed:
                object.__setattr__(self, 'state', MeshNodeState.READY)
                return True
            return False

    def mark_unhealthy(self, reason: str = "") -> None:
        """Transition node to OFFLINE atomically.

        Accepts any state (force=True) because a node can become
        unhealthy at any time.
        """
        with self._lock:
            old = self.state
            object.__setattr__(self, 'state', MeshNodeState.OFFLINE)
            if old != MeshNodeState.OFFLINE:
                logger.info(
                    "Node %s: %s → OFFLINE%s",
                    self.node_id, old.name,
                    f" ({reason})" if reason else "",
                )

    def update_load(self, active_requests: int, loaded_models: list[str] | None = None) -> None:
        """Atomically update load counters."""
        with self._lock:
            self._active_requests = active_requests
            if loaded_models is not None:
                self._loaded_models = list(loaded_models)

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def local(port: int = 8000) -> MeshNode:
        """Create a MeshNode for the local machine."""
        hostname = socket.gethostname()
        node_id = hashlib.sha256(hostname.encode()).hexdigest()[:12]
        caps = NodeCapabilities.detect()

        node = MeshNode(
            node_id=node_id,
            hostname=hostname,
            ip=MeshNode._get_local_ip(),
            port=port,
            state=MeshNodeState.READY,
            capabilities=caps,
        )
        return node

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local IP address for mesh networking."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            return ip
        except Exception:
            logger.debug("failed to detect local IP", exc_info=True)
            return "127.0.0.1"

    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if node is healthy based on heartbeat."""
        with self._lock:
            return (
                self.state not in (MeshNodeState.OFFLINE, MeshNodeState.RECOVERING)
                and time.monotonic() - self.last_heartbeat < timeout
            )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "node_id": self.node_id,
                "hostname": self.hostname,
                "ip": self.ip,
                "port": self.port,
                "state": self.state.name,
                "rank": self.rank,
                "capabilities": {
                    "total_memory_gb": self.capabilities.total_memory_gb,
                    "gpu_cores": self.capabilities.gpu_cores,
                    "cpu_cores": self.capabilities.cpu_cores,
                    "chip": self.capabilities.chip,
                    "thunderbolt_ports": self.capabilities.thunderbolt_ports,
                    "supports_jaccl": self.capabilities.supports_jaccl,
                },
                "loaded_models": list(self._loaded_models),
                "active_requests": self._active_requests,
            }

    @staticmethod
    def from_dict(data: dict) -> MeshNode:
        caps_data = data.get("capabilities", {})
        caps = NodeCapabilities(**caps_data)
        state_name = data.get("state", "INITIALIZING")
        try:
            parsed_state = MeshNodeState[state_name]
        except KeyError:
            logger.warning("Unknown state %r in from_dict, defaulting to INITIALIZING", state_name)
            parsed_state = MeshNodeState.INITIALIZING
        node = MeshNode(
            node_id=data["node_id"],
            hostname=data.get("hostname", ""),
            ip=data.get("ip", ""),
            port=data.get("port", 8000),
            state=parsed_state,
            rank=data.get("rank", -1),
            capabilities=caps,
            _loaded_models=data.get("loaded_models", []),
            _active_requests=data.get("active_requests", 0),
            joined_at=time.monotonic(),
            last_heartbeat=time.monotonic(),
        )
        return node
