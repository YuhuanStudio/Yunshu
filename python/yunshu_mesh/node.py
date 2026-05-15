from __future__ import annotations
"""Yunshu Mesh — Node representation and discovery.

Each node is a Mac (Mac Studio, MacBook Pro, Mac Mini) in the cluster.
Nodes discover each other via mDNS on the local network.
"""


import hashlib
import logging
import socket
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
    """
    node_id: str = ""
    hostname: str = ""
    ip: str = ""
    port: int = 8000
    state: MeshNodeState = MeshNodeState.INITIALIZING
    capabilities: NodeCapabilities = field(default_factory=NodeCapabilities)
    joined_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    rank: int = -1  # Assigned rank in the distributed group

    # Runtime
    _loaded_models: list[str] = field(default_factory=list)
    _active_requests: int = 0

    @staticmethod
    def local(port: int = 8000) -> MeshNode:
        """Create a MeshNode for the local machine."""
        hostname = socket.gethostname()
        node_id = hashlib.sha256(hostname.encode()).hexdigest()[:12]
        caps = NodeCapabilities.detect()

        return MeshNode(
            node_id=node_id,
            hostname=hostname,
            ip=MeshNode._get_local_ip(),
            port=port,
            state=MeshNodeState.READY,
            capabilities=caps,
        )

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local IP address for mesh networking."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            logger.debug("failed to detect local IP", exc_info=True)
            return "127.0.0.1"

    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (
            self.state not in (MeshNodeState.OFFLINE,)
            and time.time() - self.last_heartbeat < timeout
        )

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def to_dict(self) -> dict:
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
            "loaded_models": self._loaded_models,
            "active_requests": self._active_requests,
        }

    @staticmethod
    def from_dict(data: dict) -> MeshNode:
        caps_data = data.get("capabilities", {})
        caps = NodeCapabilities(**caps_data)
        return MeshNode(
            node_id=data["node_id"],
            hostname=data.get("hostname", ""),
            ip=data.get("ip", ""),
            port=data.get("port", 8000),
            state=MeshNodeState[data.get("state", "INITIALIZING")],
            rank=data.get("rank", -1),
            capabilities=caps,
            _loaded_models=data.get("loaded_models", []),
            _active_requests=data.get("active_requests", 0),
        )
