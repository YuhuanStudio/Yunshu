"""Yunshu Mesh — Mesh manager orchestrates the entire distributed layer.

Coordinates node discovery, topology management, collective operations,
and pipeline parallelism for the compute mesh.

Lifecycle:
  1. MeshManager.discover() — find nodes on the network
  2. MeshManager.initialize() — setup mx.distributed + topology
  3. MeshManager.start() — begin heartbeat + monitoring
  4. Use collective ops / pipeline for distributed inference
  5. MeshManager.shutdown() — clean exit
"""


import asyncio
import logging
import os
import threading
from typing import Any, Optional

import mlx.core as mx

from .node import MeshNode, MeshNodeState
from .topology import MeshTopology, TopologyType
from .collective import CollectiveOps
from .pipeline import PipelineParallel, auto_partition_model

logger = logging.getLogger(__name__)


class MeshManager:
    """Central manager for the Yunshu compute mesh.

    Handles node discovery, topology setup, and distributed coordination.
    This is the L3 entry point that the engine (L4) uses for
    distributed operations.
    """

    def __init__(self, backend: str = "any"):
        self._local_node: Optional[MeshNode] = None
        self._topology = MeshTopology()
        self._collective = CollectiveOps(backend=backend)
        self._pipeline: Optional[PipelineParallel] = None
        self._dp_router: Optional[Any] = None
        self._disagg_router: Optional[Any] = None
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        # C22: Event sourcing for crash recovery + audit
        self._event_log: Optional[Any] = None
        # Wave 43: RTT-aware routing (Parallax pattern)
        from .rtt_routing import RTTAwareRouter
        self._rtt_router = RTTAwareRouter.from_env()
        # Lock for thread-safe node state mutations (discovery, heartbeat, timeout)
        self._node_lock = threading.Lock()

    @property
    def is_distributed(self) -> bool:
        return self._collective.is_initialized and self._collective.size > 1

    @property
    def rank(self) -> int:
        return self._collective.rank

    @property
    def world_size(self) -> int:
        return self._collective.size

    @property
    def topology(self) -> MeshTopology:
        return self._topology

    @property
    def collective(self) -> CollectiveOps:
        return self._collective

    @property
    def group(self) -> Optional[mx.distributed.Group]:
        """Get the mx.distributed Group for engine sharding."""
        return self._collective.group

    @property
    def pipeline(self) -> Optional[PipelineParallel]:
        return self._pipeline

    @property
    def dp_router(self):
        """Get the DataParallelRouter for request routing."""
        return self._dp_router

    def initialize(
        self,
        local_port: int = 8000,
        backend: Optional[str] = None,
        topology_type: Optional[TopologyType] = None,
    ) -> bool:
        """Initialize the mesh manager.

        Args:
            local_port: Port for this node's server.
            backend: mx.distributed backend ('any', 'jaccl', 'ring', 'mpi').
            topology_type: Force a specific topology (None = auto-select).
        """
        # 1. Create local node
        self._local_node = MeshNode.local(port=local_port)
        logger.info(f"Local node: {self._local_node.hostname} ({self._local_node.ip})")
        logger.info(
            f"Capabilities: {self._local_node.capabilities.chip}, "
            f"{self._local_node.capabilities.total_memory_gb:.0f} GB UMA, "
            f"{self._local_node.capabilities.gpu_cores} GPU cores"
        )

        # 1b. C22: Initialize event log for crash recovery
        self._init_event_log()

        # 2. Try to initialize mx.distributed
        initialized = self._collective.initialize(backend=backend)

        if not initialized:
            logger.info("Running in single-node mode (no distributed backend)")
            self._topology = MeshTopology(TopologyType.SINGLE)
            self._topology.add_node(self._local_node)
            return True

        # 3. Auto-detect or use specified topology
        self._topology.add_node(self._local_node)

        if topology_type is None:
            topology_type = self._topology.auto_select()

        self._topology.topo_type = topology_type
        logger.info(f"Topology: {topology_type.value}, size={self._topology.size}")

        # 4. Setup data-parallel router for multi-node request distribution
        if self._topology.size > 1:
            self._setup_data_parallel()

        return True

    async def start(self) -> None:
        """Start background tasks (heartbeat, monitoring, discovery)."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Start node discovery if enabled
        if os.environ.get("YUNSHU_MESH_DISCOVERY", "").lower() in ("1", "true", "yes"):
            self.start_discovery()
        logger.info("Mesh manager started")

    async def shutdown(self) -> None:
        """Gracefully shutdown the mesh."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        self.stop_discovery()
        self._collective.shutdown()
        # C22: Take final snapshot before shutdown
        if self._event_log:
            state = {n.node_id: n.to_dict() for n in self._topology.nodes}
            self._event_log.take_snapshot(state)
            self._event_log.close()
            self._event_log = None
        logger.info("Mesh manager shut down")

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to track node health."""
        while self._running:
            if self._local_node:
                self._local_node.heartbeat()
            await asyncio.sleep(5.0)

    def setup_pipeline(
        self,
        num_layers: int,
        model_memory_per_layer_gb: float = 1.0,
    ) -> PipelineParallel:
        """Setup pipeline parallelism for a model.

        Args:
            num_layers: Total transformer layers.
            model_memory_per_layer_gb: Approximate memory per layer.
        """
        if self._topology.size <= 1:
            logger.info("Single node — no pipeline parallelism needed")
            self._pipeline = PipelineParallel(num_layers, 1)
            return self._pipeline

        node_mems = [
            n.capabilities.total_memory_gb * 0.7  # Reserve 30% for activations
            for n in self._topology.nodes
            if n.state == MeshNodeState.READY
        ]

        self._pipeline = auto_partition_model(
            num_layers=num_layers,
            num_nodes=self._topology.size,
            node_memory_gb=node_mems,
            model_memory_per_layer_gb=model_memory_per_layer_gb,
        )

        logger.info(
            f"Pipeline: {num_layers} layers across {self._topology.size} nodes: "
            f"{[s.num_layers for s in self._pipeline.stages]}"
        )
        return self._pipeline

    def get_stats(self) -> dict:
        """Return mesh status information."""
        stats = {
            "distributed": self.is_distributed,
            "rank": self.rank,
            "world_size": self.world_size,
            "topology": self._topology.topo_type.value,
            "nodes": len(self._topology.nodes),
            "local_node": self._local_node.to_dict() if self._local_node else None,
            "pipeline": self._pipeline.to_dict() if self._pipeline else None,
            "data_parallel": self._dp_router.get_stats() if self._dp_router else None,
            "disagg_pd": self._disagg_router.get_stats() if self._disagg_router else None,
        }
        if self._event_log:
            stats["event_log"] = self._event_log.get_stats()
        # Wave 43: RTT-aware routing stats
        stats["rtt_routing"] = self._rtt_router.get_stats()
        return stats

    def _setup_data_parallel(self) -> None:
        """Setup data-parallel router for multi-node deployments."""
        from .data_parallel import DataParallelRouter
        strategy = os.environ.get("YUNSHU_DP_STRATEGY", "least_loaded")
        self._dp_router = DataParallelRouter(strategy=strategy)
        for n in self._topology.nodes:
            self._dp_router.add_node(n.node_id, n.rank)
            # Register node in RTT router for latency-aware routing
            self._rtt_router.add_node(n.node_id, max_requests=100)
            # Set node capacity for C17 memory-proportional routing
            if hasattr(n, 'capabilities') and n.capabilities:
                self._dp_router.set_node_capacity(
                    n.node_id,
                    int(n.capabilities.total_memory_gb * 1024**3),
                    gpu_cores=n.capabilities.gpu_cores,
                )
        logger.info(f"Data-parallel router initialized: {strategy}, {len(self._topology.nodes)} nodes")

        # C20: Setup disaggregated prefill/decode router if enabled
        if os.environ.get("YUNSHU_DISAGG_PD", "0") == "1":
            self._setup_disagg_router()

    def _setup_disagg_router(self) -> None:
        """Setup disaggregated prefill/decode router (C20)."""
        from .disagg_pd import DisaggRouter, DisaggConfig
        self._disagg_router = DisaggRouter(DisaggConfig(enabled=True))
        for n in self._topology.nodes:
            caps = n.capabilities if hasattr(n, 'capabilities') else None
            mem_gb = caps.total_memory_gb if caps else 0.0
            gpu_cores = caps.gpu_cores if caps else 0
            self._disagg_router.add_node(n.node_id, memory_gb=mem_gb, gpu_cores=gpu_cores)
        logger.info(f"Disaggregated P/D router initialized: {len(self._topology.nodes)} nodes")

    # ── Discovery & Heartbeat Integration ──

    def start_discovery(self) -> None:
        """Start node discovery and heartbeat monitoring."""
        if self._local_node is None:
            logger.warning("Cannot start discovery: manager not initialized")
            return

        from .discovery import NodeDiscovery
        from .heartbeat import HeartbeatMonitor

        self._discovery = NodeDiscovery(port=self._local_node.port)
        self._heartbeat_mon = HeartbeatMonitor()

        self._discovery.on_node_discovered(self._on_peer_discovered)
        self._discovery.on_node_lost(self._on_peer_lost)
        self._heartbeat_mon.on_node_timeout(self._on_node_timeout)
        self._heartbeat_mon.on_node_recovered(self._on_node_recovered)

        self._discovery.start(self._local_node)
        peers = [n for n in self._topology.nodes if n.node_id != self._local_node.node_id]
        self._heartbeat_mon.start(self._local_node, peers)
        logger.info("Discovery and heartbeat monitoring started")

    def stop_discovery(self) -> None:
        """Stop discovery and heartbeat."""
        disc = getattr(self, '_discovery', None)
        hb = getattr(self, '_heartbeat_mon', None)
        if disc:
            disc.stop()
        if hb:
            hb.stop()
        logger.info("Discovery and heartbeat stopped")

    def get_cluster_status(self) -> dict:
        """Return full cluster status including node health."""
        hb = getattr(self, '_heartbeat_mon', None)
        health = hb.check_health() if hb else {}

        nodes = []
        for n in self._topology.nodes:
            info = n.to_dict()
            info["healthy"] = health.get(n.node_id, True)
            nodes.append(info)

        # When no heartbeat monitor is running, all nodes are assumed
        # healthy, so count from the node list rather than the empty dict.
        # The health dict only tracks peers, not the local node, so always
        # count the local node as healthy when present in the node list.
        local_is_in_nodes = any(
            n.node_id == (self._local_node.node_id if self._local_node else "")
            for n in self._topology.nodes
        )
        if health:
            healthy_count = sum(1 for h in health.values() if h)
            if local_is_in_nodes:
                healthy_count += 1
        else:
            healthy_count = len(nodes)

        return {
            "distributed": self.is_distributed,
            "topology": self._topology.topo_type.value,
            "nodes": nodes,
            "healthy_count": healthy_count,
            "total_count": len(self._topology.nodes),
        }

    def handle_node_failure(self, node_id: str) -> None:
        """Handle a node failure — update topology, log event."""
        # Avoid looking up via re-ranked index; search by node_id directly
        failed_node = None
        for n in self._topology.nodes:
            if n.node_id == node_id:
                failed_node = n
                break
        if failed_node:
            failed_node.mark_unhealthy(reason="node_failure")
            self._publish_event("node_state_change", node_id, {
                "new_state": "offline",
                "reason": "failure",
            })
            logger.warning(f"Node failure: {failed_node.hostname} ({node_id})")
            # Re-evaluate topology
            if self._topology.size > 1:
                new_type = self._topology.auto_select()
                if new_type != self._topology.topo_type:
                    self._topology.topo_type = new_type
                    logger.info(f"Topology changed to {new_type.value} after node failure")

    def _on_peer_discovered(self, node: MeshNode) -> None:
        """Callback: new peer discovered."""
        _event = None
        with self._node_lock:
            rank = self._topology.add_node(node)
            if self._dp_router:
                self._dp_router.add_node(node.node_id, rank)
                if hasattr(node, 'capabilities') and node.capabilities:
                    self._dp_router.set_node_capacity(
                        node.node_id,
                        int(node.capabilities.total_memory_gb * 1024**3),
                        gpu_cores=node.capabilities.gpu_cores,
                    )
            if self._disagg_router:
                caps = node.capabilities if hasattr(node, 'capabilities') else None
                self._disagg_router.add_node(
                    node.node_id,
                    memory_gb=caps.total_memory_gb if caps else 0.0,
                    gpu_cores=caps.gpu_cores if caps else 0,
                )
            self._rtt_router.add_node(node.node_id, max_requests=100)
            _event = ("node_join", node.node_id, {
                "hostname": node.hostname,
                "rank": rank,
                "capabilities": node.capabilities.__dict__ if hasattr(node.capabilities, '__dict__') else {},
            })
            logger.info(f"Peer discovered: {node.hostname} rank={rank}")
        if _event:
            self._publish_event(_event[0], _event[1], _event[2])

    def _on_peer_lost(self, node: MeshNode) -> None:
        """Callback: peer disappeared."""
        _event = None
        with self._node_lock:
            if node.state == MeshNodeState.OFFLINE:
                return
            node.mark_unhealthy(reason="peer_lost")
            topo_node = self._topology.get_node(node.rank)
            if topo_node is not None and topo_node.node_id == node.node_id:
                topo_node.mark_unhealthy(reason="peer_lost")
            if self._dp_router:
                self._dp_router.mark_unavailable(node.node_id)
            if self._disagg_router:
                self._disagg_router.remove_node(node.node_id)
            self._rtt_router.mark_unhealthy(node.node_id)
            _event = ("node_leave", node.node_id, {"hostname": node.hostname})
            logger.info(f"Peer lost: {node.hostname}")
        if _event:
            self._publish_event(_event[0], _event[1], _event[2])

    def _on_node_timeout(self, node: MeshNode) -> None:
        """Callback: heartbeat timeout."""
        _failure_id = None
        with self._node_lock:
            # Guard: skip if already handled by _on_peer_lost or a prior timeout
            if node.state == MeshNodeState.OFFLINE:
                return
            node.mark_unhealthy(reason="node_timeout")
            if self._dp_router:
                self._dp_router.mark_unavailable(node.node_id)
            if self._disagg_router:
                self._disagg_router.remove_node(node.node_id)
            self._rtt_router.mark_unhealthy(node.node_id)
            _failure_id = node.node_id
        if _failure_id:
            self.handle_node_failure(_failure_id)

    def _on_node_recovered(self, node: MeshNode) -> None:
        """Callback: node recovered after timeout."""
        _event = None
        with self._node_lock:
            # Guard: skip if node is already READY (duplicate recovery callback)
            if node.state == MeshNodeState.READY:
                return
            # Use mark_healthy to go through RECOVERING → READY path
            node.mark_healthy()
            topo_node = self._topology.get_node(node.rank)
            if topo_node is not None and topo_node.node_id == node.node_id:
                topo_node.mark_healthy()
            if self._dp_router:
                self._dp_router.mark_available(node.node_id)
            if self._disagg_router:
                caps = getattr(node, 'capabilities', None)
                self._disagg_router.add_node(
                    node.node_id,
                    memory_gb=caps.total_memory_gb if caps and hasattr(caps, 'total_memory_gb') else 0.0,
                    gpu_cores=caps.gpu_cores if caps and hasattr(caps, 'gpu_cores') else 0,
                )
                self._disagg_router.mark_available(node.node_id)
            self._rtt_router.mark_healthy(node.node_id)
            _event = ("node_state_change", node.node_id, {
                "new_state": "ready",
                "reason": "heartbeat_recovered",
            })
            if self._topology.size > 1:
                new_type = self._topology.auto_select()
                if new_type != self._topology.topo_type:
                    self._topology.topo_type = new_type
                    logger.info(f"Topology changed to {new_type.value} after node recovery")
            logger.info(f"Node recovered: {node.hostname}")
        if _event:
            self._publish_event(_event[0], _event[1], _event[2])

    # ── C22: Event Sourcing Helpers ──

    def _init_event_log(self) -> None:
        """Initialize event log for cluster state persistence (C22)."""
        db_dir = os.environ.get("YUNSHU_EVENT_LOG_DIR", "")
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "cluster_events.db")
        else:
            db_path = ":memory:"

        try:
            from .event_sourcing import EventLog
            self._event_log = EventLog(db_path=db_path)
            self._event_log.initialize()

            if db_path != ":memory:" and self._event_log.stats.total_events > 0:
                states = self._event_log.recover_state()
                logger.info(
                    f"Event log recovered: {len(states)} nodes from "
                    f"{self._event_log.stats.total_events} events"
                )

            # Record local node join
            if self._local_node:
                self._event_log.append("node_join", node_id=self._local_node.node_id, payload={
                    "hostname": self._local_node.hostname,
                    "rank": 0,
                })
        except Exception as e:
            logger.warning(f"Event log init failed ({e}), running without persistence")
            self._event_log = None

    def _publish_event(self, event_type: str, node_id: str, payload: dict | None = None) -> None:
        """Publish an event to the event log (C22)."""
        if self._event_log is not None:
            try:
                self._event_log.append(event_type, node_id=node_id, payload=payload)
            except Exception as e:
                logger.debug(f"Event log append failed: {e}")

