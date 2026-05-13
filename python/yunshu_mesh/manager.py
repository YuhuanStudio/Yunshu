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

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import mlx.core as mx

from .node import MeshNode, MeshNodeState, NodeCapabilities
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
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

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
        self.stop_discovery()
        self._collective.shutdown()
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
        return {
            "distributed": self.is_distributed,
            "rank": self.rank,
            "world_size": self.world_size,
            "topology": self._topology.topo_type.value,
            "nodes": len(self._topology.nodes),
            "local_node": self._local_node.to_dict() if self._local_node else None,
            "pipeline": self._pipeline.to_dict() if self._pipeline else None,
        }

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

        return {
            "distributed": self.is_distributed,
            "topology": self._topology.topo_type.value,
            "nodes": nodes,
            "healthy_count": sum(1 for h in health.values() if h),
            "total_count": len(self._topology.nodes),
        }

    def handle_node_failure(self, node_id: str) -> None:
        """Handle a node failure — update topology, log event."""
        failed_node = self._topology.get_node(self._topology.get_rank(node_id))
        if failed_node:
            failed_node.state = MeshNodeState.OFFLINE
            logger.warning(f"Node failure: {failed_node.hostname} ({node_id})")
            # Re-evaluate topology
            if self._topology.size > 1:
                new_type = self._topology.auto_select()
                if new_type != self._topology.topo_type:
                    self._topology.topo_type = new_type
                    logger.info(f"Topology changed to {new_type.value} after node failure")

    def _on_peer_discovered(self, node: MeshNode) -> None:
        """Callback: new peer discovered."""
        rank = self._topology.add_node(node)
        logger.info(f"Peer discovered: {node.hostname} rank={rank}")

    def _on_peer_lost(self, node: MeshNode) -> None:
        """Callback: peer disappeared."""
        self._topology.remove_node(node.node_id)
        logger.info(f"Peer lost: {node.hostname}")

    def _on_node_timeout(self, node: MeshNode) -> None:
        """Callback: heartbeat timeout."""
        self.handle_node_failure(node.node_id)

    def _on_node_recovered(self, node: MeshNode) -> None:
        """Callback: node recovered after timeout."""
        node.state = MeshNodeState.READY
        logger.info(f"Node recovered: {node.hostname}")
