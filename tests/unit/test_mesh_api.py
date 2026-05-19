"""Tests for Compute Mesh API and core components."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from yunshu_engine.engine import Engine, EngineConfig


class TestMeshAPI:
    """Test mesh management REST endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_mesh_status(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/mesh/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "distributed" in data
        assert "rank" in data
        assert "world_size" in data

    def test_mesh_nodes(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/mesh/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "total" in data
        assert "topology_type" in data

    def test_mesh_initialize(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.post("/api/v1/mesh/initialize", params={"backend": "any"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "initialized"

    def test_collective_test_single_node(self):
        from yunshu_gateway.main import create_app
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/mesh/collective/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "single_node"


class TestMeshTopology:
    """Test mesh topology management."""

    def test_single_node_topology(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode

        topo = MeshTopology(TopologyType.SINGLE)
        node = MeshNode(node_id="test", hostname="test-host")
        rank = topo.add_node(node)
        assert rank == 0
        assert topo.size == 1

    def test_ring_topology_neighbors(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode

        topo = MeshTopology(TopologyType.RING)
        for i in range(4):
            topo.add_node(MeshNode(node_id=f"n{i}", hostname=f"host{i}"))

        neighbors = topo.get_neighbors(0)
        assert neighbors == [3, 1]  # wrap-around + next

    def test_fully_connected_neighbors(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode

        topo = MeshTopology(TopologyType.FULLY_CONNECTED)
        for i in range(3):
            topo.add_node(MeshNode(node_id=f"n{i}", hostname=f"host{i}"))

        neighbors = topo.get_neighbors(1)
        assert set(neighbors) == {0, 2}

    def test_auto_select_single(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        topo = MeshTopology()
        assert topo.auto_select() == TopologyType.SINGLE

    def test_auto_select_ring_no_jaccl(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode, NodeCapabilities

        topo = MeshTopology()
        for i in range(3):
            n = MeshNode(node_id=f"n{i}", hostname=f"host{i}",
                         capabilities=NodeCapabilities(supports_jaccl=False))
            topo.add_node(n)
        assert topo.auto_select() == TopologyType.RING

    def test_auto_select_fully_connected_jaccl(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode, NodeCapabilities

        topo = MeshTopology()
        for i in range(3):
            n = MeshNode(node_id=f"n{i}", hostname=f"host{i}",
                         capabilities=NodeCapabilities(supports_jaccl=True))
            topo.add_node(n)
        assert topo.auto_select() == TopologyType.FULLY_CONNECTED

    def test_remove_node(self):
        from yunshu_mesh.topology import MeshTopology
        from yunshu_mesh.node import MeshNode

        topo = MeshTopology()
        topo.add_node(MeshNode(node_id="n0", hostname="h0"))
        topo.add_node(MeshNode(node_id="n1", hostname="h1"))
        topo.add_node(MeshNode(node_id="n2", hostname="h2"))

        assert topo.remove_node("n1")
        assert topo.size == 2
        assert topo.nodes[0].rank == 0
        assert topo.nodes[1].rank == 2  # rank gap preserved

    def test_pipeline_send_recv_pairs(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode

        topo = MeshTopology(TopologyType.PIPELINE)
        for i in range(4):
            topo.add_node(MeshNode(node_id=f"n{i}", hostname=f"host{i}"))

        pairs = topo.get_send_recv_pairs(0)
        assert (0, 1) in pairs
        assert (1, 2) in pairs
        assert (2, 3) in pairs

    def test_to_dict(self):
        from yunshu_mesh.topology import MeshTopology, TopologyType
        from yunshu_mesh.node import MeshNode

        topo = MeshTopology(TopologyType.RING)
        topo.add_node(MeshNode(node_id="n0", hostname="h0"))

        d = topo.to_dict()
        assert d["topology_type"] == "ring"
        assert d["size"] == 1
        assert len(d["nodes"]) == 1


class TestMeshNode:
    """Test mesh node representation."""

    def test_node_health(self):
        from yunshu_mesh.node import MeshNode, MeshNodeState

        node = MeshNode(node_id="test", state=MeshNodeState.READY)
        assert node.is_healthy()

    def test_node_offline_not_healthy(self):
        from yunshu_mesh.node import MeshNode, MeshNodeState

        node = MeshNode(node_id="test", state=MeshNodeState.OFFLINE)
        assert not node.is_healthy()

    def test_node_heartbeat(self):
        import time
        from yunshu_mesh.node import MeshNode, MeshNodeState

        node = MeshNode(
            node_id="test",
            state=MeshNodeState.READY,
            last_heartbeat=time.monotonic() - 60,  # 60s ago
        )
        assert not node.is_healthy(timeout=30)
        node.heartbeat()
        assert node.is_healthy(timeout=30)

    def test_node_serialization(self):
        from yunshu_mesh.node import MeshNode, MeshNodeState, NodeCapabilities

        node = MeshNode(
            node_id="abc123",
            hostname="mac-studio",
            ip="192.168.1.100",
            port=8000,
            state=MeshNodeState.READY,
            rank=0,
            capabilities=NodeCapabilities(
                total_memory_gb=192.0,
                gpu_cores=48,
                cpu_cores=24,
                chip="Apple M2 Ultra",
                thunderbolt_ports=6,
                supports_jaccl=True,
            ),
        )

        d = node.to_dict()
        assert d["node_id"] == "abc123"
        assert d["hostname"] == "mac-studio"
        assert d["capabilities"]["total_memory_gb"] == 192.0
        assert d["capabilities"]["supports_jaccl"] is True

        # Round-trip
        restored = MeshNode.from_dict(d)
        assert restored.node_id == "abc123"
        assert restored.capabilities.total_memory_gb == 192.0
        assert restored.state == MeshNodeState.READY
