"""Tests for Yunshu Mesh — distributed inference layer."""

import pytest

from yunshu_mesh.node import MeshNode, MeshNodeState, NodeCapabilities
from yunshu_mesh.topology import MeshTopology, TopologyType
from yunshu_mesh.pipeline import PipelineParallel


class TestMeshNode:
    """Test mesh node creation and health."""

    def test_local_node(self):
        node = MeshNode.local(port=8000)
        assert node.node_id
        assert node.hostname
        assert node.ip
        assert node.state == MeshNodeState.READY
        assert node.rank == -1

    def test_capabilities_detect(self):
        caps = NodeCapabilities.detect()
        assert caps.total_memory_gb > 0
        assert caps.cpu_cores > 0

    def test_heartbeat(self):
        node = MeshNode(node_id="test", hostname="test-host")
        assert node.is_healthy()
        node.last_heartbeat = 0  # Long time ago
        assert not node.is_healthy(timeout=1.0)
        node.heartbeat()
        assert node.is_healthy()

    def test_serialization(self):
        node = MeshNode(node_id="test", hostname="test-host", ip="10.0.0.1", rank=0)
        d = node.to_dict()
        restored = MeshNode.from_dict(d)
        assert restored.node_id == "test"
        assert restored.hostname == "test-host"
        assert restored.ip == "10.0.0.1"
        assert restored.rank == 0


class TestMeshTopology:
    """Test topology management."""

    def test_add_nodes(self):
        topo = MeshTopology()
        n1 = MeshNode(node_id="n1", capabilities=NodeCapabilities(supports_jaccl=True))
        n2 = MeshNode(node_id="n2", capabilities=NodeCapabilities(supports_jaccl=True))
        assert topo.add_node(n1) == 0
        assert topo.add_node(n2) == 1
        assert topo.size == 2

    def test_ring_neighbors(self):
        topo = MeshTopology(TopologyType.RING)
        for i in range(4):
            topo.add_node(MeshNode(node_id=f"n{i}"))
        assert topo.get_neighbors(0) == [3, 1]
        assert topo.get_neighbors(2) == [1, 3]

    def test_fully_connected_neighbors(self):
        topo = MeshTopology(TopologyType.FULLY_CONNECTED)
        for i in range(3):
            topo.add_node(MeshNode(node_id=f"n{i}"))
        assert sorted(topo.get_neighbors(0)) == [1, 2]
        assert sorted(topo.get_neighbors(1)) == [0, 2]

    def test_pipeline_neighbors(self):
        topo = MeshTopology(TopologyType.PIPELINE)
        for i in range(4):
            topo.add_node(MeshNode(node_id=f"n{i}"))
        assert topo.get_neighbors(0) == [1]
        assert topo.get_neighbors(1) == [0, 2]
        assert topo.get_neighbors(3) == [2]

    def test_auto_select_single(self):
        topo = MeshTopology()
        topo.add_node(MeshNode(node_id="n1"))
        assert topo.auto_select() == TopologyType.SINGLE

    def test_auto_select_jaccl(self):
        topo = MeshTopology()
        topo.add_node(MeshNode(node_id="n1", capabilities=NodeCapabilities(supports_jaccl=True)))
        topo.add_node(MeshNode(node_id="n2", capabilities=NodeCapabilities(supports_jaccl=True)))
        assert topo.auto_select() == TopologyType.FULLY_CONNECTED

    def test_auto_select_no_jaccl(self):
        topo = MeshTopology()
        topo.add_node(MeshNode(node_id="n1", capabilities=NodeCapabilities(supports_jaccl=False)))
        topo.add_node(MeshNode(node_id="n2", capabilities=NodeCapabilities(supports_jaccl=False)))
        assert topo.auto_select() == TopologyType.RING

    def test_remove_node(self):
        topo = MeshTopology()
        topo.add_node(MeshNode(node_id="n1"))
        topo.add_node(MeshNode(node_id="n2"))
        topo.add_node(MeshNode(node_id="n3"))
        assert topo.remove_node("n2")
        assert topo.size == 2
        assert topo.get_rank("n1") == 0
        assert topo.get_rank("n3") == 2  # rank gap preserved, not re-ranked


class TestPipelineParallel:
    """Test pipeline parallelism stage splitting."""

    def test_even_split(self):
        pp = PipelineParallel(num_layers=32, num_stages=4)
        assert len(pp.stages) == 4
        layers = [s.num_layers for s in pp.stages]
        assert sum(layers) == 32
        assert all(l == 8 for l in layers)

    def test_uneven_split(self):
        pp = PipelineParallel(num_layers=33, num_stages=4)
        layers = [s.num_layers for s in pp.stages]
        assert sum(layers) == 33
        # First stages get the extra layer
        assert layers[0] == 9

    def test_single_stage(self):
        pp = PipelineParallel(num_layers=32, num_stages=1)
        assert len(pp.stages) == 1
        assert pp.stages[0].num_layers == 32
        assert pp.stages[0].start_layer == 0
        assert pp.stages[0].end_layer == 32

    def test_get_stage(self):
        pp = PipelineParallel(num_layers=16, num_stages=4)
        s = pp.get_stage(2)
        assert s is not None
        assert s.rank == 2

    def test_serialization(self):
        pp = PipelineParallel(num_layers=16, num_stages=2)
        d = pp.to_dict()
        assert d["num_layers"] == 16
        assert d["num_stages"] == 2
        assert len(d["stages"]) == 2
