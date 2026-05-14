"""Tests for topology.py — mesh topology management."""

import pytest

from yunshu_mesh.topology import MeshTopology, TopologyType
from yunshu_mesh.node import MeshNode, MeshNodeState


class TestTopologyType:
    def test_types_defined(self):
        assert TopologyType.RING
        assert TopologyType.FULLY_CONNECTED


class TestMeshTopology:
    def _make_node(self, rank):
        return MeshNode(
            node_id=f"n{rank}",
            ip="127.0.0.1",
            port=8000 + rank,
            rank=rank,
            state=MeshNodeState.READY,
        )

    def test_create_ring(self):
        topo = MeshTopology(TopologyType.RING)
        assert topo is not None

    def test_add_node(self):
        topo = MeshTopology(TopologyType.RING)
        topo.add_node(self._make_node(0))
        assert topo.size == 1

    def test_remove_node(self):
        topo = MeshTopology(TopologyType.RING)
        topo.add_node(self._make_node(0))
        removed = topo.remove_node("n0")
        assert removed
        assert topo.size == 0

    def test_get_node_by_rank(self):
        topo = MeshTopology(TopologyType.RING)
        topo.add_node(self._make_node(2))
        found = topo.get_node(rank=2)
        assert found is not None
        assert found.node_id == "n2"

    def test_get_rank(self):
        topo = MeshTopology(TopologyType.RING)
        topo.add_node(self._make_node(3))
        assert topo.get_rank("n3") == 3

    def test_get_neighbors_ring(self):
        topo = MeshTopology(TopologyType.RING)
        for i in range(4):
            topo.add_node(self._make_node(i))
        neighbors = topo.get_neighbors(rank=1)
        assert len(neighbors) > 0

    def test_to_dict(self):
        topo = MeshTopology(TopologyType.RING)
        topo.add_node(self._make_node(0))
        d = topo.to_dict()
        assert "topology_type" in d
