"""Tests for node.py — mesh node representation."""

import pytest

from yunshu_mesh.node import MeshNode, MeshNodeState, NodeCapabilities


class TestMeshNodeState:
    def test_states_defined(self):
        assert MeshNodeState.INITIALIZING
        assert MeshNodeState.READY
        assert MeshNodeState.BUSY
        assert MeshNodeState.DRAINING
        assert MeshNodeState.OFFLINE


class TestMeshNode:
    def test_create_node(self):
        node = MeshNode(node_id="n1", ip="127.0.0.1", port=8000)
        assert node.node_id == "n1"
        assert node.ip == "127.0.0.1"
        assert node.port == 8000

    def test_default_state(self):
        node = MeshNode(node_id="n1")
        assert node.state == MeshNodeState.INITIALIZING

    def test_set_state(self):
        node = MeshNode(node_id="n1")
        node.state = MeshNodeState.READY
        assert node.state == MeshNodeState.READY

    def test_rank_default(self):
        node = MeshNode(node_id="n1")
        assert node.rank == -1

    def test_set_rank(self):
        node = MeshNode(node_id="n1", rank=2)
        assert node.rank == 2

    def test_heartbeat(self):
        node = MeshNode(node_id="n1")
        node.heartbeat()
        assert node.last_heartbeat > 0

    def test_is_healthy(self):
        node = MeshNode(node_id="n1")
        node.heartbeat()
        assert node.is_healthy(timeout=30.0)

    def test_to_dict(self):
        node = MeshNode(node_id="n1", ip="127.0.0.1", port=8000, rank=1)
        d = node.to_dict()
        assert d["node_id"] == "n1"
        assert d["ip"] == "127.0.0.1"
        assert d["rank"] == 1

    def test_from_dict(self):
        data = {
            "node_id": "n1",
            "ip": "127.0.0.1",
            "port": 8000,
            "rank": 1,
            "state": "READY",
        }
        node = MeshNode.from_dict(data)
        assert node.node_id == "n1"
        assert node.rank == 1

    def test_local_factory(self):
        node = MeshNode.local(port=9999)
        assert node.port == 9999


class TestNodeCapabilities:
    def test_default_values(self):
        caps = NodeCapabilities()
        assert caps.total_memory_gb == 0.0
        assert caps.gpu_cores == 0
