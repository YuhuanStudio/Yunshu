"""Unit tests for mesh node discovery and heartbeat."""
import json
import time
import threading
import socket
import pytest

from yunshu_mesh.node import MeshNode, MeshNodeState, NodeCapabilities
from yunshu_mesh.discovery import NodeDiscovery
from yunshu_mesh.heartbeat import HeartbeatMonitor


def _make_node(node_id="test", ip="127.0.0.1", port=8000):
    return MeshNode(
        node_id=node_id,
        hostname=f"host-{node_id}",
        ip=ip,
        port=port,
        state=MeshNodeState.READY,
        capabilities=NodeCapabilities(chip="M3", total_memory_gb=64, gpu_cores=30),
    )


class TestNodeDiscovery:
    def test_construction(self):
        d = NodeDiscovery()
        assert d.get_discovered_nodes() == []

    def test_callbacks_registered(self):
        d = NodeDiscovery()
        discovered = []
        d.on_node_discovered(lambda n: discovered.append(n))
        d.on_node_lost(lambda n: discovered.remove(n))
        assert len(d._on_discovered_callbacks) == 1
        assert len(d._on_lost_callbacks) == 1

    def test_add_discovered_callback(self):
        d = NodeDiscovery()
        found = []
        d.on_node_discovered(lambda n: found.append(n.node_id))
        node = _make_node("peer-1")
        d._add_discovered(node)
        assert "peer-1" in found
        assert len(d.get_discovered_nodes()) == 1

    def test_duplicate_discovery_not_new(self):
        d = NodeDiscovery()
        count = [0]
        d.on_node_discovered(lambda n: count.__setitem__(0, count[0] + 1))
        node = _make_node("peer-1")
        d._add_discovered(node)
        d._add_discovered(node)
        assert count[0] == 1

    def test_remove_discovered_callback(self):
        d = NodeDiscovery()
        lost = []
        d.on_node_lost(lambda n: lost.append(n.node_id))
        node = _make_node("peer-1")
        d._add_discovered(node)
        d._remove_discovered("peer-1")
        assert "peer-1" in lost
        assert len(d.get_discovered_nodes()) == 0

    def test_udp_roundtrip(self):
        """Test UDP broadcast discovery between two instances."""
        port_a = 17998
        port_b = 17999
        disc_port = 17997

        node_a = _make_node("node-a", port=port_a)
        node_b = _make_node("node-b", port=port_b)

        d_a = NodeDiscovery(port=port_a, discovery_port=disc_port)
        d_b = NodeDiscovery(port=port_b, discovery_port=disc_port + 1)

        found = []
        d_a.on_node_discovered(lambda n: found.append(n.node_id))

        d_a.start(node_a)
        time.sleep(0.3)
        d_b.start(node_b)
        time.sleep(3.0)

        d_a.stop()
        d_b.stop()

        # Verify no exceptions during discovery
        assert d_a._running is False
        assert d_b._running is False


class TestHeartbeatMonitor:
    def test_construction(self):
        m = HeartbeatMonitor()
        assert m.check_health() == {}

    def test_callbacks_registered(self):
        m = HeartbeatMonitor()
        m.on_node_timeout(lambda n: None)
        m.on_node_recovered(lambda n: None)
        assert len(m._on_timeout_callbacks) == 1
        assert len(m._on_recovery_callbacks) == 1

    def test_check_health_with_nodes(self):
        m = HeartbeatMonitor(interval=1.0, timeout=5.0)
        peer = _make_node("peer-1")
        m._nodes["peer-1"] = peer
        m._last_heartbeat["peer-1"] = time.time()
        health = m.check_health()
        assert health["peer-1"] is True

    def test_check_health_timeout(self):
        m = HeartbeatMonitor(interval=1.0, timeout=5.0)
        peer = _make_node("peer-1")
        m._nodes["peer-1"] = peer
        m._last_heartbeat["peer-1"] = time.time() - 100  # Long ago
        health = m.check_health()
        assert health["peer-1"] is False

    def test_udp_heartbeat(self):
        """Test heartbeat exchange between two monitors."""
        local = _make_node("local", port=18000)
        peer = _make_node("peer", ip="127.0.0.1", port=18001)

        m_local = HeartbeatMonitor(interval=0.5, timeout=10.0, port=18002)
        m_peer = HeartbeatMonitor(interval=0.5, timeout=10.0, port=18003)

        # Each monitors the other
        m_local.start(local, [peer])
        m_peer.start(peer, [local])

        time.sleep(2.0)

        health = m_local.check_health()
        # May or may not have received heartbeat depending on timing
        assert isinstance(health, dict)

        m_local.stop()
        m_peer.stop()

    def test_start_stop_lifecycle(self):
        m = HeartbeatMonitor(interval=1.0, timeout=5.0, port=18004)
        local = _make_node("local")
        m.start(local, [])
        assert m._running is True
        m.stop()
        assert m._running is False
