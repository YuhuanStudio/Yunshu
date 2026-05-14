"""Tests for heartbeat.py — mesh node heartbeat monitoring."""

import pytest

from yunshu_mesh.heartbeat import HeartbeatMonitor
from yunshu_mesh.node import MeshNode, MeshNodeState


class TestHeartbeatMonitor:
    def test_create_monitor(self):
        monitor = HeartbeatMonitor(interval=1.0, timeout=5.0, port=0)
        assert monitor is not None

    def test_start_and_stop(self):
        monitor = HeartbeatMonitor(interval=0.5, timeout=2.0, port=0)
        node = MeshNode(node_id="n1", ip="127.0.0.1", port=8000, rank=0, state=MeshNodeState.READY)
        monitor.start(node, [])
        monitor.stop()

    def test_check_health_empty(self):
        monitor = HeartbeatMonitor(interval=0.5, timeout=2.0, port=0)
        node = MeshNode(node_id="n1", ip="127.0.0.1", port=8000, rank=0, state=MeshNodeState.READY)
        monitor.start(node, [])
        health = monitor.check_health()
        assert isinstance(health, dict)
        monitor.stop()

    def test_callbacks_registered(self):
        monitor = HeartbeatMonitor(interval=0.5, timeout=2.0, port=0)
        monitor.on_node_timeout(lambda nid: None)
        monitor.on_node_recovered(lambda nid: None)
