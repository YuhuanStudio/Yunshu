"""Tests for C22: Event sourcing cluster state."""
import json
import pytest
import tempfile
import os

from yunshu_mesh.event_sourcing import (
    ClusterEvent,
    EventLog,
    EventLogStats,
    EventType,
    NodeState,
)


class TestEventType:
    def test_all_types(self):
        assert EventType.NODE_JOIN.value == "node_join"
        assert EventType.NODE_LEAVE.value == "node_leave"
        assert EventType.NODE_STATE_CHANGE.value == "node_state_change"
        assert EventType.MODEL_LOAD.value == "model_load"
        assert EventType.MODEL_UNLOAD.value == "model_unload"
        assert EventType.HEALTH_CHECK.value == "health_check"
        assert EventType.CAPABILITY_UPDATE.value == "capability_update"
        assert EventType.SNAPSHOT.value == "snapshot"


class TestClusterEvent:
    def test_auto_fields(self):
        e = ClusterEvent(event_type="test", node_id="n1")
        assert e.event_id  # auto-generated
        assert e.timestamp > 0
        assert e.sequence == 0

    def test_explicit_fields(self):
        e = ClusterEvent(
            event_id="abc123",
            event_type="test",
            timestamp=1000.0,
            node_id="n1",
            payload={"key": "val"},
            sequence=42,
        )
        assert e.event_id == "abc123"
        assert e.sequence == 42

    def test_to_dict(self):
        e = ClusterEvent(event_type="test", node_id="n1", payload={"x": 1})
        d = e.to_dict()
        assert d["event_type"] == "test"
        assert d["node_id"] == "n1"
        assert d["payload"] == {"x": 1}

    def test_from_dict(self):
        d = {
            "event_id": "abc",
            "event_type": "test",
            "timestamp": 1000.0,
            "node_id": "n1",
            "payload": {"x": 1},
            "sequence": 42,
        }
        e = ClusterEvent.from_dict(d)
        assert e.event_id == "abc"
        assert e.sequence == 42
        assert e.payload == {"x": 1}

    def test_roundtrip(self):
        e = ClusterEvent(event_type="test", node_id="n1", payload={"a": "b"})
        e2 = ClusterEvent.from_dict(e.to_dict())
        assert e2.event_type == e.event_type
        assert e2.node_id == e.node_id


class TestNodeState:
    def test_defaults(self):
        ns = NodeState(node_id="n1")
        assert ns.state == "offline"
        assert ns.models == []
        assert ns.capabilities == {}

    def test_to_dict(self):
        ns = NodeState(node_id="n1", state="ready", models=["llama"])
        d = ns.to_dict()
        assert d["node_id"] == "n1"
        assert d["state"] == "ready"
        assert d["models"] == ["llama"]


class TestEventLog:
    def test_memory_log(self):
        log = EventLog()
        log.initialize()
        event = log.append(EventType.NODE_JOIN.value, node_id="n1")
        assert event.sequence == 1
        assert event.event_type == "node_join"
        log.close()

    def test_sequence_increments(self):
        log = EventLog()
        log.initialize()
        e1 = log.append("test", node_id="n1")
        e2 = log.append("test", node_id="n2")
        assert e2.sequence > e1.sequence
        log.close()

    def test_replay(self):
        log = EventLog()
        log.initialize()
        log.append("test1", node_id="n1")
        log.append("test2", node_id="n2")
        events = log.replay()
        assert len(events) == 2
        assert events[0].event_type == "test1"
        assert events[1].event_type == "test2"
        log.close()

    def test_replay_from_sequence(self):
        log = EventLog()
        log.initialize()
        log.append("test1", node_id="n1")
        e2 = log.append("test2", node_id="n2")
        events = log.replay(from_sequence=e2.sequence)
        assert len(events) == 1
        assert events[0].event_type == "test2"
        log.close()

    def test_node_join_applied(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        ns = log.get_node_state("n1")
        assert ns is not None
        assert ns.state == "ready"
        assert ns.join_time > 0
        log.close()

    def test_node_leave_applied(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(EventType.NODE_LEAVE.value, node_id="n1")
        ns = log.get_node_state("n1")
        assert ns.state == "offline"
        assert ns.leave_time > 0
        log.close()

    def test_state_change_applied(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(
            EventType.NODE_STATE_CHANGE.value,
            node_id="n1",
            payload={"new_state": "busy"},
        )
        ns = log.get_node_state("n1")
        assert ns.state == "busy"
        log.close()

    def test_model_load_unload(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(
            EventType.MODEL_LOAD.value,
            node_id="n1",
            payload={"model": "qwen-7b"},
        )
        ns = log.get_node_state("n1")
        assert "qwen-7b" in ns.models

        log.append(
            EventType.MODEL_UNLOAD.value,
            node_id="n1",
            payload={"model": "qwen-7b"},
        )
        ns = log.get_node_state("n1")
        assert "qwen-7b" not in ns.models
        log.close()

    def test_health_check(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(
            EventType.HEALTH_CHECK.value,
            node_id="n1",
            payload={"status": "healthy"},
        )
        ns = log.get_node_state("n1")
        assert ns.last_health_status == "healthy"
        assert ns.last_health_check > 0
        log.close()

    def test_capability_update(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(
            EventType.CAPABILITY_UPDATE.value,
            node_id="n1",
            payload={"capabilities": {"gpu_cores": 24, "memory_gb": 64}},
        )
        ns = log.get_node_state("n1")
        assert ns.capabilities["gpu_cores"] == 24
        log.close()

    def test_get_all_states(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(EventType.NODE_JOIN.value, node_id="n2")
        states = log.get_all_states()
        assert len(states) == 2
        assert "n1" in states
        assert "n2" in states
        log.close()

    def test_query_events_by_type(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(EventType.MODEL_LOAD.value, node_id="n1", payload={"model": "m1"})
        events = log.query_events(event_type=EventType.NODE_JOIN.value)
        assert len(events) == 1
        assert events[0].event_type == "node_join"
        log.close()

    def test_query_events_by_node(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(EventType.NODE_JOIN.value, node_id="n2")
        events = log.query_events(node_id="n1")
        assert len(events) == 1
        assert events[0].node_id == "n1"
        log.close()

    def test_query_events_since(self):
        log = EventLog()
        log.initialize()
        log.append("test", node_id="n1")
        import time
        now = time.time()
        log.append("test2", node_id="n2")
        events = log.query_events(since=now)
        assert len(events) >= 1
        log.close()

    def test_snapshot_and_recovery(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(
            EventType.MODEL_LOAD.value,
            node_id="n1",
            payload={"model": "qwen-7b"},
        )

        # Take snapshot
        snapshot_state = {nid: ns.to_dict() for nid, ns in log.get_all_states().items()}
        log.take_snapshot(snapshot_state)

        # Add more events after snapshot
        log.append(
            EventType.NODE_JOIN.value,
            node_id="n2",
            payload={"model": "llama-8b"},
        )

        # Clear in-memory state and recover
        log._node_states.clear()
        states = log.recover_state()

        assert "n1" in states
        assert "n2" in states
        assert "qwen-7b" in states["n1"].models
        log.close()

    def test_get_last_snapshot(self):
        log = EventLog()
        log.initialize()
        assert log.get_last_snapshot() == 0
        log.take_snapshot({})
        assert log.get_last_snapshot() > 0
        log.close()

    def test_prune_before(self):
        log = EventLog()
        log.initialize()
        log.append("test1", node_id="n1")
        log.append("test2", node_id="n2")
        log.take_snapshot({})
        log.append("test3", node_id="n3")

        # Prune events before snapshot (but keep snapshot)
        pruned = log.prune_before(log.get_last_snapshot())
        assert pruned == 2  # test1, test2 pruned

        # Verify remaining events
        events = log.replay()
        types = [e.event_type for e in events]
        assert EventType.SNAPSHOT.value in types
        assert "test3" in types
        log.close()

    def test_stats(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        stats = log.get_stats()
        assert stats["total_events"] == 1
        assert "node_join" in stats["events_by_type"]
        assert stats["nodes_tracked"] == 1
        log.close()

    def test_persistent_storage(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Write events
            log1 = EventLog(db_path=db_path)
            log1.initialize()
            log1.append(EventType.NODE_JOIN.value, node_id="n1")
            log1.close()

            # Read back
            log2 = EventLog(db_path=db_path)
            log2.initialize()
            events = log2.replay()
            assert len(events) == 1
            assert events[0].node_id == "n1"
            log2.close()
        finally:
            os.unlink(db_path)

    def test_auto_initialize(self):
        log = EventLog()
        # append should auto-initialize
        event = log.append("test", node_id="n1")
        assert event.sequence == 1
        log.close()

    def test_model_load_duplicate_ignored(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="n1")
        log.append(EventType.MODEL_LOAD.value, node_id="n1", payload={"model": "m1"})
        log.append(EventType.MODEL_LOAD.value, node_id="n1", payload={"model": "m1"})
        ns = log.get_node_state("n1")
        assert ns.models.count("m1") == 1
        log.close()

    def test_empty_node_id_ignored(self):
        log = EventLog()
        log.initialize()
        log.append(EventType.NODE_JOIN.value, node_id="")
        assert log.get_all_states() == {}
        log.close()
