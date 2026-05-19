from __future__ import annotations
"""Yunshu Event Sourcing — C22: crash recovery + audit trail for cluster state.

Studied from exo's distributed event sourcing pattern:
- Every cluster state change (node join/leave, model load/unload, health)
  is recorded as an immutable event
- Events are appended to a persistent log (SQLite + in-memory replay)
- On crash recovery, the log is replayed to reconstruct full state
- Provides complete audit trail for debugging and compliance

Event types:
  - NODE_JOIN: Node added to cluster
  - NODE_LEAVE: Node removed from cluster
  - NODE_STATE_CHANGE: Node state transition (READY→BUSY→DRAINING→OFFLINE)
  - MODEL_LOAD: Model loaded on a node
  - MODEL_UNLOAD: Model unloaded from a node
  - HEALTH_CHECK: Periodic health check result
  - CAPABILITY_UPDATE: Node hardware capabilities changed
  - SNAPSHOT: Periodic full state snapshot for faster recovery
"""

import copy
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    NODE_JOIN = "node_join"
    NODE_LEAVE = "node_leave"
    NODE_STATE_CHANGE = "node_state_change"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    HEALTH_CHECK = "health_check"
    CAPABILITY_UPDATE = "capability_update"
    SNAPSHOT = "snapshot"


@dataclass
class ClusterEvent:
    """A single cluster state change event.

    Attributes:
        event_id: Unique event identifier (UUID).
        event_type: Type of event.
        timestamp: Unix timestamp (seconds).
        node_id: Affected node (empty for cluster-wide events).
        payload: Event-specific data (JSON-serializable).
        sequence: Monotonically increasing sequence number.
    """

    event_id: str = ""
    event_type: str = ""
    timestamp: float = 0.0
    node_id: str = ""
    payload: dict = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self):
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "payload": self.payload,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClusterEvent:
        return cls(
            event_id=d.get("event_id", ""),
            event_type=d.get("event_type", ""),
            timestamp=d.get("timestamp", 0.0),
            node_id=d.get("node_id", ""),
            payload=d.get("payload", {}),
            sequence=d.get("sequence", 0),
        )


@dataclass
class NodeState:
    """Reconstructable node state from event log.

    Built by replaying events; tracks the latest known state of a node.
    """

    node_id: str = ""
    state: str = "offline"
    models: list[str] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)
    last_health_check: float = 0.0
    last_health_status: str = "unknown"
    join_time: float = 0.0
    leave_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "models": self.models,
            "capabilities": self.capabilities,
            "last_health_check": self.last_health_check,
            "last_health_status": self.last_health_status,
            "join_time": self.join_time,
            "leave_time": self.leave_time,
        }


@dataclass
class EventLogStats:
    """Statistics for the event log."""

    total_events: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    nodes_tracked: int = 0
    snapshots_taken: int = 0
    log_size_bytes: int = 0
    last_event_time: float = 0.0


class EventLog:
    """Persistent event log for cluster state changes.

    Stores events in SQLite for durability. On recovery, replays
    events from the last SNAPSHOT to reconstruct state.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._sequence: int = 0
        self._lock = threading.Lock()
        self._stats = EventLogStats()
        # In-memory state reconstruction cache
        self._node_states: dict[str, NodeState] = {}
        self._initialized: bool = False

    def initialize(self) -> None:
        """Create or open the event log database."""
        if self._initialized:
            return

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                node_id TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
        """)
        self._conn.commit()

        # Recover sequence counter from existing events
        row = self._conn.execute(
            "SELECT MAX(sequence) FROM events"
        ).fetchone()
        if row and row[0] is not None:
            self._sequence = row[0]
            self._stats.total_events = self._sequence
            logger.info(f"Recovered event log: {self._sequence} existing events")

        self._initialized = True

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False

    def append(self, event_type: str, node_id: str = "", payload: dict | None = None) -> ClusterEvent:
        """Append a new event to the log.

        Args:
            event_type: Type of event (EventType value).
            node_id: Affected node ID.
            payload: Event-specific data.

        Returns:
            The created ClusterEvent.
        """
        if not self._initialized:
            self.initialize()

        with self._lock:
            self._sequence += 1
            event = ClusterEvent(
                event_type=event_type,
                node_id=node_id,
                payload=payload or {},
                sequence=self._sequence,
            )

            if self._conn is None:
                raise RuntimeError("Database connection not initialized")
            self._conn.execute(
                "INSERT INTO events (sequence, event_id, event_type, timestamp, node_id, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event.sequence, event.event_id, event.event_type,
                 event.timestamp, event.node_id, json.dumps(event.payload)),
            )
            self._conn.commit()

            self._stats.total_events = self._sequence
            self._stats.events_by_type[event_type] = (
                self._stats.events_by_type.get(event_type, 0) + 1
            )
            self._stats.last_event_time = event.timestamp

            self._apply_event(event)

        return event

    def replay(self, from_sequence: int = 0) -> list[ClusterEvent]:
        """Replay events from a given sequence number.

        Args:
            from_sequence: Starting sequence number (0 = beginning).

        Returns:
            List of events in order.
        """
        if not self._initialized:
            self.initialize()

        if self._conn is None:
            raise RuntimeError("EventLog not initialized: connection is None")
        rows = self._conn.execute(
            "SELECT sequence, event_id, event_type, timestamp, node_id, payload "
            "FROM events WHERE sequence >= ? ORDER BY sequence",
            (from_sequence,),
        ).fetchall()

        return [
            ClusterEvent(
                sequence=row[0],
                event_id=row[1],
                event_type=row[2],
                timestamp=row[3],
                node_id=row[4],
                payload=json.loads(row[5]),
            )
            for row in rows
        ]

    def get_last_snapshot(self) -> int:
        """Find the sequence number of the last snapshot event.

        Returns 0 if no snapshot exists.
        """
        if not self._initialized:
            self.initialize()

        if self._conn is None:
            raise RuntimeError("EventLog not initialized: connection is None")
        row = self._conn.execute(
            "SELECT MAX(sequence) FROM events WHERE event_type = ?",
            (EventType.SNAPSHOT.value,),
        ).fetchone()
        return row[0] if row and row[0] is not None else 0

    def take_snapshot(self, cluster_state: dict) -> ClusterEvent:
        """Record a full state snapshot for faster recovery.

        Args:
            cluster_state: Current cluster state (all node states, etc.).

        Returns:
            The snapshot event.
        """
        self._stats.snapshots_taken += 1
        return self.append(
            event_type=EventType.SNAPSHOT.value,
            node_id="",
            payload={"cluster_state": cluster_state},
        )

    def recover_state(self) -> dict[str, NodeState]:
        """Recover cluster state from event log.

        Strategy:
        1. Find last snapshot
        2. If snapshot exists, restore from it
        3. Replay events after snapshot
        4. Return reconstructed node states

        Returns:
            Dict of node_id → NodeState.
        """
        if not self._initialized:
            self.initialize()

        with self._lock:
            self._node_states.clear()

        # Find last snapshot
        snapshot_seq = self.get_last_snapshot()

        if snapshot_seq > 0:
            # Restore from snapshot
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")
            row = self._conn.execute(
                "SELECT payload FROM events WHERE sequence = ?",
                (snapshot_seq,),
            ).fetchone()
            if row:
                snapshot_data = json.loads(row[0])
                for nid, ns_data in snapshot_data.get("cluster_state", {}).items():
                    self._node_states[nid] = NodeState(
                        node_id=ns_data.get("node_id", nid),
                        state=ns_data.get("state", "offline"),
                        models=ns_data.get("models", []),
                        capabilities=ns_data.get("capabilities", {}),
                        last_health_check=ns_data.get("last_health_check", 0.0),
                        last_health_status=ns_data.get("last_health_status", "unknown"),
                        join_time=ns_data.get("join_time", 0.0),
                        leave_time=ns_data.get("leave_time", 0.0),
                    )

        # Replay events after snapshot
        events = self.replay(snapshot_seq + 1 if snapshot_seq > 0 else 0)
        for event in events:
            if event.event_type != EventType.SNAPSHOT.value:
                self._apply_event(event)

        self._stats.nodes_tracked = len(self._node_states)
        logger.info(
            f"Recovered cluster state: {len(self._node_states)} nodes, "
            f"from snapshot_seq={snapshot_seq}, "
            f"replayed {len(events)} events"
        )
        return dict(self._node_states)

    def _apply_event(self, event: ClusterEvent) -> None:
        """Apply an event to the in-memory node state."""
        if event.event_type == EventType.SNAPSHOT.value:
            return  # Snapshots are handled in recover_state()

        nid = event.node_id
        if not nid:
            return

        if nid not in self._node_states:
            self._node_states[nid] = NodeState(node_id=nid)

        ns = self._node_states[nid]

        if event.event_type == EventType.NODE_JOIN.value:
            ns.state = "ready"
            ns.join_time = event.timestamp

        elif event.event_type == EventType.NODE_LEAVE.value:
            ns.state = "offline"
            ns.leave_time = event.timestamp

        elif event.event_type == EventType.NODE_STATE_CHANGE.value:
            ns.state = event.payload.get("new_state", ns.state)

        elif event.event_type == EventType.MODEL_LOAD.value:
            model = event.payload.get("model", "")
            if model and model not in ns.models:
                ns.models.append(model)

        elif event.event_type == EventType.MODEL_UNLOAD.value:
            model = event.payload.get("model", "")
            if model in ns.models:
                ns.models.remove(model)

        elif event.event_type == EventType.HEALTH_CHECK.value:
            ns.last_health_check = event.timestamp
            ns.last_health_status = event.payload.get("status", "unknown")

        elif event.event_type == EventType.CAPABILITY_UPDATE.value:
            ns.capabilities = event.payload.get("capabilities", {})

    def get_node_state(self, node_id: str) -> NodeState | None:
        """Get the current state of a node."""
        with self._lock:
            return self._node_states.get(node_id)

    def get_all_states(self) -> dict[str, NodeState]:
        """Get all node states."""
        with self._lock:
            return dict(self._node_states)

    def query_events(
        self,
        event_type: str | None = None,
        node_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[ClusterEvent]:
        """Query events with optional filters.

        Args:
            event_type: Filter by event type.
            node_id: Filter by node ID.
            since: Filter by timestamp (>=).
            limit: Maximum number of events to return.

        Returns:
            List of matching events.
        """
        if not self._initialized:
            self.initialize()

        conditions = []
        params = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if node_id:
            conditions.append("node_id = ?")
            params.append(node_id)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        if self._conn is None:
            raise RuntimeError("EventLog not initialized: connection is None")
        with self._lock:
            rows = self._conn.execute(
                f"SELECT sequence, event_id, event_type, timestamp, node_id, payload "
                f"FROM events WHERE {where} ORDER BY sequence DESC LIMIT ?",
                params,
            ).fetchall()

            return [
                ClusterEvent(
                    sequence=row[0],
                    event_id=row[1],
                    event_type=row[2],
                    timestamp=row[3],
                    node_id=row[4],
                    payload=json.loads(row[5]),
                )
                for row in rows
            ]

    def prune_before(self, sequence: int) -> int:
        """Remove events before a given sequence number.

        Useful after a snapshot to free space. Protects snapshots from
        deletion and refuses to prune if no snapshot exists (which would
        make recovery impossible).

        Returns:
            Number of pruned events.
        """
        if not self._initialized:
            self.initialize()

        if self._conn is None:
            raise RuntimeError("EventLog not initialized: connection is None")

        # Guard: refuse to prune if no snapshot exists, as this would
        # destroy all events and make crash recovery impossible.
        last_snapshot = self.get_last_snapshot()
        if last_snapshot == 0:
            logger.warning("Cannot prune events: no snapshot exists yet")
            return 0

        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM events WHERE sequence < ? AND event_type != ?",
                (sequence, EventType.SNAPSHOT.value),
            )
            self._conn.commit()
            pruned = cursor.rowcount
        if pruned > 0:
            logger.info(f"Pruned {pruned} events before sequence {sequence}")
        return pruned

    @property
    def stats(self) -> EventLogStats:
        with self._lock:
            self._stats.nodes_tracked = len(self._node_states)
            if self._initialized and self._conn:
                try:
                    row = self._conn.execute(
                        "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
                    ).fetchone()
                    if row:
                        self._stats.log_size_bytes = row[0]
                except Exception:
                    logger.debug("failed to query log size", exc_info=True)
            return copy.copy(self._stats)

    def get_stats(self) -> dict[str, Any]:
        """Return event log statistics."""
        s = self.stats
        return {
            "total_events": s.total_events,
            "events_by_type": s.events_by_type,
            "nodes_tracked": s.nodes_tracked,
            "snapshots_taken": s.snapshots_taken,
            "log_size_bytes": s.log_size_bytes,
            "last_event_time": s.last_event_time,
        }
