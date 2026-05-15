from __future__ import annotations
"""RTT-aware mesh request routing (Parallax pattern).

Parallax optimizes distributed inference by measuring round-trip
times between mesh nodes and routing requests to minimize latency.
Uses exponential moving average (EMA) for RTT estimation and
weighted least-loaded routing.
"""

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NodeRTT:
    """RTT estimate for a single mesh node."""
    node_id: str
    rtt_ema: float = 0.0  # Exponential moving average RTT (ms)
    rtt_var: float = 0.0  # RTT variance (for Jacobson/Karels)
    last_probe: float = 0.0
    probe_count: int = 0
    # Load tracking
    active_requests: int = 0
    max_requests: int = 32

    def update_rtt(self, measured_ms: float, alpha: float = 0.125) -> None:
        """Update RTT estimate using Jacobson/Karels algorithm."""
        if self.probe_count == 0:
            self.rtt_ema = measured_ms
            self.rtt_var = measured_ms / 2
        else:
            delta = measured_ms - self.rtt_ema
            self.rtt_ema += alpha * delta
            self.rtt_var += alpha * (abs(delta) - self.rtt_var)

        self.last_probe = time.monotonic()
        self.probe_count += 1

    @property
    def rtt_timeout_ms(self) -> float:
        """Smoothed RTT + 4 * variation (TCP-style timeout)."""
        return self.rtt_ema + 4 * self.rtt_var

    @property
    def load_fraction(self) -> float:
        if self.max_requests <= 0:
            return 0.0
        return self.active_requests / self.max_requests


@dataclass
class RoutingScore:
    """Score for a routing decision."""
    node_id: str
    rtt_score: float
    load_score: float
    combined_score: float
    selected: bool = False


class RTTAwareRouter:
    """Routes requests to mesh nodes based on RTT + load.

    Scoring formula:
      score = (1 - rtt_weight) * (1 - load_fraction) + rtt_weight * (1 / (1 + rtt_ema))

    Lower RTT and lower load → higher score → preferred node.
    """

    def __init__(
        self,
        rtt_weight: float = 0.6,
        load_weight: float = 0.4,
        probe_interval: float = 30.0,
        rtt_fallback_ms: float = 100.0,
    ) -> None:
        self._rtt_weight = rtt_weight
        self._load_weight = load_weight
        self._probe_interval = probe_interval
        self._rtt_fallback = rtt_fallback_ms
        self._nodes: dict[str, NodeRTT] = {}
        self._route_count = 0
        self._fallback_count = 0

    @classmethod
    def from_env(cls) -> RTTAwareRouter:
        return cls(
            rtt_weight=float(os.environ.get("YUNSHU_RTT_WEIGHT", "0.6")),
            load_weight=float(os.environ.get("YUNSHU_LOAD_WEIGHT", "0.4")),
            probe_interval=float(os.environ.get("YUNSHU_PROBE_INTERVAL", "30.0")),
        )

    def add_node(self, node_id: str, max_requests: int = 32) -> None:
        self._nodes[node_id] = NodeRTT(
            node_id=node_id,
            max_requests=max_requests,
        )

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    def record_rtt(self, node_id: str, rtt_ms: float) -> None:
        """Record an RTT measurement for a node."""
        node = self._nodes.get(node_id)
        if node:
            node.update_rtt(rtt_ms)

    def record_request_start(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.active_requests += 1

    def record_request_end(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.active_requests = max(0, node.active_requests - 1)

    def route(self, exclude: set[str] | None = None) -> RoutingScore | None:
        """Select the best node for a new request."""
        if not self._nodes:
            return None

        candidates = [
            (nid, node) for nid, node in self._nodes.items()
            if (exclude is None or nid not in exclude)
            and node.active_requests < node.max_requests
        ]

        if not candidates:
            return None

        self._route_count += 1

        # Need RTT data for meaningful routing
        has_rtt = any(node.probe_count > 0 for _, node in candidates)
        if not has_rtt:
            self._fallback_count += 1
            # No RTT data — use least-loaded fallback
            best = min(candidates, key=lambda x: x[1].active_requests)
            return RoutingScore(
                node_id=best[0],
                rtt_score=self._rtt_fallback,
                load_score=1.0 - best[1].load_fraction,
                combined_score=1.0 - best[1].load_fraction,
                selected=True,
            )

        # Score each candidate
        scores: list[RoutingScore] = []
        for nid, node in candidates:
            rtt = node.rtt_ema if node.probe_count > 0 else self._rtt_fallback
            rtt_score = 1.0 / (1.0 + rtt / 100.0)
            load_score = 1.0 - node.load_fraction
            combined = self._rtt_weight * rtt_score + self._load_weight * load_score

            scores.append(RoutingScore(
                node_id=nid,
                rtt_score=rtt_score,
                load_score=load_score,
                combined_score=combined,
            ))

        best = max(scores, key=lambda s: s.combined_score)
        best.selected = True
        return best

    def route_all_scores(self, exclude: set[str] | None = None) -> list[RoutingScore]:
        """Return scored routing for all nodes (for debugging/monitoring)."""
        if not self._nodes:
            return []

        scores = []
        for nid, node in self._nodes.items():
            if exclude and nid in exclude:
                continue
            rtt = node.rtt_ema if node.probe_count > 0 else self._rtt_fallback
            rtt_score = 1.0 / (1.0 + rtt / 100.0)
            load_score = 1.0 - node.load_fraction
            combined = self._rtt_weight * rtt_score + self._load_weight * load_score

            scores.append(RoutingScore(
                node_id=nid,
                rtt_score=rtt_score,
                load_score=load_score,
                combined_score=combined,
            ))

        scores.sort(key=lambda s: -s.combined_score)
        if scores:
            scores[0].selected = True
        return scores

    def needs_probe(self, node_id: str) -> bool:
        """Check if a node needs an RTT probe."""
        node = self._nodes.get(node_id)
        if node is None:
            return False
        if node.probe_count == 0:
            return True
        return time.monotonic() - node.last_probe > self._probe_interval

    def get_stats(self) -> dict:
        return {
            "num_nodes": len(self._nodes),
            "route_count": self._route_count,
            "fallback_count": self._fallback_count,
            "nodes": {
                nid: {
                    "rtt_ema_ms": round(node.rtt_ema, 2),
                    "rtt_var_ms": round(node.rtt_var, 2),
                    "timeout_ms": round(node.rtt_timeout_ms, 2),
                    "active_requests": node.active_requests,
                    "load": round(node.load_fraction, 4),
                    "probes": node.probe_count,
                }
                for nid, node in self._nodes.items()
            },
        }
