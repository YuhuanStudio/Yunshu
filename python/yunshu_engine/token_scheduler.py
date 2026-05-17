from __future__ import annotations
"""Yunshu Token-Level Scheduler — fine-grained token budget allocation + priority inversion guard.

Three components:
1. TokenLevelScheduler — WFQ-based token budget allocation across requests
2. PriorityInversionGuard — detects and resolves priority inversion (inheritance / preemption)
3. FairnessTracker — Jain's fairness index for scheduling quality measurement

Architecture:
  Scheduler.step() → TokenLevelScheduler.compute_token_budget()
    → allocate prefill tokens (longer context = more tokens)
    → allocate decode tokens (priority + wait-time weighted)
  PriorityInversionGuard.check_inversion() before scheduling decisions
  FairnessTracker.record_allocation() after each step for metrics
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ── Token-Level Scheduling ──


@dataclass
class TokenBudgetAllocation:
    """Result of token budget computation for a single request.

    Attributes:
        request_id: The request identifier.
        prefill_tokens: Number of tokens allocated for prefill.
        decode_tokens: Number of tokens allocated for decode.
        weight: The computed WFQ weight for this request.
        priority: Original priority of the request.
        effective_priority: Priority after any boost from inversion guard.
    """

    request_id: str
    prefill_tokens: int = 0
    decode_tokens: int = 0
    weight: float = 0.0
    priority: int = 0
    effective_priority: int = 0


@dataclass
class SchedulableRequest:
    """Lightweight view of a request for token scheduling.

    Decoupled from engine.Request to keep the token scheduler independent
    of the full request lifecycle.
    """

    request_id: str
    priority: int = 0
    wait_time: float = 0.0  # seconds spent waiting
    context_length: int = 0  # prompt token count
    output_length: int = 0  # tokens generated so far
    is_prefilling: bool = False  # True if still in prefill phase
    effective_priority: int = 0  # boosted priority from inversion guard

    def __post_init__(self) -> None:
        if self.effective_priority == 0:
            self.effective_priority = self.priority


class PrefillStrategy(Enum):
    """Strategy for allocating prefill tokens."""

    PROPORTIONAL = auto()  # Proportional to context length
    EQUAL = auto()  # Equal split across requests
    PRIORITY_FIRST = auto()  # Higher priority gets more prefill budget


class DecodeStrategy(Enum):
    """Strategy for allocating decode tokens."""

    WFQ = auto()  # Weighted Fair Queuing (default)
    ROUND_ROBIN = auto()  # Simple round-robin
    PRIORITY_ONLY = auto()  # Strict priority ordering


class TokenLevelScheduler:
    """Fine-grained token-level scheduling within a batch step.

    Instead of scheduling at the request level (insert whole request or not),
    this scheduler distributes a token budget across requests using Weighted
    Fair Queuing (WFQ). Each request's weight is determined by:

      weight = w_priority * f(priority) + w_wait * g(wait_time) + w_context * h(context_length)

    where f, g, h are configurable weight functions. Default is linear scaling.

    The total token budget comes from:
    - Prefill budget: prefill_step_size * batch_utilization_factor
    - Decode budget: num_running_requests (one token per request per step)

    Usage:
        scheduler = TokenLevelScheduler()
        allocs = scheduler.compute_token_budget(requests, total_budget=2048)
        for alloc in allocs:
            # Use alloc.prefill_tokens and alloc.decode_tokens
    """

    def __init__(
        self,
        priority_weight: float = 0.5,
        wait_time_weight: float = 0.3,
        context_weight: float = 0.2,
        wait_time_boost_threshold: float = 1.0,
        wait_time_boost_factor: float = 2.0,
        min_prefill_tokens: int = 32,
        max_prefill_tokens: int = 2048,
        prefill_strategy: PrefillStrategy = PrefillStrategy.PROPORTIONAL,
        decode_strategy: DecodeStrategy = DecodeStrategy.WFQ,
    ) -> None:
        self.priority_weight = priority_weight
        self.wait_time_weight = wait_time_weight
        self.context_weight = context_weight
        self.wait_time_boost_threshold = wait_time_boost_threshold
        self.wait_time_boost_factor = wait_time_boost_factor
        self.min_prefill_tokens = min_prefill_tokens
        self.max_prefill_tokens = max_prefill_tokens
        self.prefill_strategy = prefill_strategy
        self.decode_strategy = decode_strategy

        # Stats tracking
        self._stats = {
            "budget_computations": 0,
            "total_prefill_tokens_allocated": 0,
            "total_decode_tokens_allocated": 0,
            "total_requests_scheduled": 0,
            "budget_utilization": [],
            "max_weight_seen": 0.0,
            "min_weight_seen": float("inf"),
            "steps_with_allocations": 0,
        }

    def compute_token_budget(
        self,
        requests: list[SchedulableRequest],
        total_budget: int,
    ) -> list[TokenBudgetAllocation]:
        """Distribute token budget across requests using WFQ.

        First separates requests into prefilling and decoding phases,
        then allocates budgets using the configured strategies.

        Args:
            requests: List of schedulable requests with metadata.
            total_budget: Total token budget for this step.

        Returns:
            List of TokenBudgetAllocation, one per request.
        """
        if not requests:
            return []

        self._stats["budget_computations"] += 1
        self._stats["total_requests_scheduled"] += len(requests)

        # Separate prefilling and decoding requests
        prefilling = [r for r in requests if r.is_prefilling]
        decoding = [r for r in requests if not r.is_prefilling]

        # Split budget: prefill gets proportional share based on count,
        # but at least 20% goes to decode if there are decode requests.
        n_prefill = len(prefilling)
        n_decode = len(decoding)

        if n_prefill == 0:
            prefill_budget = 0
            decode_budget = total_budget
        elif n_decode == 0:
            prefill_budget = total_budget
            decode_budget = 0
        else:
            # 60/40 split favoring decode when both exist
            prefill_budget = max(
                self.min_prefill_tokens * n_prefill,
                int(total_budget * 0.6),
            )
            decode_budget = total_budget - prefill_budget

        allocations = []

        # Allocate prefill tokens
        prefill_allocs = self.allocate_prefill_tokens(prefilling, prefill_budget)
        allocations.extend(prefill_allocs)

        # Allocate decode tokens
        decode_allocs = self.allocate_decode_tokens(decoding, decode_budget)
        allocations.extend(decode_allocs)

        # Track stats
        total_allocated = sum(a.prefill_tokens + a.decode_tokens for a in allocations)
        utilization = total_allocated / total_budget if total_budget > 0 else 0.0
        self._stats["total_prefill_tokens_allocated"] += sum(
            a.prefill_tokens for a in prefill_allocs
        )
        self._stats["total_decode_tokens_allocated"] += sum(
            a.decode_tokens for a in decode_allocs
        )
        self._stats["budget_utilization"].append(utilization)
        # Keep only last 1000 utilization samples
        if len(self._stats["budget_utilization"]) > 1000:
            self._stats["budget_utilization"] = self._stats["budget_utilization"][-1000:]

        weights = [a.weight for a in allocations if a.weight > 0]
        if weights:
            self._stats["max_weight_seen"] = max(
                self._stats["max_weight_seen"], max(weights)
            )
            self._stats["min_weight_seen"] = min(
                self._stats["min_weight_seen"], min(weights)
            )

        return allocations

    def allocate_prefill_tokens(
        self,
        requests: list[SchedulableRequest],
        prefill_budget: int,
    ) -> list[TokenBudgetAllocation]:
        """Allocate prefill tokens across prefilling requests.

        Strategies:
        - PROPORTIONAL: longer context gets more tokens
        - EQUAL: equal split
        - PRIORITY_FIRST: higher priority gets more tokens

        Each allocation is clamped to [min_prefill_tokens, max_prefill_tokens].
        """
        if not requests:
            return []
        if prefill_budget <= 0:
            return [
                TokenBudgetAllocation(
                    request_id=r.request_id,
                    prefill_tokens=0,
                    decode_tokens=0,
                    weight=0.0,
                    priority=r.priority,
                    effective_priority=r.effective_priority,
                )
                for r in requests
            ]

        allocations: list[TokenBudgetAllocation] = []

        if self.prefill_strategy == PrefillStrategy.EQUAL:
            per_request = prefill_budget // len(requests)
            for req in requests:
                tokens = max(self.min_prefill_tokens, min(per_request, self.max_prefill_tokens))
                allocations.append(
                    TokenBudgetAllocation(
                        request_id=req.request_id,
                        prefill_tokens=tokens,
                        weight=1.0,
                        priority=req.priority,
                        effective_priority=req.effective_priority,
                    )
                )

        elif self.prefill_strategy == PrefillStrategy.PRIORITY_FIRST:
            # Sort by priority descending
            sorted_reqs = sorted(requests, key=lambda r: r.effective_priority, reverse=True)
            weights = []
            for req in sorted_reqs:
                w = self._compute_weight(req)
                weights.append(w)
            total_w = sum(weights) or 1.0
            for i, req in enumerate(sorted_reqs):
                raw = int(prefill_budget * weights[i] / total_w)
                tokens = max(self.min_prefill_tokens, min(raw, self.max_prefill_tokens))
                allocations.append(
                    TokenBudgetAllocation(
                        request_id=req.request_id,
                        prefill_tokens=tokens,
                        weight=weights[i],
                        priority=req.priority,
                        effective_priority=req.effective_priority,
                    )
                )

        else:  # PROPORTIONAL (default)
            total_context = sum(r.context_length for r in requests) or 1
            for req in requests:
                fraction = req.context_length / total_context
                raw = int(prefill_budget * fraction)
                tokens = max(self.min_prefill_tokens, min(raw, self.max_prefill_tokens))
                w = self._compute_weight(req)
                allocations.append(
                    TokenBudgetAllocation(
                        request_id=req.request_id,
                        prefill_tokens=tokens,
                        weight=w,
                        priority=req.priority,
                        effective_priority=req.effective_priority,
                    )
                )

        # Enforce total budget: scale down if over budget
        total_alloc = sum(a.prefill_tokens for a in allocations)
        if total_alloc > prefill_budget and len(allocations) > 0:
            scale = prefill_budget / total_alloc
            for alloc in allocations:
                alloc.prefill_tokens = max(
                    self.min_prefill_tokens,
                    int(alloc.prefill_tokens * scale),
                )

        return allocations

    def allocate_decode_tokens(
        self,
        requests: list[SchedulableRequest],
        decode_budget: int,
    ) -> list[TokenBudgetAllocation]:
        """Allocate decode tokens across decoding requests using WFQ.

        WFQ: each request gets tokens proportional to its weight.
        Weight = f(priority, wait_time, output_length).

        Typically each request gets exactly 1 decode token per step
        (one forward pass = one token per request). The decode budget
        allows for speculative decoding where some requests may get
        multiple tokens (draft + verify).
        """
        if not requests:
            return []

        allocations: list[TokenBudgetAllocation] = []

        if self.decode_strategy == DecodeStrategy.ROUND_ROBIN:
            per_request = max(1, decode_budget // max(len(requests), 1))
            remaining_budget = decode_budget
            for req in requests:
                tokens = min(per_request, remaining_budget) if remaining_budget > 0 else 0
                allocations.append(
                    TokenBudgetAllocation(
                        request_id=req.request_id,
                        decode_tokens=tokens,
                        weight=1.0,
                        priority=req.priority,
                        effective_priority=req.effective_priority,
                    )
                )
                remaining_budget -= tokens

        elif self.decode_strategy == DecodeStrategy.PRIORITY_ONLY:
            # Strict priority: higher priority gets all tokens first
            sorted_reqs = sorted(
                requests, key=lambda r: r.effective_priority, reverse=True
            )
            remaining = decode_budget
            for req in sorted_reqs:
                # Each request gets 1 decode token per step (one forward pass = one
                # token per request).  Only allocate if budget remains.
                tokens = 1 if remaining > 0 else 0
                allocations.append(
                    TokenBudgetAllocation(
                        request_id=req.request_id,
                        decode_tokens=tokens,
                        weight=float(req.effective_priority),
                        priority=req.priority,
                        effective_priority=req.effective_priority,
                    )
                )
                remaining -= tokens

        else:  # WFQ (default)
            weights = [self._compute_weight(req) for req in requests]
            total_w = sum(weights) or 1.0
            remaining_budget = decode_budget
            for i, req in enumerate(requests):
                # Allocate proportional to weight, but respect remaining budget.
                # Use floor division so total does not systematically exceed budget.
                raw = int(decode_budget * weights[i] / total_w)
                tokens = max(1, min(raw, remaining_budget)) if remaining_budget > 0 else 0
                allocations.append(
                    TokenBudgetAllocation(
                        request_id=req.request_id,
                        decode_tokens=tokens,
                        weight=weights[i],
                        priority=req.priority,
                        effective_priority=req.effective_priority,
                    )
                )
                remaining_budget -= tokens

        return allocations

    def _compute_weight(self, req: SchedulableRequest) -> float:
        """Compute WFQ weight for a request.

        weight = w_p * norm(priority) + w_w * norm(wait_time) + w_c * norm(context)

        Normalization: each factor is scaled relative to a reasonable maximum:
        - priority: [0, 100] range, normalized by 100
        - wait_time: boosted by boost_factor if above threshold, normalized by 30s max
        - context: normalized by 8192 tokens max
        """
        # Priority component: linear in [0, 1]
        p_norm = min(req.effective_priority, 100) / 100.0

        # Wait time component: apply boost for long waits, cap at 30s
        wait = req.wait_time
        if wait > self.wait_time_boost_threshold:
            wait *= self.wait_time_boost_factor
        w_norm = min(wait, 30.0) / 30.0

        # Context length component: cap at 8192
        c_norm = min(req.context_length, 8192) / 8192.0

        weight = (
            self.priority_weight * p_norm
            + self.wait_time_weight * w_norm
            + self.context_weight * c_norm
        )

        return max(weight, 0.001)  # Floor to avoid zero weight

    def get_stats(self) -> dict:
        """Return scheduling statistics."""
        stats = dict(self._stats)
        if stats["budget_utilization"]:
            recent = stats["budget_utilization"][-100:]
            stats["avg_budget_utilization"] = round(sum(recent) / len(recent), 3)
        else:
            stats["avg_budget_utilization"] = 0.0
        if stats["min_weight_seen"] == float("inf"):
            stats["min_weight_seen"] = 0.0
        return stats


# ── Priority Inversion Guard ──


class ResolutionStrategy(Enum):
    """Strategy for resolving priority inversion."""

    INHERITANCE = auto()  # Boost low-priority to match high (default)
    PREEMPTION = auto()  # Preempt low-priority request


@dataclass
class InversionEvent:
    """Record of a priority inversion detection."""

    timestamp: float
    low_request_id: str
    high_request_id: str
    low_priority: int
    high_priority: int
    resource_type: str = "kv_slots"  # What resource is contested
    resolution: str = ""  # "inheritance" or "preemption"
    boost_applied: int = 0  # Priority boost amount


@dataclass
class InversionStats:
    """Accumulated priority inversion statistics."""

    inversions_detected: int = 0
    inheritance_applied: int = 0
    preemption_applied: int = 0
    total_boost: int = 0
    avg_boost: float = 0.0


class PriorityInversionGuard:
    """Detects and prevents priority inversion in the scheduling queue.

    Priority inversion occurs when a low-priority request holds resources
    (e.g., KV cache slots) that a high-priority request needs. This guard
    detects such situations and applies either:

    1. Priority inheritance: temporarily boost the low-priority request's
       effective priority to match the high-priority request, so it finishes
       faster and releases resources sooner. (default)

    2. Preemption: immediately preempt the low-priority request, freeing
       its resources for the high-priority request.

    Detection criteria:
    - A waiting request with priority P_high exists
    - A running request with priority P_low exists where P_low < P_high
    - The system is at capacity (no free slots)
    - The low-priority request has been running for longer than a threshold

    Usage:
        guard = PriorityInversionGuard()
        inversions = guard.check_inversion(running, waiting)
        for inv in inversions:
            guard.apply_inheritance(inv.low_request, inv.high_request)
    """

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.INHERITANCE,
        running_time_threshold: float = 2.0,
        max_boost_duration: float = 10.0,
        min_priority_gap: int = 2,
    ) -> None:
        """Initialize the priority inversion guard.

        Args:
            strategy: Resolution strategy (INHERITANCE or PREEMPTION).
            running_time_threshold: Minimum seconds a low-priority request
                must have been running before it's considered for inversion.
            max_boost_duration: Maximum seconds a priority boost lasts.
            min_priority_gap: Minimum difference between high and low priority
                to trigger inversion detection.
        """
        self.strategy = strategy
        self.running_time_threshold = running_time_threshold
        self.max_boost_duration = max_boost_duration
        self.min_priority_gap = min_priority_gap

        # Active boosts: request_id → (boosted_priority, original_priority, boost_time)
        self._active_boosts: dict[str, tuple[int, int, float]] = {}

        # Event log for analysis (bounded to prevent unbounded memory growth)
        self._events: list[InversionEvent] = []
        self._max_events = 1000

        # Accumulated stats
        self._stats = InversionStats()

    def check_inversion(
        self,
        running_requests: list[SchedulableRequest],
        waiting_requests: list[SchedulableRequest],
        capacity: int = 0,
    ) -> list[InversionEvent]:
        """Detect priority inversion scenarios.

        An inversion is detected when:
        1. There are waiting requests with higher priority than some running requests
        2. The system is at or near capacity
        3. The low-priority request has been running long enough

        Args:
            running_requests: Currently running requests.
            waiting_requests: Requests waiting to be scheduled.
            capacity: Total system capacity (0 = unknown, check all).

        Returns:
            List of detected inversion events (may be empty).
        """
        if not running_requests or not waiting_requests:
            return []

        # Expire old boosts
        self._expire_boosts()

        inversions: list[InversionEvent] = []
        now = time.monotonic()

        # Find highest-priority waiting request
        highest_waiting = max(waiting_requests, key=lambda r: r.priority)

        # Find running requests with lower priority
        for running in running_requests:
            # Skip already-boosted requests
            if running.request_id in self._active_boosts:
                continue

            priority_gap = highest_waiting.priority - running.priority
            if priority_gap < self.min_priority_gap:
                continue

            # Check running time threshold
            if running.wait_time < self.running_time_threshold:
                continue

            # Inversion detected
            event = InversionEvent(
                timestamp=now,
                low_request_id=running.request_id,
                high_request_id=highest_waiting.request_id,
                low_priority=running.priority,
                high_priority=highest_waiting.priority,
                resolution=self.strategy.name.lower(),
                boost_applied=0,
            )
            inversions.append(event)

        return inversions

    def apply_inheritance(
        self,
        low_req: SchedulableRequest,
        high_req: SchedulableRequest,
    ) -> int:
        """Apply priority inheritance to resolve inversion.

        Temporarily boosts the low-priority request's effective priority
        to match the high-priority request. The boost expires after
        max_boost_duration seconds.

        Args:
            low_req: The low-priority request holding resources.
            high_req: The high-priority request waiting for resources.

        Returns:
            The amount of priority boost applied.
        """
        boost_amount = high_req.priority - low_req.effective_priority
        if boost_amount <= 0:
            return 0

        original_priority = low_req.priority
        boosted_priority = high_req.priority
        now = time.monotonic()

        low_req.effective_priority = boosted_priority
        self._active_boosts[low_req.request_id] = (
            boosted_priority,
            original_priority,
            now,
        )

        # Update stats
        self._stats.inheritance_applied += 1
        self._stats.total_boost += boost_amount
        if self._stats.inheritance_applied > 0:
            self._stats.avg_boost = round(
                self._stats.total_boost / self._stats.inheritance_applied, 2
            )

        logger.debug(
            f"Priority inheritance: {low_req.request_id} boosted from "
            f"{original_priority} to {boosted_priority} "
            f"(waiting: {high_req.request_id} at priority {high_req.priority})"
        )

        return boost_amount

    def apply_preemption(
        self,
        low_req: SchedulableRequest,
    ) -> bool:
        """Apply preemption to resolve inversion.

        Marks the low-priority request for preemption. The actual preemption
        is performed by the scheduler (which manages the BatchGenerator).

        Args:
            low_req: The low-priority request to preempt.

        Returns:
            True if preemption was applied, False if not (e.g., request
            was already boosted).
        """
        # Don't preempt a boosted request
        if low_req.request_id in self._active_boosts:
            return False

        self._stats.preemption_applied += 1

        logger.info(
            f"Priority inversion preemption: {low_req.request_id} "
            f"(priority={low_req.priority})"
        )

        return True

    def resolve(
        self,
        running_requests: list[SchedulableRequest],
        waiting_requests: list[SchedulableRequest],
        capacity: int = 0,
    ) -> list[InversionEvent]:
        """Convenience method: detect inversions and apply resolution.

        Args:
            running_requests: Currently running requests.
            waiting_requests: Requests waiting to be scheduled.
            capacity: Total system capacity.

        Returns:
            List of resolved inversion events.
        """
        inversions = self.check_inversion(running_requests, waiting_requests, capacity)

        for event in inversions:
            # Find the actual request objects
            low_req = next(
                (r for r in running_requests if r.request_id == event.low_request_id),
                None,
            )
            high_req = next(
                (r for r in waiting_requests if r.request_id == event.high_request_id),
                None,
            )
            if low_req is None or high_req is None:
                continue

            self._stats.inversions_detected += 1

            if self.strategy == ResolutionStrategy.INHERITANCE:
                boost = self.apply_inheritance(low_req, high_req)
                event.boost_applied = boost
            else:
                self.apply_preemption(low_req)

            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events // 2:]

        return inversions

    def get_boost(self, request_id: str) -> int | None:
        """Get the current boosted priority for a request, if any.

        Returns:
            The boosted priority, or None if no active boost.
        """
        entry = self._active_boosts.get(request_id)
        if entry is None:
            return None
        boosted, original, boost_time = entry
        if time.monotonic() - boost_time > self.max_boost_duration:
            self._active_boosts.pop(request_id, None)
            return None
        return boosted

    def clear_boost(self, request_id: str) -> None:
        """Manually clear a priority boost for a request."""
        self._active_boosts.pop(request_id, None)

    def _expire_boosts(self) -> None:
        """Remove expired priority boosts."""
        now = time.monotonic()
        expired = [
            rid
            for rid, (_, _, t) in self._active_boosts.items()
            if now - t > self.max_boost_duration
        ]
        for rid in expired:
            self._active_boosts.pop(rid, None)

    def get_stats(self) -> dict:
        """Return priority inversion statistics."""
        return {
            "strategy": self.strategy.name.lower(),
            "inversions_detected": self._stats.inversions_detected,
            "inheritance_applied": self._stats.inheritance_applied,
            "preemption_applied": self._stats.preemption_applied,
            "total_boost": self._stats.total_boost,
            "avg_boost": self._stats.avg_boost,
            "active_boosts": len(self._active_boosts),
            "total_events": len(self._events),
            "running_time_threshold": self.running_time_threshold,
            "min_priority_gap": self.min_priority_gap,
        }


# ── Fairness Tracker ──


@dataclass
class AllocationRecord:
    """Record of a single token allocation event."""

    request_id: str
    tokens_allocated: int
    timestamp: float


@dataclass
class CompletionRecord:
    """Record of a request completion."""

    request_id: str
    wait_time: float  # Time spent in waiting queue
    total_time: float  # Total time from arrival to completion
    total_tokens: int  # Total tokens received


class FairnessTracker:
    """Tracks scheduling fairness across requests.

    Measures:
    - Token allocation variance: how evenly tokens are distributed
    - Wait time variance: how evenly wait times are distributed
    - Completion time variance: how evenly completion times are distributed
    - Jain's fairness index: overall fairness metric (0-1, 1 = perfectly fair)

    Jain's fairness index formula:
        J(x_1, ..., x_n) = (sum(x_i))^2 / (n * sum(x_i^2))

    where x_i is the allocation proportion for request i.

    Usage:
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 80)
        fairness = tracker.compute_fairness()  # 0-1
        unfair = tracker.get_unfair_requests()  # requests below threshold
    """

    def __init__(
        self,
        unfairness_threshold: float = 0.7,
        max_history: int = 10000,
        window_size: int = 1000,
    ) -> None:
        """Initialize the fairness tracker.

        Args:
            unfairness_threshold: Requests below this fraction of the
                mean allocation are considered "unfair" (default 0.7).
            max_history: Maximum number of allocation records to keep.
            window_size: Window size for rolling fairness computation.
        """
        self.unfairness_threshold = unfairness_threshold
        self.max_history = max_history
        self.window_size = window_size

        # Per-request allocation totals
        self._allocations: dict[str, int] = {}  # request_id → total tokens
        self._allocation_count: dict[str, int] = {}  # request_id → allocation count

        # Allocation history for rolling fairness
        self._history: list[AllocationRecord] = []

        # Completion records
        self._completions: list[CompletionRecord] = []

        # Stats
        self._total_tokens_allocated: int = 0
        self._total_allocations: int = 0
        self._total_unique_requests: int = 0  # Unique request IDs ever tracked

    def record_allocation(
        self, request_id: str, tokens_allocated: int
    ) -> None:
        """Record a token allocation event.

        Args:
            request_id: The request identifier.
            tokens_allocated: Number of tokens allocated in this event.
        """
        # Track unique requests (first allocation for this request_id)
        if request_id not in self._allocations:
            self._total_unique_requests += 1
        self._allocations[request_id] = (
            self._allocations.get(request_id, 0) + tokens_allocated
        )
        self._allocation_count[request_id] = (
            self._allocation_count.get(request_id, 0) + 1
        )
        self._total_tokens_allocated += tokens_allocated
        self._total_allocations += 1

        # Add to history
        self._history.append(
            AllocationRecord(
                request_id=request_id,
                tokens_allocated=tokens_allocated,
                timestamp=time.monotonic(),
            )
        )

        # Trim history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history // 2 :]

    def record_completion(
        self, request_id: str, wait_time: float, total_time: float
    ) -> None:
        """Record a request completion.

        Args:
            request_id: The request identifier.
            wait_time: Time spent in waiting queue (seconds).
            total_time: Total time from arrival to completion (seconds).
        """
        total_tokens = self._allocations.get(request_id, 0)
        self._completions.append(
            CompletionRecord(
                request_id=request_id,
                wait_time=wait_time,
                total_time=total_time,
                total_tokens=total_tokens,
            )
        )
        # Clean up per-request allocation tracking to prevent unbounded memory growth.
        # The completion record preserves the total_tokens snapshot for variance calculations.
        self._allocations.pop(request_id, None)
        self._allocation_count.pop(request_id, None)
        # Keep completion records bounded
        if len(self._completions) > self.max_history:
            self._completions = self._completions[-self.max_history // 2 :]

    def compute_fairness(self) -> float:
        """Compute Jain's fairness index across all tracked requests.

        Jain's fairness index:
            J = (sum(x_i))^2 / (n * sum(x_i^2))

        where x_i = total tokens allocated to request i.

        Returns:
            Fairness index in [0, 1]. 1.0 = perfectly fair.
            Returns 1.0 if no allocations or only one request.
        """
        if not self._allocations:
            return 1.0

        values = list(self._allocations.values())
        n = len(values)
        if n <= 1:
            return 1.0

        sum_x = sum(values)
        if sum_x == 0:
            return 1.0

        sum_x_sq = sum(v * v for v in values)

        # Jain's fairness index
        j = (sum_x * sum_x) / (n * sum_x_sq)
        return min(max(j, 0.0), 1.0)

    def compute_windowed_fairness(self) -> float:
        """Compute Jain's fairness index over the recent window.

        Uses the last `window_size` allocation records to compute
        fairness, giving a rolling view of scheduling quality.
        """
        if not self._history:
            return 1.0

        recent = self._history[-self.window_size :]
        # Aggregate per-request tokens in window
        window_totals: dict[str, int] = {}
        for record in recent:
            window_totals[record.request_id] = (
                window_totals.get(record.request_id, 0) + record.tokens_allocated
            )

        if not window_totals:
            return 1.0

        values = list(window_totals.values())
        n = len(values)
        if n <= 1:
            return 1.0

        sum_x = sum(values)
        if sum_x == 0:
            return 1.0

        sum_x_sq = sum(v * v for v in values)
        j = (sum_x * sum_x) / (n * sum_x_sq)
        return min(max(j, 0.0), 1.0)

    def get_unfair_requests(self) -> list[dict]:
        """Return requests that are unfairly treated.

        A request is considered "unfair" if its total allocation is below
        `unfairness_threshold` fraction of the mean allocation.

        Returns:
            List of dicts with request_id, total_tokens, expected_tokens, ratio.
        """
        if not self._allocations:
            return []

        values = list(self._allocations.values())
        mean_alloc = sum(values) / len(values) if values else 0.0
        if mean_alloc == 0:
            return []

        threshold = mean_alloc * self.unfairness_threshold
        unfair = []
        for rid, total in self._allocations.items():
            ratio = total / mean_alloc
            if total < threshold:
                unfair.append(
                    {
                        "request_id": rid,
                        "total_tokens": total,
                        "expected_tokens": round(mean_alloc),
                        "ratio": round(ratio, 3),
                    }
                )

        return unfair

    def get_allocation_variance(self) -> float:
        """Compute variance of token allocations across requests.

        Returns:
            Variance of total token allocations (0 = perfectly fair).
        """
        if len(self._allocations) < 2:
            return 0.0

        values = list(self._allocations.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance

    def get_wait_time_variance(self) -> float:
        """Compute variance of wait times from completed requests.

        Returns:
            Variance of wait times (0 = all waited the same).
        """
        if len(self._completions) < 2:
            return 0.0

        waits = [c.wait_time for c in self._completions]
        mean = sum(waits) / len(waits)
        variance = sum((w - mean) ** 2 for w in waits) / len(waits)
        return variance

    def get_completion_time_variance(self) -> float:
        """Compute variance of total completion times.

        Returns:
            Variance of total times (0 = all completed in same time).
        """
        if len(self._completions) < 2:
            return 0.0

        times = [c.total_time for c in self._completions]
        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        return variance

    def reset(self) -> None:
        """Reset all tracking state."""
        self._allocations.clear()
        self._allocation_count.clear()
        self._history.clear()
        self._completions.clear()
        self._total_tokens_allocated = 0
        self._total_allocations = 0
        self._total_unique_requests = 0

    def get_stats(self) -> dict:
        """Return fairness tracking statistics."""
        return {
            "jains_fairness_index": round(self.compute_fairness(), 4),
            "windowed_fairness_index": round(self.compute_windowed_fairness(), 4),
            "allocation_variance": round(self.get_allocation_variance(), 2),
            "wait_time_variance": round(self.get_wait_time_variance(), 4),
            "completion_time_variance": round(self.get_completion_time_variance(), 4),
            "unfair_requests": len(self.get_unfair_requests()),
            "total_requests_tracked": self._total_unique_requests,
            "total_tokens_allocated": self._total_tokens_allocated,
            "total_allocations": self._total_allocations,
            "total_completions": len(self._completions),
            "history_size": len(self._history),
            "unfairness_threshold": self.unfairness_threshold,
        }
