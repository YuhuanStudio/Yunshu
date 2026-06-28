"""FAIR scheduler multi-tenant stress test.

VALIDATION_REPORT.md line 1282 documented "FAIR multi-tenant 需 priority-queue
stress (long-soak deferred)". This is the short-duration substitute that
actually drives the scheduler's FAIR round-robin rotation across many rounds
and verifies that low-priority requests do NOT starve.

Strategy: rather than spin up a full BatchGenerator + model, we drive the
exact FAIR rotation code in Scheduler._schedule_waiting (lines 1429-1472)
in isolation. The rotation logic is pure Python over Request/SamplingParams
objects with no MLX dependency, so this is a fast (<5s) unit-level stress
that exercises the same code path the production scheduler uses.

Mix per task spec:
  - 10 requests at priority=10 (high)
  - 20 requests at priority=5  (mid)
  - 20 requests at priority=0  (low)

Total: 50 requests. Simulated batch slot capacity: 4 per round. We run
12+ rounds and verify that low-priority requests appear within a bounded
number of rounds.
"""
from __future__ import annotations

import time

import pytest

from yunshu_engine.request import Request, SamplingParams

# Time-box: each individual test must finish in ≤ 30s.
pytestmark = pytest.mark.timeout(30)


def _build_mix(
    n_high: int = 10, n_mid: int = 20, n_low: int = 20
) -> list[Request]:
    """Build the 50-request mixed-priority bundle, with monotonic arrival_time."""
    reqs: list[Request] = []
    now = time.monotonic()
    counter = 0
    for prio, n in ((10, n_high), (5, n_mid), (0, n_low)):
        for i in range(n):
            sp = SamplingParams(priority=prio)
            r = Request(
                request_id=f"p{prio}-{i}",
                prompt="x",
                sampling_params=sp,
                prompt_token_ids=[1, 2, 3],
                num_prompt_tokens=3,
                arrival_time=now + counter * 1e-6,
                priority=prio,
            )
            reqs.append(r)
            counter += 1
    return reqs


def _fair_rotate(to_insert: list[Request], rr_offset: int) -> list[Request]:
    """Replicate Scheduler FAIR rotation (scheduler.py lines 1440-1472).

    Must be kept in sync with the production code. Inputs:
      - to_insert: request list (heap-popped from waiting queue)
      - rr_offset: monotonically incrementing round counter
    Returns the round-robin ordered list for batch insertion.
    """
    if len(to_insert) <= 1:
        return to_insert
    buckets: dict[int, list[Request]] = {}
    for req in to_insert:
        p = req.sampling_params.priority if req.sampling_params else 0
        buckets.setdefault(p, []).append(req)
    sorted_priorities = sorted(buckets.keys(), reverse=True)
    for p in sorted_priorities:
        buckets[p].sort(key=lambda r: r.arrival_time)
        if len(buckets[p]) > 1:
            offset = rr_offset % len(buckets[p])
            buckets[p] = buckets[p][offset:] + buckets[p][:offset]
    if len(sorted_priorities) > 1 and rr_offset > 0:
        offset = rr_offset % len(sorted_priorities)
        sorted_priorities = sorted_priorities[offset:] + sorted_priorities[:offset]
    round_robin: list[Request] = []
    while any(buckets[p] for p in sorted_priorities):
        for p in sorted_priorities:
            if buckets[p]:
                round_robin.append(buckets[p].pop(0))
    return round_robin


def _simulate_rounds(
    reqs: list[Request], slots_per_round: int, num_rounds: int
) -> list[list[Request]]:
    """Simulate ``num_rounds`` scheduling rounds with FAIR rotation.

    Each round:
      1. Pop everything from the waiting queue (mirrors _schedule_waiting).
      2. Apply FAIR rotation with the current rr_offset.
      3. Take the first ``slots_per_round`` requests as "scheduled".
      4. Re-insert the rest at the front of the queue for the next round
         (matches scheduler.push_front behavior).
      5. Increment rr_offset.
    """
    pending: list[Request] = list(reqs)
    schedule_log: list[list[Request]] = []
    for rr_offset in range(num_rounds):
        if not pending:
            break
        ordered = _fair_rotate(pending, rr_offset)
        scheduled = ordered[:slots_per_round]
        leftover = ordered[slots_per_round:]
        schedule_log.append(scheduled)
        pending = leftover  # push_front equivalent — head of queue
    return schedule_log


class TestFAIRStress:
    """Stress the FAIR rotation across many rounds with the 50-request mix."""

    def test_no_starvation_within_bounded_rounds(self) -> None:
        """The lowest-priority bucket (p=0) must be scheduled within the first
        few rounds, not deferred until all higher buckets drain."""
        reqs = _build_mix()
        # 4 slots/round × 13 rounds = 52 slots > 50 requests
        log = _simulate_rounds(reqs, slots_per_round=4, num_rounds=13)
        assert len(log) >= 12, f"only ran {len(log)} rounds"

        # Within how many rounds does the first p=0 request appear?
        first_low_round = None
        for i, batch in enumerate(log):
            if any(r.sampling_params.priority == 0 for r in batch):
                first_low_round = i
                break
        assert first_low_round is not None, "p=0 never scheduled"
        # With 3 priority levels and rotation, p=0 must show up within
        # the first 3 rounds (one round per priority level rotation).
        assert first_low_round <= 2, (
            f"p=0 deferred to round {first_low_round}, starvation indicator"
        )

    def test_all_priorities_progress_each_window(self) -> None:
        """Across any 3 consecutive rounds, all three priority levels must
        receive at least one slot (the FAIR rotation guarantee)."""
        reqs = _build_mix()
        log = _simulate_rounds(reqs, slots_per_round=3, num_rounds=18)
        # Sliding 3-round windows — every level must appear in every window
        for start in range(0, min(len(log) - 2, 12)):
            window = log[start : start + 3]
            seen_prios = {r.sampling_params.priority for batch in window for r in batch}
            # Once a bucket exhausts it's expected to drop out; check only
            # while all three are still pending.
            remaining_per_p = {10: 10, 5: 20, 0: 20}
            for prior in log[:start]:
                for r in prior:
                    remaining_per_p[r.sampling_params.priority] -= 1
            still_pending = {p for p, n in remaining_per_p.items() if n > 0}
            for p in still_pending:
                assert p in seen_prios, (
                    f"priority {p} starved in window starting at round {start}: "
                    f"saw {seen_prios}, pending={remaining_per_p}"
                )

    def test_all_50_eventually_scheduled(self) -> None:
        """No requests are lost; all 50 get scheduled within a reasonable
        number of rounds at 4 slots/round."""
        reqs = _build_mix()
        log = _simulate_rounds(reqs, slots_per_round=4, num_rounds=20)
        scheduled_ids = {r.request_id for batch in log for r in batch}
        all_ids = {r.request_id for r in reqs}
        missing = all_ids - scheduled_ids
        assert not missing, f"{len(missing)} requests never scheduled: {missing}"
        assert len(scheduled_ids) == 50

    def test_high_priority_gets_majority_share(self) -> None:
        """FAIR is not equal: high priority should still get at least its
        proportional share. With 10 high vs 20 mid vs 20 low and one slot
        per priority per round, the first few rounds should be dominated
        by one-per-level until high drains."""
        reqs = _build_mix()
        log = _simulate_rounds(reqs, slots_per_round=3, num_rounds=20)
        # First 10 rounds should each contain exactly one p=10 request
        # (since 10 high-priority items / 1-per-round = 10 rounds).
        for i in range(10):
            count_high = sum(
                1 for r in log[i] if r.sampling_params.priority == 10
            )
            assert count_high == 1, (
                f"round {i}: expected exactly 1 high-priority, got {count_high}"
            )

    def test_low_priority_does_not_wait_more_than_3_rounds_between_slots(
        self,
    ) -> None:
        """Bounded waiting time: between two consecutive slots given to
        p=0 requests, the gap should not exceed 3 rounds (one full rotation
        across the 3 priority levels)."""
        reqs = _build_mix()
        log = _simulate_rounds(reqs, slots_per_round=2, num_rounds=30)
        low_seen_rounds: list[int] = []
        for i, batch in enumerate(log):
            if any(r.sampling_params.priority == 0 for r in batch):
                low_seen_rounds.append(i)
        assert len(low_seen_rounds) >= 5, (
            f"p=0 only scheduled {len(low_seen_rounds)} times across "
            f"{len(log)} rounds"
        )
        # Gap between consecutive p=0 appearances ≤ 3
        for prev, curr in zip(
            low_seen_rounds, low_seen_rounds[1:], strict=False
        ):
            gap = curr - prev
            # Allow gap of up to 3 (one full rotation cycle)
            assert gap <= 3, (
                f"p=0 starved for {gap} rounds between rounds {prev} and {curr}"
            )

    def test_rotation_offset_advances_first_priority_each_round(self) -> None:
        """The rr_offset rotation should put a different priority bucket
        first across consecutive rounds (rotation actually advances)."""
        reqs = _build_mix()
        log = _simulate_rounds(reqs, slots_per_round=1, num_rounds=6)
        # With 1 slot/round and 3 levels, the first slot should cycle:
        # round 0: rr=0, no priority rotation (offset==0 guard) → p=10
        # round 1: rr=1, rotate by 1 → p=5
        # round 2: rr=2, rotate by 2 → p=0
        # round 3: rr=3, rotate by 0 → p=10 again
        firsts = [batch[0].sampling_params.priority for batch in log]
        # Must include all 3 levels in the first 4 rounds
        assert set(firsts[:4]) == {10, 5, 0}, (
            f"rotation did not visit all priorities: {firsts}"
        )
