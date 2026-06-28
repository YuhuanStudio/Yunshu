"""Priority/FAIR scheduling gate (Wave 683) — model-free, deterministic.

W683 plumbed the scheduling policy from EngineCoreConfig → SchedulerConfig (it was
hardwired FCFS, so PRIORITY/FAIR preemption + aging were unreachable). This gates
the END-TO-END path deterministically (no model, no timing flakiness):

 (1) PLUMBING: EngineCore(scheduler_policy="priority"|"fair"|"fcfs"|bogus) resolves
     to the right SchedulingPolicy on the live scheduler (bogus → FCFS fallback).
 (2) ORDERING: the policy actually changes selection — a PRIORITY waiting queue
     pops the highest-priority request first even though it arrived last, while an
     FCFS queue preserves arrival order. This is the behaviour W683 unlocked.

Run: PYTHONPATH=. uv run python scripts/verify_priority_scheduling.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


class _FakeDetok:
    last_segment = ""
    def reset(self): self.last_segment = ""
    def add_token(self, t): self.last_segment = f"t{t}"
    def finalize(self): self.last_segment = ""


class _FakeTok:
    eos_token_ids = [3]
    has_thinking = False
    def encode(self, text, **kw): return list(range(len(text)))
    def decode(self, toks): return " ".join(f"t{t}" for t in toks)
    @property
    def detokenizer(self): return _FakeDetok()


def main() -> int:
    from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
    from yunshu_engine.scheduler import SchedulingPolicy
    from yunshu_engine.priority_queue import make_waiting_queue

    checks: dict[str, bool] = {}

    # (1) Plumbing: policy string → live scheduler policy.
    def _policy(p):
        core = EngineCore(None, _FakeTok(),
                          config=EngineCoreConfig(scheduler_policy=p),
                          executor=ThreadPoolExecutor(max_workers=1))
        return core.scheduler.config.policy

    checks["plumb default → FCFS"] = (
        EngineCore(None, _FakeTok(), executor=ThreadPoolExecutor(max_workers=1)
                   ).scheduler.config.policy == SchedulingPolicy.FCFS)
    checks["plumb 'priority' → PRIORITY"] = (_policy("priority") == SchedulingPolicy.PRIORITY)
    checks["plumb 'fair' → FAIR"] = (_policy("fair") == SchedulingPolicy.FAIR)
    checks["plumb 'bogus' → FCFS fallback"] = (_policy("bogus") == SchedulingPolicy.FCFS)

    # (2) Ordering: PRIORITY pops highest first (arrival-independent), FCFS keeps arrival.
    def _req(rid, prio):
        return SimpleNamespace(request_id=rid, sampling_params=SimpleNamespace(priority=prio))

    # Arrival order A(0), B(10), C(5); priority should pop B, C, A.
    pq = make_waiting_queue(SchedulingPolicy.PRIORITY)
    for rid, prio in [("A", 0), ("B", 10), ("C", 5)]:
        pq.push(_req(rid, prio), priority=prio)
    prio_order = [pq.pop().request_id for _ in range(3)]
    checks["PRIORITY pops high→low (B,C,A)"] = (prio_order == ["B", "C", "A"])

    fq = make_waiting_queue(SchedulingPolicy.FCFS)
    for rid, prio in [("A", 0), ("B", 10), ("C", 5)]:
        fq.push(_req(rid, prio), priority=prio)
    fcfs_order = [fq.pop().request_id for _ in range(3)]
    checks["FCFS keeps arrival (A,B,C)"] = (fcfs_order == ["A", "B", "C"])

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"  · priority_order={prio_order}  fcfs_order={fcfs_order}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
