"""CPU checks of the real-engine jump measurement driver (no engine, no browser)."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "brand_live_jump",
    Path(__file__).resolve().parents[2] / "scripts" / "research" / "brand_live_jump.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_plan_is_bursty_deterministic_and_inside_the_run():
    plan = mod.build_plan(29, 90)
    assert plan == mod.build_plan(29, 90)
    assert 8 <= len(plan) <= 60
    assert all(0 <= at < 90 for at, _ in plan)
    times = [at for at, _ in plan]
    assert times == sorted(times)
    # bursts: some requests start within 0.5 s of the previous one
    assert sum(b - a < 0.5 for a, b in zip(times, times[1:], strict=False)) >= len(plan) // 2
    assert {k for _, k in plan} <= {"short", "long", "cancel"}


def test_judge_flags_failed_requests_and_failed_jump_rules():
    ok = [{"kind": "short", "status": 200, "events": 5}] * 8
    assert mod.judge(ok, 0) == []
    assert any(
        "failed" in p for p in mod.judge(ok + [{"kind": "short", "error": "x"}], 0)
    )
    assert any("jump rules" in p for p in mod.judge(ok, 1))
    assert any("only" in p for p in mod.judge(ok[:3], 0))
