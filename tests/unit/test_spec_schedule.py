"""Cost-aware draft budget."""

from __future__ import annotations

from yunshu_engine.spec_schedule import WARMUP, NodeBudget


def _run(budget, cost_fn, land_p, rounds=200):
    """Feed the budget rounds whose cost and landings follow the given model."""
    picks = []
    for r in range(rounds):
        n = budget.choose(room=100)
        picks.append(n)
        landed = [i for i in range(n) if land_p[i] > (0.5 + 0.0 * r)]
        budget.observe(n, landed, cost_fn(n))
    return picks


def test_warmup_uses_full_budget():
    b = NodeBudget(7)
    assert [b.choose(100) for _ in range(1)] == [7]
    for _ in range(WARMUP):
        b.observe(7, [0, 1], 55.0)
    assert b.rounds == WARMUP


def test_flat_cost_keeps_wide_trees():
    b = NodeBudget(7, prior=[0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15])
    land = [1, 1, 1, 0.9, 0.6, 0.6, 0.6]  # >0.5 -> lands
    picks = _run(b, lambda n: 43.0 + (10.0 + 0.6 * n if n else 0.0), land)
    settled = picks[WARMUP + 5 :]
    assert max(set(settled), key=settled.count) >= 5


def test_steep_cost_shrinks_the_budget():
    b = NodeBudget(7, prior=[0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15])
    land = [1, 0, 0, 0, 0, 0, 0]
    picks = _run(b, lambda n: 60.0 + (10.0 + 6.0 * n if n else 0.0), land, rounds=300)
    settled = picks[100:]
    assert max(set(settled), key=settled.count) <= 2


def test_room_limits_the_budget():
    b = NodeBudget(7)
    assert b.choose(3) == 3
    assert b.choose(0) == 0


def test_expected_tokens_is_prefix_sum():
    b = NodeBudget(3, prior=[0.5, 0.25, 0.1])
    assert abs(b.expected_tokens(2) - 1.75) < 1e-9
