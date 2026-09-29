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


def test_chain_budget_replaces_dflash_block_size(monkeypatch):
    """Cost-aware chain depth: wide when cycles are cheap, narrow when rows cost."""
    import random
    import types

    dflash_mod = __import__("mlx_vlm.speculative.dflash", fromlist=["x"])
    original = dflash_mod._dflash_next_block_size
    from yunshu_engine import spec_schedule

    monkeypatch.setattr(dflash_mod, "_dflash_next_block_size", original)
    spec_schedule.install_chain_budget()
    choose = dflash_mod._dflash_next_block_size
    try:
        for row_ms, expect_wide in ((0.6, True), (9.0, False)):
            clock = [0.0]
            monkeypatch.setattr(
                spec_schedule,
                "time",
                types.SimpleNamespace(perf_counter=lambda: clock[0]),
                raising=False,
            )
            import time as real_time

            monkeypatch.setattr(real_time, "perf_counter", lambda: clock[0])
            model = types.SimpleNamespace(accept_lens=[], draft_lens=[])
            rng = random.Random(0)
            sizes = []
            for r in range(120):
                bs = choose(model, 8, 1000)
                sizes.append(bs)
                accepted = 0
                while (
                    accepted < bs - 1
                    and rng.random() < (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)[accepted]
                ):
                    accepted += 1
                model.accept_lens.append(accepted)
                model.draft_lens.append(bs - 1)
                clock[0] += (55.0 + row_ms * (bs - 1)) / 1e3
            settled = sizes[60:]
            wide = sum(1 for s in settled if s >= 6) / len(settled)
            assert (wide > 0.5) == expect_wide, (row_ms, sizes[-12:])
    finally:
        dflash_mod._dflash_next_block_size = original
