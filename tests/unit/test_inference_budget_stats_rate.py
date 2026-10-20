"""get_stats() with a positive global rate limit must not deadlock."""

from __future__ import annotations

import threading

from yunshu_engine.inference_budget import InferenceBudgetManager


def test_get_stats_with_rate_limit_does_not_deadlock():
    mgr = InferenceBudgetManager(global_token_rate_limit=10)
    out: list[dict] = []
    t = threading.Thread(target=lambda: out.append(mgr.get_stats()), daemon=True)
    t.start()
    t.join(2.0)
    assert not t.is_alive(), "get_stats() deadlocked on the budget lock"
    assert out[0]["rate_limited"] is False
