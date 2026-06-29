"""Tests for inference_budget.py — token/time budget enforcement."""

import time
from unittest.mock import patch

from yunshu_engine.inference_budget import (
    InferenceBudget,
    InferenceBudgetManager,
)


class TestInferenceBudget:
    def test_token_budget(self):
        b = InferenceBudget(request_id="r1", max_tokens=10)
        assert not b.is_token_exhausted
        assert b.tokens_remaining == 10
        b.consume_tokens(5)
        assert b.tokens_remaining == 5
        b.consume_tokens(5)
        assert b.is_token_exhausted

    def test_time_budget(self):
        # the wall budget is an INACTIVITY timeout measured from the
        # last generated token, so the clock only runs once generation starts.
        b = InferenceBudget(request_id="r1", max_wall_time_ms=0.001)
        b.consume_tokens(1)  # start generating
        time.sleep(0.01)
        assert b.is_time_exhausted  # no new token for 10ms >> 0.001ms

    def test_time_budget_queue_not_counted(self):
        # A request still queued / prefilling (no token yet) must NOT be time
        # exhausted no matter how long it waits (the N>=96 collapse fix).
        b = InferenceBudget(request_id="r1", max_wall_time_ms=0.001)
        time.sleep(0.01)
        assert not b.is_time_exhausted
        assert b.exhaustion_reason is None

    def test_time_budget_progress_resets(self):
        # Steady (even slow) progress keeps a request alive — each token resets
        # the inactivity clock.
        b = InferenceBudget(request_id="r1", max_wall_time_ms=2000.0)
        for _ in range(5):
            b.consume_tokens(1)
            time.sleep(0.01)  # 10ms << 2000ms budget (generous for slow CI)
            assert not b.is_time_exhausted

    def test_time_budget_not_exhausted(self):
        b = InferenceBudget(request_id="r1", max_wall_time_ms=30000)
        assert not b.is_time_exhausted
        assert b.wall_time_remaining_ms > 0

    def test_cost_budget(self):
        b = InferenceBudget(
            request_id="r1",
            prompt_tokens=100,
            cost_per_1k_tokens=0.01,
            max_cost=0.001,
        )
        b.consume_tokens(50)
        assert b.estimated_cost == 150 / 1000 * 0.01
        # Cost: 150 * 0.01 / 1000 = 0.0015 > max_cost=0.001
        assert b.is_cost_exhausted

    def test_no_cost_budget(self):
        b = InferenceBudget(request_id="r1", max_cost=None)
        assert not b.is_cost_exhausted

    def test_thinking_budget(self):
        b = InferenceBudget(request_id="r1", thinking_budget=5)
        b.consume_tokens(3, is_thinking=True)
        assert b.thinking_tokens_used == 3
        assert not b.is_thinking_exhausted
        b.consume_tokens(3, is_thinking=True)
        assert b.is_thinking_exhausted

    def test_no_thinking_budget(self):
        b = InferenceBudget(request_id="r1", thinking_budget=None)
        b.consume_tokens(100, is_thinking=True)
        assert not b.is_thinking_exhausted

    def test_exhaustion_reason_tokens(self):
        b = InferenceBudget(request_id="r1", max_tokens=1)
        b.consume_tokens(1)
        assert b.exhaustion_reason == "max_tokens_reached"

    def test_exhaustion_reason_time(self):
        b = InferenceBudget(request_id="r1", max_wall_time_ms=0.001)
        b.consume_tokens(1)  # start the inactivity clock
        time.sleep(0.01)
        assert b.exhaustion_reason == "wall_time_exceeded"

    def test_exhaustion_reason_none(self):
        b = InferenceBudget(request_id="r1", max_tokens=1000)
        assert b.exhaustion_reason is None

    def test_is_exhausted(self):
        b = InferenceBudget(request_id="r1", max_tokens=1)
        b.consume_tokens(1)
        assert b.is_exhausted

    def test_elapsed_ms(self):
        # elapsed_ms is ACTIVE generation time (from the first token), so it is
        # 0 until generation starts, then accrues.
        b = InferenceBudget(request_id="r1")
        assert b.elapsed_ms == 0.0  # not started
        b.consume_tokens(1)
        time.sleep(0.01)
        assert b.elapsed_ms > 5

    def test_default_values(self):
        b = InferenceBudget(request_id="r1")
        assert b.max_tokens == 512
        assert b.tokens_used == 0
        assert b.priority == 0


class TestInferenceBudgetManager:
    def test_register_budget(self):
        mgr = InferenceBudgetManager()
        budget = mgr.register("r1", max_tokens=100)
        assert budget.max_tokens == 100
        assert mgr.active_count == 1

    def test_consume_tokens(self):
        mgr = InferenceBudgetManager()
        mgr.register("r1", max_tokens=10)
        result = mgr.consume("r1", 5)
        assert result is None
        result = mgr.consume("r1", 5)
        assert result == "max_tokens_reached"

    def test_consume_thinking(self):
        mgr = InferenceBudgetManager()
        mgr.register("r1", max_tokens=100, thinking_budget=10)
        result = mgr.consume("r1", 5, is_thinking=True)
        assert result is None  # 5 < 10
        result = mgr.consume("r1", 5, is_thinking=True)
        assert result == "thinking_budget_reached"  # 10 >= 10

    def test_consume_nonexistent(self):
        mgr = InferenceBudgetManager()
        result = mgr.consume("nonexistent", 1)
        assert result == "budget_not_found"

    def test_check_budgets(self):
        mgr = InferenceBudgetManager()
        mgr.register("r1", max_tokens=1)
        mgr.register("r2", max_tokens=100)
        mgr.consume("r1", 1)
        exhausted = mgr.check_budgets()
        assert len(exhausted) == 1
        assert exhausted[0][0] == "r1"
        assert exhausted[0][1] == "max_tokens_reached"

    def test_remove_budget(self):
        mgr = InferenceBudgetManager()
        mgr.register("r1")
        budget = mgr.remove("r1")
        assert budget is not None
        assert mgr.active_count == 0
        assert mgr._completed == 1

    def test_remove_nonexistent(self):
        mgr = InferenceBudgetManager()
        assert mgr.remove("nonexistent") is None

    def test_get_budget(self):
        mgr = InferenceBudgetManager()
        mgr.register("r1", max_tokens=42)
        budget = mgr.get_budget("r1")
        assert budget.max_tokens == 42

    def test_default_max_tokens(self):
        mgr = InferenceBudgetManager(default_max_tokens=256)
        budget = mgr.register("r1")
        assert budget.max_tokens == 256

    def test_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "YUNSHU_DEFAULT_MAX_TOKENS": "1024",
                "YUNSHU_MAX_WALL_TIME_MS": "60000.0",
            },
        ):
            mgr = InferenceBudgetManager.from_env()
            budget = mgr.register("r1")
            assert budget.max_tokens == 1024
            assert budget.max_wall_time_ms == 60000.0

    def test_stats(self):
        mgr = InferenceBudgetManager()
        mgr.register("r1", max_tokens=10)
        mgr.consume("r1", 10)
        mgr.consume("r1", 1)  # triggers exhaustion
        mgr.remove("r1")
        stats = mgr.get_stats()
        assert stats["completed"] == 1
        assert stats["total_tokens_consumed"] == 11
        assert "max_tokens_reached" in stats["exhaustion_breakdown"]

    def test_rate_limiting(self):
        mgr = InferenceBudgetManager(global_token_rate_limit=10)
        mgr.register("r1", max_tokens=100)
        for _ in range(10):
            mgr.consume("r1", 1)
        assert mgr.is_rate_limited()

    def test_no_rate_limiting(self):
        mgr = InferenceBudgetManager(global_token_rate_limit=0)
        mgr.register("r1", max_tokens=100)
        mgr.consume("r1", 50)
        assert not mgr.is_rate_limited()
