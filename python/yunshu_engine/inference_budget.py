from __future__ import annotations
"""Inference budget manager — token/time budget enforcement per request.

Enforces per-request budgets for:
  - Token generation limit (max_tokens)
  - Time budget (max_wall_time_ms)
  - Cost budget (estimated tokens × cost_per_token)
  - Thinking budget (reasoning token cap)

Integrates with the engine loop to check budgets during generation
and stop requests that exceed their allocated resources.
"""

import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InferenceBudget:
    """Budget allocation for a single request."""
    request_id: str
    # Token budgets
    max_tokens: int = 512
    prompt_tokens: int = 0
    tokens_used: int = 0
    thinking_tokens_used: int = 0
    thinking_budget: int | None = None
    # Time budgets
    max_wall_time_ms: float = 30000.0  # 30s default
    created_at: float = field(default_factory=time.monotonic)
    # Cost tracking
    cost_per_1k_tokens: float = 0.0  # USD per 1K tokens
    max_cost: float | None = None
    # Priority for preemption (higher = more important)
    priority: int = 0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.created_at) * 1000

    @property
    def wall_time_remaining_ms(self) -> float:
        return max(0.0, self.max_wall_time_ms - self.elapsed_ms)

    @property
    def estimated_cost(self) -> float:
        total = self.prompt_tokens + self.tokens_used
        return total / 1000.0 * self.cost_per_1k_tokens

    @property
    def is_token_exhausted(self) -> bool:
        return self.tokens_used >= self.max_tokens

    @property
    def is_time_exhausted(self) -> bool:
        return self.elapsed_ms >= self.max_wall_time_ms

    @property
    def is_cost_exhausted(self) -> bool:
        if self.max_cost is None:
            return False
        return self.estimated_cost >= self.max_cost

    @property
    def is_thinking_exhausted(self) -> bool:
        if self.thinking_budget is None or self.thinking_budget <= 0:
            return False
        return self.thinking_tokens_used >= self.thinking_budget

    @property
    def is_exhausted(self) -> bool:
        return (
            self.is_token_exhausted
            or self.is_time_exhausted
            or self.is_cost_exhausted
        )

    @property
    def exhaustion_reason(self) -> str | None:
        if self.is_token_exhausted:
            return "max_tokens_reached"
        if self.is_time_exhausted:
            return "wall_time_exceeded"
        if self.is_cost_exhausted:
            return "cost_budget_exceeded"
        return None

    def consume_tokens(self, count: int, is_thinking: bool = False) -> None:
        self.tokens_used += count
        if is_thinking:
            self.thinking_tokens_used += count


class InferenceBudgetManager:
    """Manages inference budgets across all active requests.

    Integration:
    - EngineCore.add_request() → register_budget()
    - EngineCore._engine_loop() step → check_budgets()
    - BatchedEngine.generate() → register_budget()
    - After each token → consume() + check
    """

    def __init__(
        self,
        default_max_tokens: int = 512,
        default_max_wall_time_ms: float = 30000.0,
        global_token_rate_limit: int = 0,  # 0 = unlimited
    ) -> None:
        self._default_max_tokens = default_max_tokens
        self._default_wall_time = default_max_wall_time_ms
        self._global_rate_limit = global_token_rate_limit
        self._budgets: dict[str, InferenceBudget] = {}
        self._completed: int = 0
        self._exhausted: dict[str, int] = defaultdict(int)
        self._total_tokens_consumed: int = 0
        self._global_token_window: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> InferenceBudgetManager:
        return cls(
            default_max_tokens=int(os.environ.get("YUNSHU_DEFAULT_MAX_TOKENS", "512")),
            default_max_wall_time_ms=float(os.environ.get("YUNSHU_MAX_WALL_TIME_MS", "30000.0")),
            global_token_rate_limit=int(os.environ.get("YUNSHU_GLOBAL_TOKEN_RATE", "0")),
        )

    def register(
        self,
        request_id: str,
        max_tokens: int | None = None,
        prompt_tokens: int = 0,
        thinking_budget: int | None = None,
        max_wall_time_ms: float | None = None,
        cost_per_1k_tokens: float = 0.0,
        max_cost: float | None = None,
        priority: int = 0,
    ) -> InferenceBudget:
        """Register a budget for a new request."""
        budget = InferenceBudget(
            request_id=request_id,
            max_tokens=self._default_max_tokens if max_tokens is None else max_tokens,
            prompt_tokens=prompt_tokens,
            thinking_budget=thinking_budget,
            max_wall_time_ms=self._default_wall_time if max_wall_time_ms is None else max_wall_time_ms,
            cost_per_1k_tokens=cost_per_1k_tokens,
            max_cost=max_cost,
            priority=priority,
        )
        with self._lock:
            self._budgets[request_id] = budget
        return budget

    def consume(self, request_id: str, tokens: int = 1, is_thinking: bool = False) -> str | None:
        """Consume tokens for a request. Returns exhaustion reason or None."""
        with self._lock:
            budget = self._budgets.get(request_id)
            if budget is None:
                return "budget_not_found"

            budget.consume_tokens(tokens, is_thinking)
            self._total_tokens_consumed += tokens

            # Track global rate
            if self._global_rate_limit > 0:
                now = time.monotonic()
                self._global_token_window.append((now, tokens))
                self._prune_rate_window(now)

            if budget.is_exhausted:
                reason = budget.exhaustion_reason or "unknown"
                self._exhausted[reason] += 1
                return reason

            if is_thinking and budget.is_thinking_exhausted:
                return "thinking_budget_reached"

        return None

    def check_budgets(self) -> list[tuple[str, str]]:
        """Check all budgets and return (request_id, reason) for exhausted ones.

        Includes both full budget exhaustion (token/time/cost) and
        thinking budget exhaustion.
        """
        exhausted = []
        with self._lock:
            for rid, budget in list(self._budgets.items()):
                if budget.is_exhausted:
                    reason = budget.exhaustion_reason or "unknown"
                    exhausted.append((rid, reason))
                elif budget.is_thinking_exhausted:
                    exhausted.append((rid, "thinking_budget_reached"))
        return exhausted

    def remove(self, request_id: str) -> InferenceBudget | None:
        """Remove a budget (request finished or cleaned up)."""
        with self._lock:
            budget = self._budgets.pop(request_id, None)
            if budget:
                self._completed += 1
        return budget

    def get_budget(self, request_id: str) -> InferenceBudget | None:
        with self._lock:
            return self._budgets.get(request_id)

    def is_rate_limited(self) -> bool:
        """Check if global token rate limit is exceeded."""
        if self._global_rate_limit <= 0:
            return False
        with self._lock:
            now = time.monotonic()
            self._prune_rate_window(now)
            total = sum(n for _, n in self._global_token_window)
            return total >= self._global_rate_limit

    def _prune_rate_window(self, now: float, window_seconds: float = 60.0) -> None:
        cutoff = now - window_seconds
        self._global_token_window = [
            (t, n) for t, n in self._global_token_window if t > cutoff
        ]

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._budgets)

    def get_stats(self) -> dict:
        with self._lock:
            avg_tokens = (
                self._total_tokens_consumed / self._completed
                if self._completed > 0
                else 0.0
            )
            return {
                "active_budgets": len(self._budgets),
                "completed": self._completed,
                "total_tokens_consumed": self._total_tokens_consumed,
                "avg_tokens_per_request": round(avg_tokens, 1),
                "exhaustion_breakdown": dict(self._exhausted),
                "rate_limited": self.is_rate_limited(),
            }
