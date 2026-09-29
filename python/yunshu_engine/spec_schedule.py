"""Cost-aware draft budget for tree rounds.

A round verifies the first ``n`` nodes of a topology (``spec_topology`` orders
nodes most-valuable first, so every prefix is a valid tree). Node ``i`` lands
with probability ``p[i]`` (it is on the accepted path), so the round commits
``1 + sum(p[:n])`` tokens in expectation, and the budget that maximizes
expected tokens per millisecond is

    argmax_n (1 + sum(p[:n])) / cycle_ms(n)

Both tables are learned online: ``p`` from which nodes the walk kept, and
``cycle_ms`` from the measured wall time of rounds (drafting included for
``n > 0``, none for ``n = 0``, a plain step). Verify cost grows with rows and
with context (Qwen3.8-27B at 131K: an 8-row window costs more than it lands),
so the budget shrinks by itself where wide windows do not pay. Costs of budgets
not run lately are interpolated; a neighbouring budget is tried now and then so
the table follows the context length.
"""

from __future__ import annotations

EMA = 0.2  # weight of the newest round in the node-landing estimates
COST_EMA = 0.3  # weight of the newest round in the cycle-time estimates
EXPLORE_EVERY = 24  # rounds between neighbour probes once settled
WARMUP = 3  # rounds at the full budget before any probing


class NodeBudget:
    def __init__(
        self,
        size: int,
        prior=None,
        plain_ms: float = 43.0,
        row_ms: float = 0.8,
        draft_ms: float = 10.0,
    ):
        self.size = int(size)
        prior = list(prior or [])
        # per-node landing probability: the prior, decaying for unlisted nodes
        self.p = [float(prior[i]) if i < len(prior) else 0.05 for i in range(self.size)]
        self.cost: dict[int, float] = {}
        self._prior_cost = (plain_ms, row_ms, draft_ms)
        self.rounds = 0
        self._probe = 0
        self.last = self.size

    # -- cost model ------------------------------------------------------------------
    def cycle_ms(self, n: int) -> float:
        if n in self.cost:
            return self.cost[n]
        known = sorted(self.cost)
        plain, row, draft = self._prior_cost
        if not known:
            return plain + (draft + row * n if n else 0.0)
        above = [k for k in known if k > n]
        below = [k for k in known if k < n]
        if n == 0:
            # no drafting: the cheapest measured budget minus what drafting and its rows cost
            k = known[0]
            return max(1.0, self.cost[k] - draft - row * k)
        if below and above:
            a, b = below[-1], above[0]
            return self.cost[a] + (self.cost[b] - self.cost[a]) * (n - a) / (b - a)
        near = known[0] if above else known[-1]
        slope = row
        others = [k for k in known if k != near and k > 0]
        if near and others:
            far = min(others, key=lambda k: abs(k - near))
            slope = max(row * 0.25, (self.cost[near] - self.cost[far]) / (near - far))
        return self.cost[near] + slope * (n - near)

    def expected_tokens(self, n: int) -> float:
        return 1.0 + sum(self.p[:n])

    def best(self) -> int:
        return max(
            range(self.size + 1),
            key=lambda n: self.expected_tokens(n) / self.cycle_ms(n),
        )

    # -- per round -------------------------------------------------------------------
    def choose(self, room: int) -> int:
        """Nodes to verify this round (``room``: most tokens the request can still use)."""
        cap = min(self.size, max(0, room))
        if self.rounds < WARMUP:
            n = cap
        else:
            n = min(self.best(), cap)
            if (self.rounds - WARMUP) % EXPLORE_EVERY == EXPLORE_EVERY - 1:
                self._probe += 1
                cand = [m for m in (n - 2, n + 2, n // 2) if 0 <= m <= cap and m != n]
                if cand:
                    n = cand[self._probe % len(cand)]
        self.last = n
        return n

    def observe(
        self, n: int, landed_nodes, cycle_ms: float, *, first: bool = False
    ) -> None:
        """``landed_nodes``: indices (in topology order) of the drafted nodes on the accepted path."""
        self.rounds += 1
        landed = set(landed_nodes)
        for i in range(min(n, self.size)):
            self.p[i] += EMA * ((1.0 if i in landed else 0.0) - self.p[i])
        if first:
            return  # the first round carries the prefill-to-decode switch
        before = self.cost.get(n)
        self.cost[n] = (
            cycle_ms if before is None else before + COST_EMA * (cycle_ms - before)
        )


__all__ = ["NodeBudget"]
