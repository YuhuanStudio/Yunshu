"""Cost-aware adaptive MTP draft depth for mlx-vlm speculative decoding.

Upstream's Qwen3.5 drafter always uses the requested block size. The best block
differs by content: on Qwen3.8-27B with the exact verify kernels, code peaks at
block 4 (~67 tok/s) while prose peaks at block 3 (~53 tok/s) and block 5 costs
~25% on both (docs/research/runs/2026-09-28-matrix/). This controller picks the
block each round from what it has measured in the running request:

- per-position acceptance ``p[k]`` = P(draft k accepted | drafts < k accepted),
  an EMA updated from the accepted count of every round;
- per-block cycle time (wall time between successive rounds), an EMA kept
  across requests because it is a property of the model and machine;
- expected tokens per round ``1 + sum_k prod_{j<=k} p[j]`` divided by the
  cycle time; every ``EXPLORE_EVERY`` rounds a neighbouring block is tried.

Acceptance resets for each new request (content changes, see the oMLX branch
that re-measures per request). Only the block size changes, never which tokens
are accepted, so greedy output stays token-identical to AR.
"""

from __future__ import annotations

import os
import time

EMA = 0.25
EXPLORE_EVERY = 16
PRIOR_P = 0.8


class DepthController:
    def __init__(self, min_block: int = 2, max_block: int = 6, start: int = 4):
        self.min_block = min_block
        self.max_block = max_block
        self.start = max(min_block, min(start, max_block))
        self.cost: dict[int, float] = {}
        self.picks: dict[int, int] = {}
        self._reset_request()

    def _reset_request(self) -> None:
        self.p = [PRIOR_P] * (self.max_block + 1)
        self.seen = 0
        self.rounds = 0
        self.last_block: int | None = None
        self.last_time: float | None = None

    def _cost_of(self, block: int) -> float | None:
        if block in self.cost:
            return self.cost[block]
        known = sorted(self.cost.items())
        if not known:
            return None
        if len(known) == 1:
            (b0, c0) = known[0]
            return c0 * (1.0 + 0.12 * (block - b0))  # verify rows are cheap-ish
        (b0, c0), (b1, c1) = known[0], known[-1]
        slope = (c1 - c0) / (b1 - b0)
        return max(1e-4, c0 + slope * (block - b0))

    def _expected_tokens(self, block: int) -> float:
        total, run = 1.0, 1.0
        for k in range(1, block):
            run *= self.p[k]
            total += run
        return total

    def observe(self, accept_lens: list[int], now: float) -> None:
        if len(accept_lens) < self.seen:  # drafter reset -> new request
            self._reset_request()
        if self.last_block is not None and len(accept_lens) > self.seen:
            accepted = accept_lens[-1]
            for k in range(1, self.last_block):
                if k <= accepted:
                    self.p[k] += EMA * (1.0 - self.p[k])
                else:
                    self.p[k] += EMA * (0.0 - self.p[k])
                    break
            if self.last_time is not None:
                dt = now - self.last_time
                old = self.cost.get(self.last_block)
                self.cost[self.last_block] = (
                    dt if old is None else old + EMA * (dt - old)
                )
            self.rounds += 1
        self.seen = len(accept_lens)

    def choose(self, cap: int) -> int:
        hi = min(self.max_block, cap)
        lo = min(self.min_block, hi)
        if self.rounds == 0:
            return max(lo, min(self.start, hi))
        best, best_rate = self.last_block or self.start, -1.0
        for block in range(lo, hi + 1):
            cost = self._cost_of(block)
            if cost is None:
                continue
            rate = self._expected_tokens(block) / cost
            if rate > best_rate:
                best, best_rate = block, rate
        if self.rounds % EXPLORE_EVERY == 0:
            neighbour = best + (1 if (self.rounds // EXPLORE_EVERY) % 2 else -1)
            if lo <= neighbour <= hi:
                best = neighbour
        return max(lo, min(best, hi))


_CONTROLLERS: dict[int, DepthController] = {}


def stats() -> dict:
    """Block-choice histogram and cycle costs per drafter (for probes/logs)."""
    return {
        key: {
            "picks": dict(c.picks),
            "cost_ms": {b: round(v * 1000, 1) for b, v in c.cost.items()},
        }
        for key, c in _CONTROLLERS.items()
    }


def install(max_block: int | None = None, start: int = 4) -> bool:
    """Route mlx-vlm's MTP block choice through a per-drafter controller."""
    import mlx_vlm.speculative.mtp as mtp

    if getattr(mtp._mtp_next_block_size, "_yunshu_adaptive", False):
        return True
    cap_default = int(max_block or os.environ.get("YUNSHU_MTP_MAX_BLOCK", "6"))
    original = mtp._mtp_next_block_size

    def next_block_size(draft_model, requested, configured, remaining):
        accept_lens = getattr(draft_model, "accept_lens", None)
        if accept_lens is None:
            return original(draft_model, requested, configured, remaining)
        ctl = _CONTROLLERS.get(id(draft_model))
        if ctl is None:
            ctl = _CONTROLLERS[id(draft_model)] = DepthController(
                max_block=cap_default, start=start
            )
        now = time.perf_counter()
        ctl.observe(accept_lens, now)
        block = ctl.choose(min(cap_default, remaining))
        ctl.last_block, ctl.last_time = block, now
        ctl.picks[block] = ctl.picks.get(block, 0) + 1
        return block

    next_block_size._yunshu_adaptive = True
    mtp._mtp_next_block_size = next_block_size
    return True
