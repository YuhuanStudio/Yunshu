# Upstream (inspired): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/dflash.py @ c8da6659
"""Cost-aware draft budget (chain depth or tree nodes).

A round verifies the first ``n`` drafts of a chain, or the first ``n`` nodes of a
tree ordered best first (every prefix is a valid tree). Draft ``i`` lands
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

EMA = 0.03  # floor on the weight of the newest round in the node-landing estimates
PRIOR_WEIGHT = (
    8.0  # pseudo-rounds the prior counts for; a rank's weight is 1 / (rounds + this)
)
SAMPLES = 8  # cycle-time samples kept per budget
PROBE = 6  # rounds spent alternating the full and half budget to read the row cost
REPROBE_EVERY = 64  # rounds between probes once settled
WARMUP = 3  # rounds at the full budget before any probing
MIN_GAIN = 1.03  # a smaller budget must beat the full one by this factor


class NodeBudget:
    def __init__(
        self,
        size: int,
        prior=None,
        plain_ms: float = 43.0,
        row_ms: float = 0.5,
        draft_ms: float = 10.0,
    ):
        self.size = int(size)
        prior = list(prior or [])
        # per-node landing probability: the prior, decaying for unlisted nodes
        self.p = [float(prior[i]) if i < len(prior) else 0.05 for i in range(self.size)]
        self.seen = [0] * self.size
        self.samples: dict[int, list[float]] = {}
        self._prior_cost = (plain_ms, row_ms, draft_ms)
        self.rounds = 0
        self.last = self.size
        self._half = max(1, self.size // 2)

    # -- cost model ------------------------------------------------------------------
    @staticmethod
    def _median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def _fit(self):
        """(anchor budget, its median cycle ms, ms per row) from what was measured."""
        known = sorted(n for n, v in self.samples.items() if n > 0 and len(v) >= 2)
        plain, row, draft = self._prior_cost
        if not known:
            return None
        lo, hi = known[0], known[-1]
        med_lo = self._median(self.samples[lo])
        if hi - lo >= 2:
            slope = (self._median(self.samples[hi]) - med_lo) / (hi - lo)
            return lo, med_lo, max(slope, 0.2)
        return lo, med_lo, row

    def cycle_ms(self, n: int) -> float:
        plain, row, draft = self._prior_cost
        fit = self._fit()
        if n == 0 and self.samples.get(0):
            return self._median(self.samples[0])
        if fit is None:
            return plain + (draft + row * n if n else 0.0)
        lo, med, slope = fit
        if n == 0:
            return max(1.0, med - draft - slope * lo)
        return med + slope * (n - lo)

    def smoothed(self) -> list[float]:
        """Landing probabilities made non-increasing in rank (pool adjacent violators).

        Nodes are ordered best first, so rank ``i`` cannot land more often than rank
        ``i - 1``. The raw per-rank EMAs are noisy and read 0.0 after a few misses on the
        rarely verified tail; pooling keeps the tail's real total, which is what decides
        whether a wider round pays when the row cost is nearly flat."""
        blocks: list[list[float]] = []  # [mean, count]
        for value in self.p:
            blocks.append([value, 1.0])
            while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
                v2, w2 = blocks.pop()
                v1, w1 = blocks.pop()
                blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
        out: list[float] = []
        for mean, count in blocks:
            out.extend([mean] * int(count))
        return out

    def expected_tokens(self, n: int) -> float:
        return 1.0 + sum(self.smoothed()[:n])

    def best(self) -> int:
        """The budget with the most expected tokens per ms, or the full one unless
        another is clearly better: measured cycle times are noisy."""

        def rate(n):
            return self.expected_tokens(n) / self.cycle_ms(n)

        top = max(range(self.size + 1), key=rate)
        return top if rate(top) > MIN_GAIN * rate(self.size) else self.size

    # -- per round -------------------------------------------------------------------
    def choose(self, room: int) -> int:
        """Nodes to verify this round (``room``: most tokens the request can still use)."""
        cap = min(self.size, max(0, room))
        r = self.rounds - WARMUP
        if r < 0:
            n = cap
        elif r < PROBE or (r % REPROBE_EVERY) < PROBE and r >= REPROBE_EVERY:
            n = cap if r % 2 == 0 else min(self._half, cap)  # read the row cost
        else:
            n = min(self.best(), cap)
        self.last = n
        return n

    def observe(
        self, n: int, landed_nodes, cycle_ms: float, *, first: bool = False
    ) -> None:
        """``landed_nodes``: indices (in topology order) of the drafted nodes on the accepted path."""
        self.rounds += 1
        landed = set(landed_nodes)
        for i in range(min(n, self.size)):
            weight = max(EMA, 1.0 / (self.seen[i] + PRIOR_WEIGHT))
            self.seen[i] += 1
            self.p[i] += weight * ((1.0 if i in landed else 0.0) - self.p[i])
        if first:
            return  # the first round carries the prefill-to-decode switch
        bucket = self.samples.setdefault(n, [])
        bucket.append(cycle_ms)
        del bucket[:-SAMPLES]


__all__ = ["NodeBudget"]


# -- chain depth ---------------------------------------------------------------------

# landing probability of the i-th draft of a chain before the round history says otherwise
CHAIN_PRIOR = [0.74, 0.49, 0.29, 0.19, 0.12, 0.09, 0.08]


def install_chain_budget() -> bool:
    """Choose upstream's DFlash chain depth by expected tokens per millisecond
    (``NodeBudget`` over depths) instead of its acceptance-rate heuristic, which
    collapses to two drafts on prose and keeps the block wide at long context
    where wide verifies cost more than they land. Idempotent."""
    import time

    from mlx_vlm.speculative import dflash as dflash_mod

    if getattr(dflash_mod._dflash_next_block_size, "_yunshu_budget", False):
        return True
    state: dict[int, list] = {}

    def next_block(
        draft_model, requested_block_total, remaining_budget, initial_block_size=None
    ):
        block_total = min(requested_block_total, remaining_budget)
        if block_total <= 2:
            return block_total
        now = time.perf_counter()
        rounds = len(getattr(draft_model, "accept_lens", []) or [])
        st = state.get(id(draft_model))
        if st is None or rounds == 0:
            st = state[id(draft_model)] = [
                NodeBudget(min(7, requested_block_total - 1), prior=CHAIN_PRIOR),
                None,
                None,
            ]
        budget, last_time, last_n = st
        if last_n is not None and rounds:
            accepted = int(draft_model.accept_lens[-1])
            budget.observe(
                last_n,
                range(min(accepted, last_n)),
                (now - last_time) * 1e3,
                first=rounds == 1,
            )
        n = max(1, min(budget.choose(block_total - 1), block_total - 1))
        st[1], st[2] = now, n
        return n + 1

    next_block._yunshu_budget = True
    dflash_mod._dflash_next_block_size = next_block
    return True
