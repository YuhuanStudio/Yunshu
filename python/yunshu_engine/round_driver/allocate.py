"""Cost-aware draft allocation across rows (TensorFold's rule, MIT).

Adapted from TensorFold ``engine/allocate.py`` (https://github.com/ashhart/
TensorFold, MIT; see THIRD_PARTY_NOTICES.md). Every row's pending token always
runs; each extra draft row is granted greedily to the row whose next draft is
most likely to land (``chain`` probabilities: the ``j``-th draft lands only if
every earlier one did), and the round keeps the allocation that maximizes
expected committed tokens per unit of step cost. With a flat cost curve
(bandwidth-bound: more rows cost ~nothing) rows draft deep; once wider forwards
cost more than they land, drafting stops by itself — no row-count thresholds.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence


def allocate(
    fixed_rows: int,
    probs: Sequence[Sequence[float]],
    cost_ms,
    max_rows: int,
    chain_ms: float = 0.0,
    padded: bool = False,
) -> list[int]:
    """Drafts per row. ``fixed_rows``: rows that run anyway (one pending
    token per decoding row); ``probs[s]``: row ``s``'s chance that its 1st,
    2nd, .. draft lands (at most its window); ``cost_ms(rows)``: step cost at
    ``rows`` packed rows; ``chain_ms``: drafting cost per chain depth (the
    deepest row sets how many sequential head steps the drafts take).
    ``padded``: rows run right-padded to the deepest window, so a step costs
    ``fixed_rows * (1 + deepest)`` packed rows and ``max_rows`` bounds that."""
    counts = [0] * len(probs)
    rows = int(fixed_rows)
    expected = float(sum(1 for _ in probs))  # every row commits >= 1 token
    deepest = 0

    def rate(total: int) -> float:
        spent = float(cost_ms(total)) + chain_ms * deepest
        return expected / max(spent, 1e-6)

    best, best_counts = rate(rows), list(counts)
    heap = [(-p[0], s) for s, p in enumerate(probs) if p]
    heapq.heapify(heap)
    while heap and rows < max_rows:
        neg, s = heapq.heappop(heap)
        counts[s] += 1
        deepest = max(deepest, counts[s])
        rows = fixed_rows * (1 + deepest) if padded else rows + 1
        if rows > max_rows:
            break
        expected -= neg
        if counts[s] < len(probs[s]):
            heapq.heappush(heap, (-probs[s][counts[s]], s))
        now = rate(rows)
        if now > best:
            best, best_counts = now, list(counts)
    return best_counts


def chain(rates: Sequence[float], count: int) -> list[float]:
    """Chance the ``j``-th draft lands given per-depth acceptance ``rates``."""
    out, reach = [], 1.0
    for j in range(count):
        reach *= rates[j] if j < len(rates) else rates[-1]
        out.append(reach)
    return out


SLOPE_PRIOR = 0.03  # extra cost per row, as a fraction of a lone measured step


class CostCurve:
    """Step cost by packed rows: measured points, linear between them, the
    per-row slope of the last two points beyond the widest one (with a single
    point, ``SLOPE_PRIOR`` of it per row)."""

    def __init__(self, points: dict[int, float] | None = None):
        self.points: dict[int, float] = dict(points or {})

    def observe(self, rows: int, ms: float, weight: float = 0.2) -> None:
        old = self.points.get(rows)
        self.points[rows] = ms if old is None else old + weight * (ms - old)

    def __call__(self, rows: int) -> float:
        if not self.points:
            return float(rows)  # unknown: every row costs the same
        xs = sorted(self.points)
        if rows in self.points:
            return self.points[rows]
        below = [x for x in xs if x < rows]
        above = [x for x in xs if x > rows]
        if below and above:
            a, b = below[-1], above[0]
            fa, fb = self.points[a], self.points[b]
            return fa + (fb - fa) * (rows - a) / (b - a)
        if above:
            return self.points[above[0]]
        if len(xs) >= 2:
            a, b = xs[-2], xs[-1]
            slope = max((self.points[b] - self.points[a]) / (b - a), 0.0)
        else:
            # one point: rows beyond it cost a few percent each (decode is
            # bandwidth-bound; wider windows get measured, then interpolated)
            slope = self.points[xs[-1]] * SLOPE_PRIOR
        return self.points[xs[-1]] + slope * (rows - xs[-1])


__all__ = ["CostCurve", "allocate", "chain"]
