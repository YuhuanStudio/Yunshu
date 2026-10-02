"""Request-local economics for copy proposals, with discrete verify widths.

This is a policy prototype: callers supply a curve measured on their actual
backend/context, rather than interpolate through a kernel-width transition.
Observations are complete round costs (not additive barrier components).
"""

from collections import deque
from collections.abc import Mapping
from math import isfinite


class CopyCosts:
    def __init__(self, row_ms: Mapping[int, float], model_ms: float):
        if not row_ms or any(
            r < 3 or not isfinite(t) or t <= 0 for r, t in row_ms.items()
        ):
            raise ValueError("copy costs need positive timings for widths >= 3")
        if not isfinite(model_ms) or model_ms <= 0:
            raise ValueError("model round cost must be positive")
        self.row_ms = dict(sorted(row_ms.items()))
        self.model_ms = model_ms
        self.model_samples: deque[tuple[int, float]] = deque(maxlen=32)
        self.copy_samples: dict[int, deque[tuple[int, int, float, bool]]] = {}
        self.choices = 0

    def observe_model(self, committed: int, elapsed_ms: float) -> None:
        if committed > 0 and isfinite(elapsed_ms) and elapsed_ms > 0:
            self.model_samples.append((committed, elapsed_ms))

    def observe_copy(
        self,
        proposed: int,
        accepted: int,
        elapsed_ms: float,
        *,
        confident: bool = False,
    ) -> None:
        if proposed < 2 or not 0 <= accepted <= proposed:
            raise ValueError("copy acceptance is outside the proposed window")
        if not isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        bucket = self.copy_samples.setdefault(proposed + 1, deque(maxlen=32))
        bucket.append((proposed, accepted, elapsed_ms, confident))

    def cost(self, rows: int) -> float:
        samples = self.copy_samples.get(rows)
        if samples and len(samples) >= 3:
            return sum(s[2] for s in samples) / len(samples)
        # Charge the next measured bucket as a conservative bound. Never
        # extrapolate a slope through an unmeasured width or tile transition.
        return next((t for r, t in self.row_ms.items() if r >= rows), float("inf"))

    def choose(self, drafts: int, model_tpr: float, *, confident: bool = False) -> int:
        """Draft count maximizing commits/ms; 0 means the model drafts.

        Unmeasured acceptance starts optimistic. A sample at depth D gives
        observations only through D; truncation cannot label unseen tails.
        Three model rounds are required before replacing the supplied prior.
        """
        if drafts < 2:
            return 0
        self.choices += 1
        if len(self.model_samples) >= 3:
            model_rate = sum(t for t, _ in self.model_samples) / sum(
                ms for _, ms in self.model_samples
            )
        else:
            model_rate = model_tpr / self.model_ms
        samples = [
            s
            for bucket in self.copy_samples.values()
            for s in bucket
            if s[3] == confident
        ]
        probabilities = []
        for depth in range(1, drafts + 1):
            observed = [a >= depth for p, a, _, _ in samples if p >= depth]
            probabilities.append(sum(observed) / len(observed) if observed else 1.0)
        best, rate = 0, model_rate
        # Price only measured buckets and the actual short/end window.
        widths = {min(drafts, r - 1) for r in self.row_ms}
        widths.add(drafts)
        for n in sorted(widths):
            if n < 2:
                continue
            candidate = (1 + sum(probabilities[:n])) / self.cost(n + 1)
            if candidate > rate * 1.01:
                best, rate = n, candidate
        # A different copy island must be able to recover after earlier misses.
        # Its occasional short probe is charged like every other copy round.
        return min(3, drafts) if best == 0 and self.choices % 16 == 0 else best


__all__ = ["CopyCosts"]
