"""CPU-only ordering for canonical prefill atoms and bounded decode quanta.

Estimates affect dispatch only. Cache lookup, token spans and sampling stay with
those requests' existing generators. An old waiter gets one atom, then ages
again from its last service; a stream of small arrivals cannot starve it.
"""

from __future__ import annotations

from dataclasses import dataclass

AGING_S = 10.0
DECODE_QUANTUM_S = 0.25
PREFILL_TOKEN_S = 0.00121
FIXED_S = 0.236


@dataclass(frozen=True)
class Work:
    arrived: float
    last_service: float
    uncached_tokens: int
    priority: int = 0
    restore_s: float = 0.0

    def key(self, now: float) -> tuple[float, ...]:
        waited = max(0.0, now - self.last_service)
        if waited >= AGING_S:
            return (0.0, self.last_service, self.arrived)
        service = FIXED_S + PREFILL_TOKEN_S * max(1, self.uncached_tokens)
        service += max(0.0, self.restore_s)
        # HRRN within the interactive class; captured auxiliary work yields
        # until idle or aged. Smaller remaining work breaks equal ratios.
        return (
            1.0,
            float(self.priority < 0),
            -(1 + waited / service),
            service,
            self.arrived,
        )
