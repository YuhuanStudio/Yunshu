"""CPU-only ordering for canonical prefill atoms and bounded decode quanta.

Estimates affect dispatch only. Cache lookup, token spans and sampling stay with
those requests' existing generators. After one overtaking atom, an interactive
waiter retains FIFO protection until its prefill completes. A selected atom that
enters its final 2048-token window finishes that episode through first-token
delivery before FIFO protection resumes; new arrivals cannot repeat the bypass.
"""

from __future__ import annotations

from dataclasses import dataclass

AGING_S = 10.0
MAX_PREFILL_SKIPS = 1
DECODE_QUANTUM_S = 0.075
# Decode-first (vLLM chunked prefill, SGLang mixed batches): every running row
# advances alongside each prefill chunk. Separate generators cannot share a
# forward pass here, so a prefill atom of t seconds earns decode time
# DECODE_SHARE * t (parity), capped. The work is conserved: a short request
# that finishes inside the window costs the long prefill only its own decode
# time, but its tokens no longer trickle out one per 2048-token atom.
DECODE_SHARE = 1.0
# Only rows within their first DECODE_EARLY_TOKENS tokens (a tool call, a JSON
# answer, the start of a reply) get the generous share. Once every decoding row
# is past that, a long reply earns DECODE_TAIL_SHARE, which bounds how much a
# long cold prefill is slowed by someone else's long generation.
DECODE_EARLY_TOKENS = 128
DECODE_TAIL_SHARE = 0.1
DECODE_QUANTUM_MAX_S = 4.0
# A row that has just received its first token gets one burst of decode before
# the next prefill atom, so a short tool call or answer completes inside a
# single window instead of waiting for a second long atom (~2-3 s each).
DECODE_FIRST_BURST_S = 1.0
PRIMARY_HANDOFF_S = 0.10
# After a slice, a request still between templating and submit gets this long
# (at most PREPARE_WAIT_MAX times per episode, 48 ms in all) to reach the queue
# before the next prefill atom starts.
PREPARE_WAIT_S = 0.004
PREPARE_WAIT_MAX = 12
PREFILL_TOKEN_S = 0.00121
FIXED_S = 0.236


def decode_quantum(atom_s: float, early: bool = True) -> float:
    """Decode time owed to running rows after a prefill atom that took ``atom_s``."""
    share = DECODE_SHARE if early else DECODE_TAIL_SHARE
    return min(DECODE_QUANTUM_MAX_S, max(DECODE_QUANTUM_S, share * atom_s))


@dataclass(frozen=True)
class Work:
    arrived: float
    last_service: float
    uncached_tokens: int
    priority: int = 0
    restore_s: float = 0.0
    skips: int = 0

    def key(self, now: float) -> tuple[float, ...]:
        waited = max(0.0, now - self.last_service)
        if waited >= AGING_S or (
            self.priority >= 0 and self.skips >= MAX_PREFILL_SKIPS
        ):
            return (0.0, self.arrived)
        service = FIXED_S + PREFILL_TOKEN_S * max(1, self.uncached_tokens)
        service += max(0.0, self.restore_s)
        # HRRN until bounded overtaking promotes an interactive prefill to FIFO.
        # Captured auxiliary work yields
        # until idle or aged. Smaller remaining work breaks equal ratios.
        return (
            1.0,
            float(self.priority < 0),
            -(1 + waited / service),
            service,
            self.arrived,
        )
