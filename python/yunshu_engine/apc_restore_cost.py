"""Device-local admission estimates for automatic media checkpoint restores.

Learn from serialized requests, without a model/device lookup table. A one-token
revisit uses decode; a longer suffix launches prefill again. Charge that launch
the observed cold-prefill latency, conservatively, until suffix timings exist.
This affects reuse only: cold recomputation remains the correctness fallback.
"""

from dataclasses import dataclass


@dataclass
class MediaRestoreCost:
    cold_s: float = 0.0
    token_s: float = 0.0
    revisit_s: float = 0.0
    skips: int = 0

    def observe(self, tokens: int, cached: int, seconds: float) -> None:
        if tokens <= 0 or not 0 < seconds < 60:
            return
        if cached == 0:
            # A long prefill supplies the per-token slope, but its entire
            # latency is not a suffix launch cost.
            self.token_s = seconds / tokens
            if tokens <= 512:
                self.cold_s = seconds
        elif tokens - cached == 1:
            self.revisit_s = seconds

    def worth(self, checkpoint_bytes: int, prefix: int, suffix: int) -> bool:
        if suffix <= 1 or not self.cold_s or not self.token_s:
            return True
        # Read and write the checkpoint, with modest effective RAM bandwidth.
        # The observed revisit covers Python/MLX launch and cache merge work;
        # charging the byte term again is a conservative admission estimate.
        restore = self.revisit_s + 2 * checkpoint_bytes / 80_000_000_000
        cached = restore + self.cold_s + suffix * self.token_s
        cold = (prefix + suffix) * self.token_s
        return cached < cold
