from __future__ import annotations

"""Yunshu Adaptive Speculative Decode Controller — dynamic draft length (K).

Dynamically adjusts the
number of speculative draft tokens (K) based on acceptance rate feedback:

  - High acceptance rate (> increase_threshold): increase K (more aggressive)
  - Low acceptance rate (< decrease_threshold): decrease K (less wasted compute)
  - Exponential Moving Average (EMA) smooths per-step acceptance rates
  - Cooldown prevents oscillation between adjustments
  - Min/max bounds keep K in a safe range

Integration:
  - Enabled via YUNSHU_ADAPTIVE_SPEC=1 env var (requires N-gram spec active)
  - BatchedEngine creates AdaptiveSpecController when both are enabled
  - After each speculation step, record_step() is called with results
  - get_draft_length() returns the recommended K for the next step
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSpecConfig:
    """Configuration for the adaptive speculative decode controller.

    Defaults are tuned so spec is **never slower** than plain decode: it starts
    small, and on low acceptance backs off all the way to K=0 (no draft → plain
    single-token decode, zero spec overhead), periodically probing to recover if
    the output turns repetitive. Spec only ever costs extra when it's actually
    winning.
    """

    min_draft_length: int = 0  # 0 = back off to plain decode (no draft) entirely
    max_draft_length: int = 8
    initial_draft_length: int = 2  # pessimistic start — climbs only if accepted
    ema_alpha: float = 0.3
    increase_threshold: float = 0.8
    decrease_threshold: float = 0.5
    increase_step: int = 1
    decrease_step: int = 1
    cooldown_steps: int = 3
    probe_interval: int = 16  # while idle (K=0), probe every N steps to re-measure
    probe_length: int = 2  # draft length used for a probe


class AdaptiveSpecController:
    """Dynamic spec decode draft length controller.

    Inspired by SGLang's AdaptiveController. Adjusts the number of draft
    tokens (K) based on the acceptance rate of recent speculation steps.

    Uses exponential moving average (EMA) to smooth acceptance rate
    and cooldown-based hysteresis to prevent oscillation.
    """

    def __init__(
        self,
        min_draft_length: int = 0,
        max_draft_length: int = 8,
        initial_draft_length: int = 2,
        ema_alpha: float = 0.3,
        increase_threshold: float = 0.8,
        decrease_threshold: float = 0.5,
        increase_step: int = 1,
        decrease_step: int = 1,
        cooldown_steps: int = 3,
        probe_interval: int = 16,
        probe_length: int = 2,
    ) -> None:
        if min_draft_length < 0:
            raise ValueError(f"min_draft_length must be >= 0, got {min_draft_length}")
        if max_draft_length < max(min_draft_length, 1):
            raise ValueError(
                f"max_draft_length ({max_draft_length}) must be >= "
                f"max(min_draft_length, 1)"
            )
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if decrease_threshold >= increase_threshold:
            raise ValueError(
                f"decrease_threshold ({decrease_threshold}) must be < "
                f"increase_threshold ({increase_threshold})"
            )
        if cooldown_steps < 0:
            raise ValueError(f"cooldown_steps must be >= 0, got {cooldown_steps}")
        if probe_interval < 1:
            raise ValueError(f"probe_interval must be >= 1, got {probe_interval}")

        self._min_k = min_draft_length
        self._max_k = max_draft_length
        self._current_k = max(
            min_draft_length, min(initial_draft_length, max_draft_length)
        )
        self._ema_alpha = ema_alpha
        self._increase_threshold = increase_threshold
        self._decrease_threshold = decrease_threshold
        self._increase_step = increase_step
        self._decrease_step = decrease_step
        self._cooldown_steps = cooldown_steps
        self._probe_interval = probe_interval
        self._probe_length = min(probe_length, max_draft_length)

        # Internal tracking state
        self._ema_rate: float | None = None  # None until first observation
        self._steps_since_adjust: int = 0
        self._idle_streak: int = 0  # steps spent at K=0 since the last probe
        self._total_steps: int = 0
        self._total_draft: int = 0
        self._total_accepted: int = 0
        self._adjustments: int = 0

    def record_step(self, draft_length: int, accepted: int) -> None:
        """Record the result of one speculation step.

        Updates the EMA acceptance rate and potentially adjusts K.

        Args:
            draft_length: Number of draft tokens proposed.
            accepted: Number of draft tokens accepted.
        """
        self._total_steps += 1
        self._total_draft += draft_length
        # Clamp accepted to draft_length to prevent rate > 1.0
        accepted = min(accepted, draft_length)
        self._total_accepted += accepted

        # Compute current step acceptance rate (clamped to [0, 1])
        current_rate = accepted / draft_length if draft_length > 0 else 0.0

        # Update EMA
        if self._ema_rate is None:
            self._ema_rate = current_rate
        else:
            self._ema_rate = (
                self._ema_alpha * current_rate
                + (1.0 - self._ema_alpha) * self._ema_rate
            )

        # Idle recovery: K==0 means we're backed off to plain decode and this call
        # is a periodic PROBE. If the probe was well accepted, the output turned
        # predictable again — leave idle immediately (bypass cooldown). Otherwise
        # stay idle; get_draft_length keeps probing every probe_interval steps.
        if self._current_k == 0:
            if current_rate >= self._increase_threshold:
                self._current_k = max(1, self._increase_step)
                self._adjustments += 1
                self._steps_since_adjust = 0
            return

        # Active mode (K>=1): EMA-driven adjust with cooldown hysteresis. Can back
        # off all the way to 0 (min_k) → plain decode with zero spec overhead.
        self._steps_since_adjust += 1
        if self._steps_since_adjust < self._cooldown_steps:
            return

        if self._ema_rate > self._increase_threshold:
            new_k = min(self._current_k + self._increase_step, self._max_k)
        elif self._ema_rate < self._decrease_threshold:
            new_k = max(self._current_k - self._decrease_step, self._min_k)
        else:
            # Within the hysteresis band — no change
            return

        if new_k != self._current_k:
            self._current_k = new_k
            self._adjustments += 1
            self._steps_since_adjust = 0

    def get_draft_length(self) -> int:
        """Recommended draft length for the next step.

        Returns 0 when backed off to plain decode (low acceptance) — the caller
        skips the draft+verify and does a plain single-token step (zero spec
        overhead). While idle it returns a small probe length every
        ``probe_interval`` steps so acceptance can be re-measured and K recover.
        """
        if self._current_k > 0:
            self._idle_streak = 0
            return self._current_k
        self._idle_streak += 1
        if self._idle_streak >= self._probe_interval:
            self._idle_streak = 0
            return self._probe_length
        return 0

    def get_stats(self) -> dict:
        """Return controller statistics for monitoring."""
        overall_rate = (
            self._total_accepted / self._total_draft if self._total_draft > 0 else 0.0
        )
        return {
            "enabled": True,
            "current_k": self._current_k,
            "min_k": self._min_k,
            "max_k": self._max_k,
            "ema_rate": round(self._ema_rate, 4)
            if self._ema_rate is not None
            else None,
            "overall_acceptance_rate": round(overall_rate, 4),
            "total_steps": self._total_steps,
            "total_draft_tokens": self._total_draft,
            "total_accepted_tokens": self._total_accepted,
            "adjustments": self._adjustments,
            "steps_since_last_adjust": self._steps_since_adjust,
            "increase_threshold": self._increase_threshold,
            "decrease_threshold": self._decrease_threshold,
            "ema_alpha": self._ema_alpha,
        }

    @classmethod
    def from_env(cls) -> AdaptiveSpecController | None:
        """Create an AdaptiveSpecController from environment variables.

        Returns None if YUNSHU_ADAPTIVE_SPEC is not set to 1/true/yes.
        """
        val = os.environ.get("YUNSHU_ADAPTIVE_SPEC", "").strip().lower()
        if val not in ("1", "true", "yes"):
            return None

        config = AdaptiveSpecConfig(
            min_draft_length=int(os.environ.get("YUNSHU_ADAPTIVE_SPEC_MIN_K", "0")),
            max_draft_length=int(os.environ.get("YUNSHU_ADAPTIVE_SPEC_MAX_K", "8")),
            initial_draft_length=int(
                os.environ.get("YUNSHU_ADAPTIVE_SPEC_INITIAL_K", "2")
            ),
            ema_alpha=float(os.environ.get("YUNSHU_ADAPTIVE_SPEC_EMA_ALPHA", "0.3")),
            increase_threshold=float(
                os.environ.get("YUNSHU_ADAPTIVE_SPEC_INCREASE_THRESH", "0.8")
            ),
            decrease_threshold=float(
                os.environ.get("YUNSHU_ADAPTIVE_SPEC_DECREASE_THRESH", "0.5")
            ),
            increase_step=int(
                os.environ.get("YUNSHU_ADAPTIVE_SPEC_INCREASE_STEP", "1")
            ),
            decrease_step=int(
                os.environ.get("YUNSHU_ADAPTIVE_SPEC_DECREASE_STEP", "1")
            ),
            cooldown_steps=int(os.environ.get("YUNSHU_ADAPTIVE_SPEC_COOLDOWN", "3")),
            probe_interval=int(
                os.environ.get("YUNSHU_ADAPTIVE_SPEC_PROBE_INTERVAL", "16")
            ),
            probe_length=int(os.environ.get("YUNSHU_ADAPTIVE_SPEC_PROBE_K", "2")),
        )

        controller = cls(
            min_draft_length=config.min_draft_length,
            max_draft_length=config.max_draft_length,
            initial_draft_length=config.initial_draft_length,
            ema_alpha=config.ema_alpha,
            increase_threshold=config.increase_threshold,
            decrease_threshold=config.decrease_threshold,
            increase_step=config.increase_step,
            decrease_step=config.decrease_step,
            cooldown_steps=config.cooldown_steps,
            probe_interval=config.probe_interval,
            probe_length=config.probe_length,
        )

        logger.info(
            f"Adaptive spec controller initialized: "
            f"k=[{config.min_draft_length},{config.max_draft_length}], "
            f"initial_k={config.initial_draft_length}, "
            f"ema_alpha={config.ema_alpha}, "
            f"thresholds=[{config.decrease_threshold},{config.increase_threshold}]"
        )
        return controller
