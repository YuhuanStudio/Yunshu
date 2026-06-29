"""Tests for AdaptiveSpecController — dynamic draft length for spec decode."""

import pytest

from yunshu_engine.adaptive_spec import AdaptiveSpecConfig, AdaptiveSpecController

# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_defaults(self):
        ctrl = AdaptiveSpecController()
        assert ctrl.get_draft_length() == 4  # initial_draft_length
        stats = ctrl.get_stats()
        assert stats["current_k"] == 4
        assert stats["min_k"] == 1
        assert stats["max_k"] == 8
        assert stats["ema_rate"] is None
        assert stats["total_steps"] == 0
        assert stats["adjustments"] == 0

    def test_custom_config(self):
        ctrl = AdaptiveSpecController(
            min_draft_length=2,
            max_draft_length=16,
            initial_draft_length=10,
            ema_alpha=0.5,
            increase_threshold=0.9,
            decrease_threshold=0.3,
            increase_step=2,
            decrease_step=3,
            cooldown_steps=10,
        )
        assert ctrl.get_draft_length() == 10
        stats = ctrl.get_stats()
        assert stats["current_k"] == 10
        assert stats["min_k"] == 2
        assert stats["max_k"] == 16
        assert stats["ema_alpha"] == 0.5
        assert stats["increase_threshold"] == 0.9
        assert stats["decrease_threshold"] == 0.3

    def test_initial_k_clamped_to_min(self):
        ctrl = AdaptiveSpecController(min_draft_length=3, initial_draft_length=1)
        assert ctrl.get_draft_length() == 3

    def test_initial_k_clamped_to_max(self):
        ctrl = AdaptiveSpecController(max_draft_length=4, initial_draft_length=10)
        assert ctrl.get_draft_length() == 4

    def test_validation_min_draft_length(self):
        with pytest.raises(ValueError, match="min_draft_length must be >= 1"):
            AdaptiveSpecController(min_draft_length=0)

    def test_validation_max_lt_min(self):
        with pytest.raises(ValueError, match="max_draft_length"):
            AdaptiveSpecController(min_draft_length=5, max_draft_length=3)

    def test_validation_ema_alpha_zero(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveSpecController(ema_alpha=0.0)

    def test_validation_ema_alpha_negative(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveSpecController(ema_alpha=-0.1)

    def test_validation_decrease_ge_increase(self):
        with pytest.raises(ValueError, match="decrease_threshold"):
            AdaptiveSpecController(decrease_threshold=0.8, increase_threshold=0.8)

    def test_validation_negative_cooldown(self):
        with pytest.raises(ValueError, match="cooldown_steps"):
            AdaptiveSpecController(cooldown_steps=-1)


# ---------------------------------------------------------------------------
# Acceptance rate above threshold -> increase K
# ---------------------------------------------------------------------------


class TestIncrease:
    def test_high_acceptance_increases_k(self):
        """When acceptance rate is consistently high, K should increase."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            increase_threshold=0.8,
            cooldown_steps=3,
            increase_step=2,
        )
        # Feed 3 steps (cooldown) with high acceptance (all accepted)
        for _ in range(3):
            ctrl.record_step(4, 4)  # 100% acceptance
        # After cooldown, K should increase by 2
        assert ctrl.get_draft_length() == 6

    def test_multiple_increases_cap_at_max(self):
        """K should cap at max_draft_length."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            max_draft_length=6,
            increase_threshold=0.7,
            cooldown_steps=2,
            increase_step=1,
        )
        # Multiple rounds of high acceptance
        for _ in range(2):
            ctrl.record_step(4, 4)
        assert ctrl.get_draft_length() == 5
        for _ in range(2):
            ctrl.record_step(5, 5)
        assert ctrl.get_draft_length() == 6  # capped
        # Another round should stay at max
        for _ in range(2):
            ctrl.record_step(6, 6)
        assert ctrl.get_draft_length() == 6  # still capped


# ---------------------------------------------------------------------------
# Acceptance rate below threshold -> decrease K
# ---------------------------------------------------------------------------


class TestDecrease:
    def test_low_acceptance_decreases_k(self):
        """When acceptance rate is consistently low, K should decrease."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            decrease_threshold=0.5,
            cooldown_steps=3,
            decrease_step=2,
        )
        # Feed 3 steps with low acceptance (0 accepted)
        for _ in range(3):
            ctrl.record_step(4, 0)
        assert ctrl.get_draft_length() == 2

    def test_multiple_decreases_floor_at_min(self):
        """K should floor at min_draft_length."""
        ctrl = AdaptiveSpecController(
            min_draft_length=2,
            initial_draft_length=4,
            decrease_threshold=0.5,
            cooldown_steps=2,
            decrease_step=1,
        )
        for _ in range(2):
            ctrl.record_step(4, 0)
        assert ctrl.get_draft_length() == 3
        for _ in range(2):
            ctrl.record_step(3, 0)
        assert ctrl.get_draft_length() == 2
        # Another round should stay at min
        for _ in range(2):
            ctrl.record_step(2, 0)
        assert ctrl.get_draft_length() == 2


# ---------------------------------------------------------------------------
# K stays within [min, max] bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_k_never_exceeds_max(self):
        ctrl = AdaptiveSpecController(
            min_draft_length=1,
            max_draft_length=5,
            initial_draft_length=5,
            increase_threshold=0.4,
            decrease_threshold=0.2,
            cooldown_steps=1,
            increase_step=3,
        )
        # Even with 100% acceptance, K stays at max
        ctrl.record_step(5, 5)
        assert ctrl.get_draft_length() == 5

    def test_k_never_goes_below_min(self):
        ctrl = AdaptiveSpecController(
            min_draft_length=3,
            max_draft_length=8,
            initial_draft_length=3,
            decrease_threshold=0.6,
            cooldown_steps=1,
            decrease_step=5,
        )
        # Even with 0% acceptance, K stays at min
        ctrl.record_step(3, 0)
        assert ctrl.get_draft_length() == 3


# ---------------------------------------------------------------------------
# Cooldown: no adjustment before cooldown_steps
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_no_adjustment_before_cooldown(self):
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            increase_threshold=0.4,
            decrease_threshold=0.2,
            cooldown_steps=5,
            increase_step=1,
        )
        # Feed 4 high-acceptance steps (1 less than cooldown)
        for _ in range(4):
            ctrl.record_step(4, 4)
        assert ctrl.get_draft_length() == 4  # No change yet
        stats = ctrl.get_stats()
        assert stats["adjustments"] == 0
        assert stats["steps_since_last_adjust"] == 4

        # 5th step triggers adjustment
        ctrl.record_step(4, 4)
        assert ctrl.get_draft_length() == 5
        assert ctrl.get_stats()["adjustments"] == 1

    def test_cooldown_resets_after_adjustment(self):
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            increase_threshold=0.4,
            decrease_threshold=0.2,
            cooldown_steps=2,
            increase_step=1,
        )
        # 2 steps -> adjustment
        ctrl.record_step(4, 4)
        ctrl.record_step(4, 4)
        assert ctrl.get_draft_length() == 5
        assert ctrl.get_stats()["steps_since_last_adjust"] == 0

        # Next step starts new cooldown (should NOT adjust)
        ctrl.record_step(5, 5)
        assert ctrl.get_draft_length() == 5
        assert ctrl.get_stats()["steps_since_last_adjust"] == 1


# ---------------------------------------------------------------------------
# EMA smoothing: sudden change doesn't cause immediate K change
# ---------------------------------------------------------------------------


class TestEMASmoothing:
    def test_sudden_change_does_not_cause_immediate_k_change(self):
        """After many high-acceptance steps, one low step shouldn't drop K."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            ema_alpha=0.1,  # Very slow smoothing
            increase_threshold=0.8,
            decrease_threshold=0.5,
            cooldown_steps=1,
        )
        # Build up high EMA with 10 steps of 100% acceptance
        for _ in range(10):
            ctrl.record_step(4, 4)

        # EMA should be close to 1.0
        ema_before = ctrl.get_stats()["ema_rate"]
        assert ema_before > 0.9

        # One bad step (0% acceptance)
        ctrl.record_step(4, 0)

        ema_after = ctrl.get_stats()["ema_rate"]
        # EMA should still be above decrease_threshold
        assert ema_after > 0.5, f"EMA dropped too fast: {ema_after}"
        # K should not decrease
        assert ctrl.get_draft_length() == 4 or ctrl.get_draft_length() >= 4

    def test_ema_converges_to_current_rate(self):
        """EMA should eventually converge to the new acceptance rate."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            ema_alpha=0.5,
            increase_threshold=0.99,
            decrease_threshold=0.01,
            cooldown_steps=0,
        )
        # 20 steps of 50% acceptance
        for _ in range(20):
            ctrl.record_step(4, 2)

        ema = ctrl.get_stats()["ema_rate"]
        assert abs(ema - 0.5) < 0.05, f"EMA didn't converge: {ema}"


# ---------------------------------------------------------------------------
# get_stats() correctness
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_after_recording(self):
        ctrl = AdaptiveSpecController(
            initial_draft_length=3,
            cooldown_steps=1,
            increase_threshold=0.9,
        )
        ctrl.record_step(3, 2)
        ctrl.record_step(3, 3)

        stats = ctrl.get_stats()
        assert stats["total_steps"] == 2
        assert stats["total_draft_tokens"] == 6
        assert stats["total_accepted_tokens"] == 5
        assert stats["overall_acceptance_rate"] == round(5 / 6, 4)
        assert stats["current_k"] == 3
        assert stats["ema_rate"] is not None
        assert stats["enabled"] is True

    def test_stats_empty(self):
        ctrl = AdaptiveSpecController()
        stats = ctrl.get_stats()
        assert stats["total_steps"] == 0
        assert stats["total_draft_tokens"] == 0
        assert stats["total_accepted_tokens"] == 0
        assert stats["overall_acceptance_rate"] == 0.0
        assert stats["ema_rate"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_accepted(self):
        """100% acceptance rate should increase K to max."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=2,
            max_draft_length=8,
            increase_threshold=0.7,
            cooldown_steps=1,
            increase_step=2,
        )
        for _ in range(4):
            ctrl.record_step(ctrl.get_draft_length(), ctrl.get_draft_length())

        assert ctrl.get_draft_length() == 8
        stats = ctrl.get_stats()
        assert stats["overall_acceptance_rate"] == 1.0

    def test_all_rejected(self):
        """0% acceptance rate should decrease K to min."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=6,
            min_draft_length=1,
            max_draft_length=8,
            decrease_threshold=0.5,
            cooldown_steps=1,
            decrease_step=1,
        )
        for _ in range(6):
            ctrl.record_step(ctrl.get_draft_length(), 0)

        assert ctrl.get_draft_length() == 1

    def test_single_token_drafts(self):
        """Single-token drafts (K=1) should still work."""
        ctrl = AdaptiveSpecController(
            min_draft_length=1,
            max_draft_length=2,
            initial_draft_length=1,
            cooldown_steps=1,
            increase_threshold=0.8,
        )
        # K=1, accepted=1 -> 100% rate
        ctrl.record_step(1, 1)
        assert ctrl.get_draft_length() == 2  # Increases to max

    def test_zero_draft_length(self):
        """Zero-length draft should be handled gracefully."""
        ctrl = AdaptiveSpecController(initial_draft_length=4)
        ctrl.record_step(0, 0)
        assert ctrl.get_stats()["ema_rate"] == 0.0
        assert ctrl.get_draft_length() == 4  # No crash

    def test_accepted_exceeds_draft(self):
        """Accepted tokens are clamped to draft_length to prevent rate > 1.0."""
        ctrl = AdaptiveSpecController(initial_draft_length=4)
        ctrl.record_step(4, 6)  # Accepted clamped to 4
        assert ctrl.get_stats()["ema_rate"] == 1.0  # rate = min(6,4)/4 = 1.0


# ---------------------------------------------------------------------------
# from_env() tests
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("YUNSHU_ADAPTIVE_SPEC", raising=False)
        result = AdaptiveSpecController.from_env()
        assert result is None

    def test_enabled_with_1(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC", "1")
        result = AdaptiveSpecController.from_env()
        assert result is not None
        assert result.get_stats()["enabled"] is True

    def test_enabled_with_true(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC", "true")
        result = AdaptiveSpecController.from_env()
        assert result is not None

    def test_enabled_with_yes(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC", "yes")
        result = AdaptiveSpecController.from_env()
        assert result is not None

    def test_disabled_with_0(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC", "0")
        result = AdaptiveSpecController.from_env()
        assert result is None

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC", "1")
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC_MIN_K", "2")
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC_MAX_K", "12")
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC_INITIAL_K", "6")
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC_EMA_ALPHA", "0.5")
        monkeypatch.setenv("YUNSHU_ADAPTIVE_SPEC_COOLDOWN", "10")
        result = AdaptiveSpecController.from_env()
        assert result is not None
        stats = result.get_stats()
        assert stats["min_k"] == 2
        assert stats["max_k"] == 12
        assert stats["current_k"] == 6
        assert stats["ema_alpha"] == 0.5
        assert stats["steps_since_last_adjust"] == 0


# ---------------------------------------------------------------------------
# Hysteresis band test
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_no_change_in_hysteresis_band(self):
        """Rate between decrease and increase thresholds -> K stays same."""
        ctrl = AdaptiveSpecController(
            initial_draft_length=4,
            increase_threshold=0.8,
            decrease_threshold=0.5,
            cooldown_steps=1,
        )
        # Feed 65% acceptance rate (between 0.5 and 0.8)
        for _ in range(10):
            ctrl.record_step(10, 6)  # 60% rate -> in the band

        # K should not have changed
        assert ctrl.get_draft_length() == 4
        assert ctrl.get_stats()["adjustments"] == 0


# ---------------------------------------------------------------------------
# AdaptiveSpecConfig dataclass test
# ---------------------------------------------------------------------------


class TestAdaptiveSpecConfig:
    def test_defaults(self):
        config = AdaptiveSpecConfig()
        assert config.min_draft_length == 1
        assert config.max_draft_length == 8
        assert config.initial_draft_length == 4
        assert config.ema_alpha == 0.3
        assert config.increase_threshold == 0.8
        assert config.decrease_threshold == 0.5
        assert config.increase_step == 1
        assert config.decrease_step == 1
        assert config.cooldown_steps == 5
