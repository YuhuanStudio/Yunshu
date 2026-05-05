"""Tests for ThinkingSegmentSubstore wiring in Scheduler.

Verifies that the scheduler correctly:
- Creates a ThinkingSegmentSubstore on init
- Exposes it via get_thinking_store()
- Includes its stats in get_stats()
- Initializes and clears _thinking_state on lifecycle events
"""

import pytest
from unittest.mock import MagicMock

from yunshu_engine.scheduler import Scheduler, SchedulerConfig
from yunshu_kv.thinking_segment import ThinkingSegmentSubstore, ThinkingSegmentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(**config_overrides) -> Scheduler:
    """Create a Scheduler with mock model/tokenizer (no real GPU needed)."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = [2]
    tokenizer.encode = MagicMock(return_value=[1])
    tokenizer.has_thinking = False
    config = SchedulerConfig(model_name="test-model", **config_overrides)
    return Scheduler(model, tokenizer, config)


# ===========================================================================
# 1. __init__ creates ThinkingSegmentSubstore
# ===========================================================================

class TestInitThinkingStore:

    def test_init_creates_thinking_store_instance(self):
        """Scheduler.__init__ must create a ThinkingSegmentSubstore."""
        sched = _make_scheduler()
        assert hasattr(sched, "_thinking_store")
        assert isinstance(sched._thinking_store, ThinkingSegmentSubstore)

    def test_init_creates_empty_thinking_state(self):
        """_thinking_state must be an empty dict after construction."""
        sched = _make_scheduler()
        assert hasattr(sched, "_thinking_state")
        assert sched._thinking_state == {}

    def test_thinking_store_uses_default_config(self):
        """The substore should be created with a default ThinkingSegmentConfig."""
        sched = _make_scheduler()
        assert isinstance(sched._thinking_store.config, ThinkingSegmentConfig)
        # Spot-check a default value
        assert sched._thinking_store.config.max_total_segments == 1000
        assert sched._thinking_store.config.max_segments_per_conversation == 10


# ===========================================================================
# 2. get_thinking_store() returns the substore
# ===========================================================================

class TestGetThinkingStore:

    def test_returns_thinking_store(self):
        """get_thinking_store() must return the internal substore."""
        sched = _make_scheduler()
        store = sched.get_thinking_store()
        assert store is sched._thinking_store
        assert isinstance(store, ThinkingSegmentSubstore)

    def test_returns_same_instance_on_repeated_calls(self):
        """Repeated calls must return the same object (identity check)."""
        sched = _make_scheduler()
        first = sched.get_thinking_store()
        second = sched.get_thinking_store()
        assert first is second

    def test_returned_store_is_functional(self):
        """The returned store should respond to normal API calls."""
        sched = _make_scheduler()
        store = sched.get_thinking_store()
        # get_stats should return a valid dict
        stats = store.get_stats()
        assert isinstance(stats, dict)
        assert "total_segments" in stats
        assert stats["total_segments"] == 0


# ===========================================================================
# 3. get_stats() includes thinking_segment_store stats
# ===========================================================================

class TestGetStatsThinkingSegment:

    def test_stats_includes_thinking_segment_store_key(self):
        """get_stats() must contain a 'thinking_segment_store' key."""
        sched = _make_scheduler()
        stats = sched.get_stats()
        assert "thinking_segment_store" in stats

    def test_thinking_segment_store_stats_is_dict(self):
        """The 'thinking_segment_store' value must be a dict."""
        sched = _make_scheduler()
        stats = sched.get_stats()
        assert isinstance(stats["thinking_segment_store"], dict)

    def test_thinking_segment_store_stats_has_expected_fields(self):
        """The nested stats dict must include all ThinkingSegmentSubstore fields."""
        sched = _make_scheduler()
        stats = sched.get_stats()
        ts_stats = stats["thinking_segment_store"]
        for key in ("total_segments", "conversations_tracked", "stored",
                     "hits", "misses", "evictions", "tokens_saved", "hit_rate"):
            assert key in ts_stats, f"Missing key: {key}"

    def test_thinking_segment_store_stats_reflects_real_state(self):
        """If we store data in the substore, get_stats() must reflect it."""
        sched = _make_scheduler()
        store = sched.get_thinking_store()

        # Store a segment with enough tokens to exceed min_tokens_to_cache (32)
        thinking_tokens = list(range(64))
        context_tokens = list(range(10))
        result = store.store("conv-1", thinking_tokens, context_tokens, kv_data="fake-kv")
        assert result is not None

        stats = sched.get_stats()
        ts_stats = stats["thinking_segment_store"]
        assert ts_stats["stored"] == 1
        assert ts_stats["total_segments"] == 1
        assert ts_stats["conversations_tracked"] == 1

    def test_stats_also_has_basic_scheduler_fields(self):
        """get_stats() must still include the basic scheduler fields."""
        sched = _make_scheduler()
        stats = sched.get_stats()
        for key in ("waiting", "running", "total_requests", "finished",
                     "step_counter", "total_prompt_tokens",
                     "total_completion_tokens", "num_requests_processed"):
            assert key in stats, f"Missing key: {key}"


# ===========================================================================
# 4. _thinking_state initialized and cleared on deep_reset
# ===========================================================================

class TestThinkingStateLifecycle:

    def test_thinking_state_starts_empty(self):
        """_thinking_state must be empty right after construction."""
        sched = _make_scheduler()
        assert sched._thinking_state == {}

    def test_thinking_state_populated_manually(self):
        """We can manually populate _thinking_state (simulating response processing)."""
        sched = _make_scheduler()
        sched._thinking_state["req-1"] = {
            "in_thinking": True,
            "thinking_start_idx": 5,
            "was_in_thinking": False,
        }
        assert len(sched._thinking_state) == 1
        assert sched._thinking_state["req-1"]["in_thinking"] is True

    def test_deep_reset_clears_thinking_state(self):
        """deep_reset() must clear _thinking_state."""
        sched = _make_scheduler()
        sched._thinking_state["req-1"] = {
            "in_thinking": True,
            "thinking_start_idx": 5,
            "was_in_thinking": False,
        }
        sched._thinking_state["req-2"] = {
            "in_thinking": False,
            "thinking_start_idx": None,
            "was_in_thinking": True,
        }
        assert len(sched._thinking_state) == 2

        sched.deep_reset()

        assert sched._thinking_state == {}

    def test_deep_reset_clears_thinking_state_even_with_no_batch_gen(self):
        """deep_reset() clears _thinking_state even when _batch_gen is None."""
        sched = _make_scheduler()
        assert sched._batch_gen is None
        sched._thinking_state["req-x"] = {"in_thinking": False, "thinking_start_idx": None, "was_in_thinking": False}
        sched.deep_reset()
        assert sched._thinking_state == {}

    def test_deep_reset_clears_all_other_state(self):
        """deep_reset() clears all related internal state."""
        sched = _make_scheduler()
        sched._thinking_state["r1"] = {"in_thinking": True, "thinking_start_idx": 0, "was_in_thinking": False}
        sched._thinking_processors["r1"] = MagicMock()
        sched._detokenizers["r1"] = MagicMock()
        sched._pending_abort_ids.add("r1")
        sched._uids_to_remove.append(42)

        sched.deep_reset()

        assert sched._thinking_state == {}
        assert sched._thinking_processors == {}
        assert sched._detokenizers == {}
        assert sched._pending_abort_ids == set()
        assert sched._uids_to_remove == []

    def test_shutdown_calls_deep_reset(self):
        """shutdown() delegates to deep_reset(), so it also clears _thinking_state."""
        sched = _make_scheduler()
        sched._thinking_state["r1"] = {"in_thinking": True, "thinking_start_idx": 0, "was_in_thinking": False}
        sched.shutdown()
        assert sched._thinking_state == {}

    def test_thinking_state_not_cleared_by_get_stats(self):
        """get_stats() must not clear _thinking_state as a side-effect."""
        sched = _make_scheduler()
        sched._thinking_state["r1"] = {"in_thinking": True, "thinking_start_idx": 0, "was_in_thinking": False}
        sched.get_stats()
        assert "r1" in sched._thinking_state

    def test_thinking_state_entry_structure(self):
        """Each _thinking_state entry must have the expected keys."""
        sched = _make_scheduler()
        # Simulate what _process_responses creates
        sched._thinking_state["req-1"] = {
            "in_thinking": False,
            "thinking_start_idx": None,
            "was_in_thinking": False,
        }
        entry = sched._thinking_state["req-1"]
        assert "in_thinking" in entry
        assert "thinking_start_idx" in entry
        assert "was_in_thinking" in entry
        assert isinstance(entry["in_thinking"], bool)
