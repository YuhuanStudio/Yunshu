"""Tests for PrefillProgressTracker — per-request prefill progress tracking."""

from yunshu_engine.prefill_progress import PrefillProgressTracker, get_prefill_tracker


class TestPrefillProgressTracker:
    def test_empty(self):
        tracker = PrefillProgressTracker()
        assert tracker.active_count == 0
        assert tracker.get_all_progress() == {}

    def test_update_adds_entry(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        assert tracker.active_count == 1

    def test_update_removes_on_complete(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        tracker.update("req-1", 100, 100, "model-a")
        assert tracker.active_count == 0

    def test_remove(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        tracker.remove("req-1")
        assert tracker.active_count == 0

    def test_remove_nonexistent(self):
        tracker = PrefillProgressTracker()
        tracker.remove("nonexistent")  # Should not raise

    def test_get_model_progress(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        tracker.update("req-2", 30, 60, "model-b")
        result = tracker.get_model_progress("model-a")
        assert len(result) == 1
        assert result[0]["request_id"] == "req-1"
        assert result[0]["progress_pct"] == 50.0

    def test_get_model_progress_filters(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        tracker.update("req-2", 30, 60, "model-b")
        result = tracker.get_model_progress("model-c")
        assert len(result) == 0

    def test_get_all_progress(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        tracker.update("req-2", 30, 60, "model-b")
        result = tracker.get_all_progress()
        assert "model-a" in result
        assert "model-b" in result
        assert len(result["model-a"]) == 1
        assert len(result["model-b"]) == 1

    def test_progress_pct_calculation(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 75, 100, "model-a")
        result = tracker.get_model_progress("model-a")
        assert result[0]["progress_pct"] == 75.0

    def test_speed_tracking(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 0, 100, "model-a")
        tracker.update("req-1", 50, 100, "model-a")
        result = tracker.get_model_progress("model-a")
        assert result[0]["speed_tok_s"] > 0

    def test_eta_calculation(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 0, 100, "model-a")
        tracker.update("req-1", 50, 100, "model-a")
        result = tracker.get_model_progress("model-a")
        assert result[0]["eta_s"] is not None

    def test_eta_none_when_no_speed(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        result = tracker.get_model_progress("model-a")
        assert result[0]["eta_s"] is None

    def test_clear(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 100, "model-a")
        tracker.update("req-2", 30, 60, "model-b")
        tracker.clear()
        assert tracker.active_count == 0

    def test_singleton(self):
        t1 = get_prefill_tracker()
        t2 = get_prefill_tracker()
        assert t1 is t2
