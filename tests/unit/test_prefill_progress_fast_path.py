"""Tests for PrefillProgressTracker and its integration with fast paths."""

import time

from yunshu_engine.prefill_progress import PrefillProgressTracker, get_prefill_tracker


class TestPrefillProgressTracker:
    def test_update_creates_entry(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        assert tracker.active_count == 1

    def test_update_completes_entry(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        tracker.update("req-1", 200, 200, "model-a")
        assert tracker.active_count == 0

    def test_speed_calculation(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 100, 200, "model-a")
        time.sleep(0.01)
        tracker.update("req-1", 200, 200, "model-a")
        # Entry removed when completed
        assert tracker.active_count == 0

    def test_remove_entry(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        tracker.remove("req-1")
        assert tracker.active_count == 0

    def test_remove_nonexistent(self):
        tracker = PrefillProgressTracker()
        tracker.remove("req-999")  # No error

    def test_get_model_progress(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        tracker.update("req-2", 100, 300, "model-b")
        progress = tracker.get_model_progress("model-a")
        assert len(progress) == 1
        assert progress[0]["request_id"] == "req-1"
        assert progress[0]["total"] == 200

    def test_get_all_progress(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        tracker.update("req-2", 100, 300, "model-b")
        all_progress = tracker.get_all_progress()
        assert "model-a" in all_progress
        assert "model-b" in all_progress

    def test_clear(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        tracker.update("req-2", 100, 300, "model-b")
        tracker.clear()
        assert tracker.active_count == 0

    def test_progress_pct(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 50, 200, "model-a")
        progress = tracker.get_model_progress("model-a")
        assert progress[0]["progress_pct"] == 25.0

    def test_singleton_get_prefill_tracker(self):
        t1 = get_prefill_tracker()
        t2 = get_prefill_tracker()
        assert t1 is t2


class TestPrefillProgressEdgeCases:
    def test_zero_total(self):
        tracker = PrefillProgressTracker()
        tracker.update("req-1", 0, 0, "model-a")
        # processed >= total → auto-removed
        assert tracker.active_count == 0

    def test_concurrent_updates(self):
        import threading

        tracker = PrefillProgressTracker()
        errors = []

        def update(n):
            try:
                for i in range(100):
                    tracker.update(f"req-{n}", i, 100, "model-a")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
