"""Tests for request tracking and cancellation."""
import asyncio
import pytest


class TestRequestTracker:
    def test_register_and_list(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        gen = tracker.register("req-1", "test-model")
        assert gen.request_id == "req-1"
        assert gen.model == "test-model"
        assert not gen.cancel_event.is_set()
        assert tracker.active_count == 1

    def test_unregister(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        tracker.register("req-1")
        tracker.unregister("req-1")
        assert tracker.active_count == 0

    def test_cancel_specific(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        gen = tracker.register("req-1")
        found = tracker.cancel("req-1")
        assert found is True
        assert gen.cancel_event.is_set()

    def test_cancel_not_found(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        found = tracker.cancel("nonexistent")
        assert found is False

    def test_cancel_all(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        g1 = tracker.register("req-1")
        g2 = tracker.register("req-2")
        g3 = tracker.register("req-3")
        count = tracker.cancel_all()
        assert count == 3
        assert g1.cancel_event.is_set()
        assert g2.cancel_event.is_set()
        assert g3.cancel_event.is_set()

    def test_is_cancelled(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        tracker.register("req-1")
        assert not tracker.is_cancelled("req-1")
        tracker.cancel("req-1")
        assert tracker.is_cancelled("req-1")

    def test_is_cancelled_not_found(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        assert not tracker.is_cancelled("nonexistent")

    def test_list_active(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        tracker.register("req-1", "model-a")
        tracker.register("req-2", "model-b")
        active = tracker.list_active()
        assert len(active) == 2
        ids = {a["request_id"] for a in active}
        assert "req-1" in ids
        assert "req-2" in ids

    def test_elapsed_s(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        gen = tracker.register("req-1")
        assert gen.elapsed_s >= 0

    def test_unregister_idempotent(self):
        from yunshu_engine.request_tracker import RequestTracker
        tracker = RequestTracker()
        tracker.register("req-1")
        tracker.unregister("req-1")
        tracker.unregister("req-1")  # should not raise
        assert tracker.active_count == 0


class TestGetRequestTracker:
    def test_singleton(self):
        from yunshu_engine.request_tracker import get_request_tracker
        t1 = get_request_tracker()
        t2 = get_request_tracker()
        assert t1 is t2


class TestCancelEndpoint:
    def test_cancel_specific_request(self):
        from yunshu_gateway.routers.cancel import cancel_generation, CancelRequest
        from yunshu_engine.request_tracker import get_request_tracker
        tracker = get_request_tracker()
        gen = tracker.register("test-cancel-1", "test")
        result = asyncio.run(
            cancel_generation(CancelRequest(request_id="test-cancel-1"))
        )
        assert result["status"] == "cancelled"
        tracker.unregister("test-cancel-1")

    def test_cancel_all_requests(self):
        from yunshu_gateway.routers.cancel import cancel_generation, CancelRequest
        from yunshu_engine.request_tracker import get_request_tracker
        tracker = get_request_tracker()
        tracker.register("test-cancel-all-1")
        tracker.register("test-cancel-all-2")
        result = asyncio.run(
            cancel_generation(CancelRequest(cancel_all=True))
        )
        assert result["status"] == "cancelled"
        assert result["count"] >= 2
        tracker.unregister("test-cancel-all-1")
        tracker.unregister("test-cancel-all-2")

    def test_cancel_not_found(self):
        from yunshu_gateway.routers.cancel import cancel_generation, CancelRequest
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                cancel_generation(CancelRequest(request_id="nonexistent"))
            )
        assert exc_info.value.status_code == 404

    def test_cancel_no_params(self):
        from yunshu_gateway.routers.cancel import cancel_generation, CancelRequest
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                cancel_generation(CancelRequest())
            )
        assert exc_info.value.status_code == 400
