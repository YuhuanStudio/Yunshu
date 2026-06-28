"""Tests for request tracking and cancellation."""
import asyncio
import os

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
    @staticmethod
    def _make_mock_request(headers: dict | None = None):
        """Create a mock FastAPI Request object for testing."""
        from unittest.mock import MagicMock
        mock_req = MagicMock()
        mock_req.headers = headers or {}
        # Simulate no middleware auth state (role/rbac_key/tenant all unset so
        # _check_auth falls through to the static-token / AUTH_DISABLED path).
        mock_state = MagicMock()
        mock_state.role = None
        mock_state.rbac_key = None
        mock_state.tenant = None
        mock_req.state = mock_state
        return mock_req

    def test_cancel_specific_request(self):
        # Ensure no auth token is set so _check_auth passes
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)
        from yunshu_engine.request_tracker import get_request_tracker
        from yunshu_gateway.routers.cancel import CancelRequest, cancel_generation
        tracker = get_request_tracker()
        tracker.register("test-cancel-1", "test")
        result = asyncio.run(
            cancel_generation(CancelRequest(request_id="test-cancel-1"), self._make_mock_request())
        )
        assert result["status"] == "cancelled"
        tracker.unregister("test-cancel-1")

    def test_cancel_all_requests(self):
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)
        from yunshu_engine.request_tracker import get_request_tracker
        from yunshu_gateway.routers.cancel import CancelRequest, cancel_generation
        tracker = get_request_tracker()
        tracker.register("test-cancel-all-1")
        tracker.register("test-cancel-all-2")
        result = asyncio.run(
            cancel_generation(CancelRequest(cancel_all=True), self._make_mock_request())
        )
        assert result["status"] == "cancelled"
        assert result["count"] >= 2
        tracker.unregister("test-cancel-all-1")
        tracker.unregister("test-cancel-all-2")

    def test_cancel_not_found(self):
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)
        from fastapi import HTTPException

        from yunshu_gateway.routers.cancel import CancelRequest, cancel_generation
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                cancel_generation(CancelRequest(request_id="nonexistent"), self._make_mock_request())
            )
        assert exc_info.value.status_code == 404

    def test_cancel_no_params(self):
        os.environ.pop("YUNSHU_AUTH_TOKEN", None)
        from fastapi import HTTPException

        from yunshu_gateway.routers.cancel import CancelRequest, cancel_generation
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                cancel_generation(CancelRequest(), self._make_mock_request())
            )
        assert exc_info.value.status_code == 400

    def test_cancel_rejects_missing_auth(self):
        """Cancel endpoint should require auth when YUNSHU_AUTH_TOKEN is set."""
        # Must also ensure YUNSHU_AUTH_DISABLED is not set (conftest sets it by default)
        old_disabled = os.environ.pop("YUNSHU_AUTH_DISABLED", None)
        os.environ["YUNSHU_AUTH_TOKEN"] = "test-secret-key"
        try:
            from fastapi import HTTPException

            from yunshu_gateway.routers.cancel import CancelRequest, cancel_generation
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    cancel_generation(
                        CancelRequest(request_id="anything"),
                        self._make_mock_request(),
                    )
                )
            assert exc_info.value.status_code == 401
        finally:
            os.environ.pop("YUNSHU_AUTH_TOKEN", None)
            if old_disabled is not None:
                os.environ["YUNSHU_AUTH_DISABLED"] = old_disabled

    def test_cancel_accepts_valid_auth(self):
        """Cancel endpoint should accept requests with valid auth token."""
        # Must also ensure YUNSHU_AUTH_DISABLED is not set
        old_disabled = os.environ.pop("YUNSHU_AUTH_DISABLED", None)
        os.environ["YUNSHU_AUTH_TOKEN"] = "test-secret-key"
        try:
            from yunshu_engine.request_tracker import get_request_tracker
            from yunshu_gateway.routers.cancel import CancelRequest, cancel_generation
            tracker = get_request_tracker()
            tracker.register("test-cancel-auth-1")
            result = asyncio.run(
                cancel_generation(
                    CancelRequest(request_id="test-cancel-auth-1"),
                    self._make_mock_request({"Authorization": "Bearer test-secret-key"}),
                )
            )
            assert result["status"] == "cancelled"
            tracker.unregister("test-cancel-auth-1")
        finally:
            os.environ.pop("YUNSHU_AUTH_TOKEN", None)
            if old_disabled is not None:
                os.environ["YUNSHU_AUTH_DISABLED"] = old_disabled
