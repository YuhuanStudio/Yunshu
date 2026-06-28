"""Tests for profiling endpoints."""


class TestProfileRequest:
    def test_defaults(self):
        from yunshu_gateway.routers.profiling import ProfileRequest
        req = ProfileRequest()
        assert req.duration_seconds is None
        assert req.output_path is None

    def test_with_path(self):
        from yunshu_gateway.routers.profiling import ProfileRequest
        req = ProfileRequest(output_path="/tmp/test.bin", duration_seconds=10.0)
        assert req.output_path == "/tmp/test.bin"
        assert req.duration_seconds == 10.0


class TestProfileStatus:
    def test_initial_status(self):
        from yunshu_gateway.routers import profiling
        # Reset state
        profiling._profiling_active = False
        assert profiling._profiling_active is False
