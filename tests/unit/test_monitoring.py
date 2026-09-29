"""Tests for MTP + RadixTree monitoring endpoints and gateway fixes."""

import pytest


class TestMTPMonitoringEndpoint:
    """Verify MTP stats are exposed in /spec-decode monitoring endpoint."""

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        from yunshu_gateway.main import create_app

        monkeypatch.setenv("YUNSHU_DEBUG_ROUTES", "1")
        app = create_app()
        return TestClient(app)

    def test_spec_decode_endpoint_includes_mtp_field(self, client):
        resp = client.get("/debug/spec-decode")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # Each model should have mtp_enabled field
        for model_info in data["models"]:
            assert "mtp_enabled" in model_info


class TestCompletionsSpecDecode:
    """Verify spec_decode is forwarded in completions streaming path."""

    def test_completions_request_has_spec_decode(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(model="test", prompt="test", spec_decode=True)
        assert req.spec_decode is True

    def test_completions_request_has_logprobs(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(model="test", prompt="test", logprobs=1)
        assert req.logprobs >= 1


class TestMTPPatchImport:
    """Verify mtp_patch functions are importable and callable."""

    def test_apply_mtp_patch_importable(self):
        from yunshu_engine.mtp_patch import apply_mtp_patch

        assert callable(apply_mtp_patch)

    def test_load_model_with_mtp_importable(self):
        from yunshu_engine.mtp_patch import load_model_with_mtp

        assert callable(load_model_with_mtp)

    def test_apply_mtp_patch_idempotent(self):
        from yunshu_engine.mtp_patch import apply_mtp_patch

        result1 = apply_mtp_patch()
        result2 = apply_mtp_patch()
        assert result1 == result2


class TestChatRouterMultiChoiceSpecDecode:
    """Verify multi-choice streaming path accepts spec_decode."""

    def test_chat_request_has_spec_decode(self):
        from yunshu_gateway.routers.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "test"}],
            spec_decode=True,
        )
        assert req.spec_decode is True


class TestMonitoringEndpointsComplete:
    """Verify all monitoring endpoints are accessible."""

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        from yunshu_gateway.main import create_app

        monkeypatch.setenv("YUNSHU_DEBUG_ROUTES", "1")
        app = create_app()
        return TestClient(app)

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/debug/system",
            "/debug/models",
            "/debug/spec-decode",
            "/debug/kv-cache",
            "/debug/memory-guard",
            "/debug/ssd-cache",
            "/debug/per-model",
            "/debug/prometheus",
            "/metrics",
        ],
    )
    def test_endpoint_200(self, client, endpoint):
        resp = client.get(endpoint)
        assert resp.status_code == 200, f"{endpoint} returned {resp.status_code}"


def test_debug_routes_off_by_default(monkeypatch):
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    monkeypatch.delenv("YUNSHU_DEBUG_ROUTES", raising=False)
    client = TestClient(create_app())
    assert client.get("/debug/engine").status_code == 404
    assert client.get("/metrics").status_code == 200
