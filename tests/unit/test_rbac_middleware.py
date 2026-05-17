"""RBAC-integrated middleware and admin endpoint tests."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from yunshu_engine.engine import Engine, EngineConfig


class TestTenantAuthMiddlewareRBAC:
    """Test that TenantAuthMiddleware routes ys_ keys through RBACManager."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_ys_key_authenticates_via_rbac(self):
        from yunshu_gateway.main import create_app
        from yunshu_control.role_manager import RBACManager, Role

        app = create_app()
        rbac = RBACManager()
        raw_key, _ = rbac.create_key("test-user", role=Role.ADMIN)
        app.state.rbac_manager = rbac

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            # ys_ key should authenticate
            resp = client.get(
                "/api/v1/admin/models",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert resp.status_code == 200

    def test_ys_key_invalid_rejected(self):
        from yunshu_gateway.main import create_app
        from yunshu_control.role_manager import RBACManager

        app = create_app()
        app.state.rbac_manager = RBACManager()

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            resp = client.get(
                "/api/v1/admin/models",
                headers={"Authorization": "Bearer ys_invalid_key"},
            )
            assert resp.status_code == 401

    def test_non_ys_key_uses_static_token(self):
        from yunshu_gateway.main import create_app

        app = create_app()

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "my-secret"}):
            client = TestClient(app)
            # Non-ys_ key should match static token
            resp = client.get(
                "/api/v1/admin/models",
                headers={"Authorization": "Bearer my-secret"},
            )
            assert resp.status_code == 200

    def test_public_paths_skip_auth(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_no_auth_token_allows_all(self):
        """With YUNSHU_AUTH_DISABLED=true, requests pass without token."""
        from yunshu_gateway.main import create_app

        app = create_app()
        with patch.dict(os.environ, {"YUNSHU_AUTH_DISABLED": "true"}, clear=False):
            if "YUNSHU_AUTH_TOKEN" in os.environ:
                del os.environ["YUNSHU_AUTH_TOKEN"]
            client = TestClient(app)
            resp = client.get("/api/v1/admin/models")
            assert resp.status_code == 200


class TestRBACKeyManagement:
    """Test admin key management endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)
        self._auth_env = patch.dict(os.environ, {"YUNSHU_AUTH_DISABLED": "true"})
        self._auth_env.start()

    def teardown_method(self):
        self._auth_env.stop()

    def test_create_rbac_key(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "dev-key", "role": "developer", "slo_class": "premium"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"].startswith("ys_")
        assert data["name"] == "dev-key"
        assert data["role"] == "DEVELOPER"
        assert data["slo_class"] == "PREMIUM"

    def test_list_rbac_keys(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        client.post("/api/v1/admin/keys", json={"name": "key1", "role": "admin"})
        client.post("/api/v1/admin/keys", json={"name": "key2", "role": "user"})

        resp = client.get("/api/v1/admin/keys")
        assert resp.status_code == 200
        keys = resp.json()["keys"]
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"key1", "key2"}

    def test_delete_rbac_key(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        client.post("/api/v1/admin/keys", json={"name": "to-delete", "role": "user"})

        resp = client.delete("/api/v1/admin/keys/to-delete")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        # Should be gone from list
        resp = client.get("/api/v1/admin/keys")
        keys = resp.json()["keys"]
        assert not any(k["name"] == "to-delete" for k in keys)

    def test_delete_nonexistent_key(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.delete("/api/v1/admin/keys/nonexistent")
        assert resp.status_code == 404

    def test_create_key_invalid_role(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "bad", "role": "superadmin"},
        )
        assert resp.status_code == 400

    def test_create_key_invalid_slo(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "bad", "slo_class": "ultra"},
        )
        assert resp.status_code == 400

    def test_create_key_with_expiry(self):
        from yunshu_gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "temp", "role": "user", "expires_days": 7},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["expires_at"] is not None


class TestRBACPermissionEnforcement:
    """Test that RBAC permissions are enforced on admin endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yunshu_gateway.engine import set_engine
        self._engine = Engine(EngineConfig())
        self._engine._model = object()
        self._engine._model_name = "test-model"
        self._engine._running = True
        set_engine(self._engine)

    def test_user_cannot_register_models(self):
        from yunshu_gateway.main import create_app
        from yunshu_control.role_manager import RBACManager, Role

        app = create_app()
        rbac = RBACManager()
        raw_key, _ = rbac.create_key("user-key", role=Role.USER)
        app.state.rbac_manager = rbac

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/admin/models/register",
                json={"model_id": "test", "model_path": "/tmp/test"},
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert resp.status_code == 403

    def test_admin_can_register_models(self):
        from yunshu_gateway.main import create_app
        from yunshu_control.role_manager import RBACManager, Role

        app = create_app()
        rbac = RBACManager()
        raw_key, _ = rbac.create_key("admin-key", role=Role.ADMIN)
        app.state.rbac_manager = rbac

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            resp = client.get(
                "/api/v1/admin/models",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert resp.status_code == 200

    def test_developer_can_load_models(self):
        from yunshu_gateway.main import create_app
        from yunshu_control.role_manager import RBACManager, Role

        app = create_app()
        rbac = RBACManager()
        raw_key, _ = rbac.create_key("dev-key", role=Role.DEVELOPER)
        app.state.rbac_manager = rbac

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            # Developer has can_load_models and can_view_admin
            resp = client.get(
                "/api/v1/admin/models",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert resp.status_code == 200

    def test_user_cannot_manage_tokens(self):
        from yunshu_gateway.main import create_app
        from yunshu_control.role_manager import RBACManager, Role

        app = create_app()
        rbac = RBACManager()
        raw_key, _ = rbac.create_key("user-key", role=Role.USER)
        app.state.rbac_manager = rbac

        with patch.dict(os.environ, {"YUNSHU_AUTH_TOKEN": "secret"}):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/admin/keys",
                json={"name": "new-key", "role": "user"},
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert resp.status_code == 403

    def test_no_auth_configured_allows_all(self):
        """With YUNSHU_AUTH_DISABLED=true, all endpoints are open."""
        from yunshu_gateway.main import create_app

        app = create_app()
        with patch.dict(os.environ, {"YUNSHU_AUTH_DISABLED": "true"}):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/admin/keys",
                json={"name": "open-key", "role": "admin"},
            )
            assert resp.status_code == 200
