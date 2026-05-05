"""RBAC (Role-Based Access Control) system tests."""

import pytest
import time

from yunshu_control.role_manager import (
    RBACManager,
    Role,
    SLOClass,
    RolePermissions,
    ROLE_PERMISSIONS,
)


class TestRolePermissions:
    def test_admin_has_all_permissions(self):
        perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert perms.can_load_models
        assert perms.can_unload_models
        assert perms.can_register_models
        assert perms.can_manage_tokens
        assert perms.can_view_admin
        assert perms.can_benchmark

    def test_developer_permissions(self):
        perms = ROLE_PERMISSIONS[Role.DEVELOPER]
        assert perms.can_load_models
        assert perms.can_unload_models
        assert not perms.can_register_models
        assert not perms.can_manage_tokens
        assert perms.can_view_admin

    def test_user_permissions(self):
        perms = ROLE_PERMISSIONS[Role.USER]
        assert not perms.can_load_models
        assert not perms.can_unload_models
        assert not perms.can_register_models
        assert not perms.can_manage_tokens
        assert not perms.can_view_admin


class TestSLOClass:
    def test_scheduling_priority(self):
        from yunshu_control.role_manager import APIKey

        key_be = APIKey(key_hash="test", name="test", slo_class=SLOClass.BEST_EFFORT)
        key_std = APIKey(key_hash="test", name="test", slo_class=SLOClass.STANDARD)
        key_prem = APIKey(key_hash="test", name="test", slo_class=SLOClass.PREMIUM)

        assert key_be.get_scheduling_priority() < key_std.get_scheduling_priority()
        assert key_std.get_scheduling_priority() < key_prem.get_scheduling_priority()


class TestAPIKey:
    def test_default_not_expired(self):
        from yunshu_control.role_manager import APIKey

        key = APIKey(key_hash="test", name="test")
        assert not key.is_expired()

    def test_expired_key(self):
        from yunshu_control.role_manager import APIKey

        key = APIKey(key_hash="test", name="test", expires_at=time.time() - 100)
        assert key.is_expired()

    def test_has_permission(self):
        from yunshu_control.role_manager import APIKey

        admin_key = APIKey(key_hash="test", name="admin", role=Role.ADMIN)
        assert admin_key.has_permission("can_load_models")
        assert admin_key.has_permission("can_view_admin")

        user_key = APIKey(key_hash="test", name="user", role=Role.USER)
        assert not user_key.has_permission("can_load_models")

    def test_can_access_model_wildcard(self):
        from yunshu_control.role_manager import APIKey

        key = APIKey(key_hash="test", name="test", role=Role.USER)
        assert key.can_access_model("any-model")
        assert key.can_access_model("qwen3-7b")


class TestRBACManager:
    def test_create_and_authenticate(self):
        manager = RBACManager()
        raw_key, api_key = manager.create_key("test-key", role=Role.ADMIN)

        assert raw_key.startswith("ys_")
        assert api_key.name == "test-key"
        assert api_key.role == Role.ADMIN

        # Authenticate
        auth = manager.authenticate(raw_key)
        assert auth is not None
        assert auth.name == "test-key"

    def test_authenticate_wrong_key(self):
        manager = RBACManager()
        manager.create_key("test", role=Role.USER)

        auth = manager.authenticate("ys_wrong_key")
        assert auth is None

    def test_revoke_key(self):
        manager = RBACManager()
        raw_key, _ = manager.create_key("test", role=Role.USER)

        count = manager.revoke_key("test")
        assert count == 1

        auth = manager.authenticate(raw_key)
        assert auth is None  # revoked

    def test_delete_key(self):
        manager = RBACManager()
        raw_key, _ = manager.create_key("test", role=Role.USER)

        count = manager.delete_key("test")
        assert count == 1

    def test_list_keys(self):
        manager = RBACManager()
        manager.create_key("key1", role=Role.ADMIN)
        manager.create_key("key2", role=Role.USER)

        keys = manager.list_keys()
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"key1", "key2"}

    def test_get_permissions(self):
        manager = RBACManager()
        perms = manager.get_permissions(Role.ADMIN)
        assert perms.can_load_models

    def test_create_key_with_expiry(self):
        manager = RBACManager()
        raw_key, api_key = manager.create_key(
            "temp", role=Role.USER, expires_days=1
        )
        assert api_key.expires_at is not None
        assert not api_key.is_expired()

    def test_multiple_roles(self):
        manager = RBACManager()
        _, admin = manager.create_key("admin", role=Role.ADMIN)
        _, dev = manager.create_key("dev", role=Role.DEVELOPER)
        _, user = manager.create_key("user", role=Role.USER)

        assert admin.role == Role.ADMIN
        assert dev.role == Role.DEVELOPER
        assert user.role == Role.USER

        assert ROLE_PERMISSIONS[Role.ADMIN].max_models == 100
        assert ROLE_PERMISSIONS[Role.DEVELOPER].max_models == 10
        assert ROLE_PERMISSIONS[Role.USER].max_models == 1


class TestRBACPersistence:
    def test_keys_survive_restart(self, tmp_path):
        path = tmp_path / "keys.json"
        mgr = RBACManager(persist_path=path)
        raw_key, api_key = mgr.create_key("test-user", role=Role.ADMIN)
        assert path.exists()

        # Simulate restart — new manager loads from same file
        mgr2 = RBACManager(persist_path=path)
        result = mgr2.authenticate(raw_key)
        assert result is not None
        assert result.name == "test-user"
        assert result.role == Role.ADMIN

    def test_revoke_persists(self, tmp_path):
        path = tmp_path / "keys.json"
        mgr = RBACManager(persist_path=path)
        raw_key, _ = mgr.create_key("to-revoke")

        mgr.revoke_key("to-revoke")
        mgr2 = RBACManager(persist_path=path)
        assert mgr2.authenticate(raw_key) is None

    def test_delete_persists(self, tmp_path):
        path = tmp_path / "keys.json"
        mgr = RBACManager(persist_path=path)
        raw_key, _ = mgr.create_key("to-delete")

        mgr.delete_key("to-delete")
        mgr2 = RBACManager(persist_path=path)
        assert mgr2.authenticate(raw_key) is None
        assert mgr2.list_keys() == []

    def test_corrupted_file_starts_empty(self, tmp_path):
        path = tmp_path / "keys.json"
        path.write_text("not valid json {{{")
        mgr = RBACManager(persist_path=path)
        assert mgr.list_keys() == []

    def test_missing_file_starts_empty(self, tmp_path):
        path = tmp_path / "nonexistent" / "keys.json"
        mgr = RBACManager(persist_path=path)
        assert mgr.list_keys() == []

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "keys.json"
        mgr = RBACManager(persist_path=path)
        mgr.create_key("test")
        assert path.exists()

    def test_slo_class_persists(self, tmp_path):
        path = tmp_path / "keys.json"
        mgr = RBACManager(persist_path=path)
        raw_key, _ = mgr.create_key("premium-user", slo_class=SLOClass.PREMIUM)
        mgr2 = RBACManager(persist_path=path)
        result = mgr2.authenticate(raw_key)
        assert result.slo_class == SLOClass.PREMIUM

    def test_custom_rate_limits_persist(self, tmp_path):
        path = tmp_path / "keys.json"
        mgr = RBACManager(persist_path=path)
        raw_key, _ = mgr.create_key("limited", requests_per_minute=10, tokens_per_minute=1000)
        mgr2 = RBACManager(persist_path=path)
        result = mgr2.authenticate(raw_key)
        assert result.requests_per_minute == 10
        assert result.tokens_per_minute == 1000
