"""Tests for multi-tenant auth and model manager."""

import pytest

from yunshu_control.tenant import Quota, TenantManager, TenantTier


class TestTenantManager:
    def test_create_and_authenticate(self):
        mgr = TenantManager()
        tenant_id, api_key = mgr.create_tenant("test-user")

        tenant = mgr.authenticate(api_key)
        assert tenant is not None
        assert tenant.name == "test-user"
        assert tenant.tenant_id == tenant_id

    def test_invalid_key(self):
        mgr = TenantManager()
        mgr.create_tenant("test-user")
        assert mgr.authenticate("invalid-key") is None

    def test_rate_limiting(self):
        mgr = TenantManager()
        tid, key = mgr.create_tenant("limited", quota=Quota(requests_per_minute=3))

        tenant = mgr.authenticate(key)
        assert tenant.check_rate_limit() is True
        tenant.record_request()
        assert tenant.check_rate_limit() is True
        tenant.record_request()
        assert tenant.check_rate_limit() is True
        tenant.record_request()
        # 3 requests in window → should be blocked
        assert tenant.check_rate_limit() is False

    def test_concurrent_limit(self):
        mgr = TenantManager()
        tid, key = mgr.create_tenant(
            "limited", quota=Quota(max_concurrent=2)
        )
        tenant = mgr.authenticate(key)

        assert tenant.check_rate_limit() is True
        tenant.record_request()
        tenant.record_request()
        assert tenant.check_rate_limit() is False  # 2 active

        tenant.finish_request()
        assert tenant.check_rate_limit() is True  # back to 1 active

    def test_tier_quotas(self):
        mgr = TenantManager()
        _, _ = mgr.create_tenant("free", tier=TenantTier.FREE)
        _, _ = mgr.create_tenant("pro", tier=TenantTier.PRO)
        _, _ = mgr.create_tenant("enterprise", tier=TenantTier.ENTERPRISE)

        tenants = mgr.list_tenants()
        assert len(tenants) == 3
        free = next(t for t in tenants if t["name"] == "free")
        assert free["quota_rpm"] == 20

    def test_list_tenants(self):
        mgr = TenantManager()
        mgr.create_tenant("user1")
        mgr.create_tenant("user2")

        result = mgr.list_tenants()
        assert len(result) == 2
        names = {t["name"] for t in result}
        assert names == {"user1", "user2"}


class TestTenantIsolation:
    """Verify tenants cannot affect each other."""

    def test_different_keys_different_tenants(self):
        """Each API key maps to exactly one tenant."""
        mgr = TenantManager()
        tid1, key1 = mgr.create_tenant("tenant-a")
        tid2, key2 = mgr.create_tenant("tenant-b")

        t1 = mgr.authenticate(key1)
        t2 = mgr.authenticate(key2)

        assert t1.tenant_id != t2.tenant_id
        assert t1.name == "tenant-a"
        assert t2.name == "tenant-b"

    def test_cross_key_cannot_authenticate(self):
        """Tenant A's key cannot authenticate as Tenant B."""
        mgr = TenantManager()
        _, key_a = mgr.create_tenant("tenant-a")
        _, key_b = mgr.create_tenant("tenant-b")

        # A's key gets A's tenant
        t = mgr.authenticate(key_a)
        assert t.name == "tenant-a"

        # B's key gets B's tenant
        t = mgr.authenticate(key_b)
        assert t.name == "tenant-b"

    def test_rate_limiting_isolated_per_tenant(self):
        """One tenant's rate limit doesn't affect another tenant."""
        mgr = TenantManager()
        _, key_a = mgr.create_tenant("tenant-a", quota=Quota(requests_per_minute=2))
        _, key_b = mgr.create_tenant("tenant-b", quota=Quota(requests_per_minute=100))

        t_a = mgr.authenticate(key_a)
        t_b = mgr.authenticate(key_b)

        # Exhaust tenant A's rate limit
        t_a.record_request()
        t_a.record_request()
        assert t_a.check_rate_limit() is False

        # Tenant B is unaffected
        assert t_b.check_rate_limit() is True

    def test_concurrent_requests_isolated(self):
        """One tenant hitting concurrent limit doesn't block another."""
        mgr = TenantManager()
        _, key_a = mgr.create_tenant("a", quota=Quota(max_concurrent=1))
        _, key_b = mgr.create_tenant("b", quota=Quota(max_concurrent=5))

        t_a = mgr.authenticate(key_a)
        t_b = mgr.authenticate(key_b)

        t_a.record_request()
        assert t_a.check_rate_limit() is False  # max 1

        assert t_b.check_rate_limit() is True  # unaffected

    def test_deactivated_tenant_cannot_authenticate(self):
        """Deactivating one tenant doesn't affect others."""
        mgr = TenantManager()
        tid_a, key_a = mgr.create_tenant("tenant-a")
        _, key_b = mgr.create_tenant("tenant-b")

        mgr.deactivate_tenant(tid_a)
        assert mgr.authenticate(key_a) is None  # deactivated
        assert mgr.authenticate(key_b) is not None  # still works

    def test_deleted_tenant_cannot_authenticate(self):
        """Deleting one tenant doesn't affect others."""
        mgr = TenantManager()
        tid_a, key_a = mgr.create_tenant("tenant-a")
        _, key_b = mgr.create_tenant("tenant-b")

        mgr.delete_tenant(tid_a)
        assert mgr.authenticate(key_a) is None  # deleted
        assert mgr.authenticate(key_b) is not None  # still works

    def test_inactive_tenant_cannot_authenticate(self):
        """Inactive tenant is rejected at auth time."""
        mgr = TenantManager()
        tid, key = mgr.create_tenant("inactive-test")
        mgr.deactivate_tenant(tid)

        result = mgr.authenticate(key)
        assert result is None
