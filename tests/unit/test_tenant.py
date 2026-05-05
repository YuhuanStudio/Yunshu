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
