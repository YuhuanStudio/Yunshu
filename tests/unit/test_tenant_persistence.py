"""Tests for tenant persistence (JSON file store)."""
import json
import os
import tempfile

import pytest

from yunshu_control.tenant_store import TenantManager, TenantTier


class TestTenantPersistence:
    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "tenants.json")
        mgr1 = TenantManager(persist_path=path)
        tid, key = mgr1.create_tenant("alice", TenantTier.PRO)

        # File should exist
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["tenants"]) == 1

        # Reload from file
        mgr2 = TenantManager(persist_path=path)
        tenant = mgr2.authenticate(key)
        assert tenant is not None
        assert tenant.name == "alice"
        assert tenant.tier == TenantTier.PRO

    def test_persist_multiple_tenants(self, tmp_path):
        path = str(tmp_path / "tenants.json")
        mgr = TenantManager(persist_path=path)
        mgr.create_tenant("alice", TenantTier.FREE)
        mgr.create_tenant("bob", TenantTier.ENTERPRISE)

        mgr2 = TenantManager(persist_path=path)
        assert len(mgr2.list_tenants()) == 2

    def test_deactivate_persists(self, tmp_path):
        path = str(tmp_path / "tenants.json")
        mgr1 = TenantManager(persist_path=path)
        tid, key = mgr1.create_tenant("alice")

        mgr1.deactivate_tenant(tid)

        mgr2 = TenantManager(persist_path=path)
        tenant = mgr2.authenticate(key)
        assert tenant is None  # deactivated

    def test_delete_persists(self, tmp_path):
        path = str(tmp_path / "tenants.json")
        mgr1 = TenantManager(persist_path=path)
        tid, key = mgr1.create_tenant("alice")

        mgr1.delete_tenant(tid)

        mgr2 = TenantManager(persist_path=path)
        assert len(mgr2.list_tenants()) == 0

    def test_no_path_is_in_memory(self):
        mgr = TenantManager()
        tid, key = mgr.create_tenant("alice")
        assert mgr.authenticate(key) is not None

    def test_corrupt_file_loads_empty(self, tmp_path):
        path = str(tmp_path / "tenants.json")
        with open(path, "w") as f:
            f.write("not json")
        mgr = TenantManager(persist_path=path)
        assert len(mgr.list_tenants()) == 0

    def test_missing_file_loads_empty(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        mgr = TenantManager(persist_path=path)
        assert len(mgr.list_tenants()) == 0

    def test_atomic_write_no_partial(self, tmp_path):
        path = str(tmp_path / "tenants.json")
        mgr = TenantManager(persist_path=path)
        mgr.create_tenant("alice")

        with open(path) as f:
            data = json.load(f)
        assert "tenants" in data
        # No .tmp file should remain
        assert not os.path.exists(path + ".tmp")
