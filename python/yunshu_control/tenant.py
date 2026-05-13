"""Yunshu Multi-Tenant — Re-export from tenant_store (persistent version).

The non-persistent TenantManager has been replaced by tenant_store.TenantManager
which uses SQLite for persistence. Use `from yunshu_control import TenantManager`
or `from yunshu_control.tenant_store import TenantManager` directly.
"""
from .tenant_store import Quota, Tenant, TenantManager, TenantTier  # noqa: F401

import warnings
warnings.warn(
    "yunshu_control.tenant is deprecated, use yunshu_control.tenant_store",
    DeprecationWarning,
    stacklevel=2,
)
