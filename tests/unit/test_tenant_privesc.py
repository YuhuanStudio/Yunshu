"""the legacy-tenant privesc keystone (W730/W451/W691) was unswept in three
_check_permission copies — models.py (load/unload), sleep.py (sleep/wake), bench.py
(benchmarks). Each had `if tenant is not None: return`, granting a low-privilege legacy
tenant EVERY permission — so a legacy tenant could load/unload models and run GPU
benchmarks, outranking even a USER RBAC key (correctly 403'd). Legacy tenants now keep
inference-class access but admin-class ops require a real admin role."""
from __future__ import annotations

import types

import pytest
from fastapi import HTTPException

from yunshu_gateway.routers import (
    bench as BENCH,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers import (
    models as MODELS,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers import (
    sleep as SLEEP,  # noqa: N812  # intentional short module alias
)


def _req(tenant="t1", role=""):
    st = types.SimpleNamespace(rbac_key=None, tenant=tenant, role=role)
    return types.SimpleNamespace(state=st, headers={})


@pytest.fixture(autouse=True)
def _no_auth_disabled(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)


def test_legacy_tenant_keeps_inference():
    # can_infer is non-privileged → allowed (no raise)
    MODELS._check_permission(_req(), "can_infer")


def test_legacy_tenant_denied_model_load():
    with pytest.raises(HTTPException) as ei:
        MODELS._check_permission(_req(), "can_load_models")
    assert ei.value.status_code == 403
    with pytest.raises(HTTPException):
        MODELS._check_permission(_req(), "can_unload_models")


def test_admin_role_tenant_allowed_load():
    MODELS._check_permission(_req(role="admin"), "can_load_models")
    MODELS._check_permission(_req(role="org-ADMIN"), "can_load_models")


def test_sleep_denied_for_legacy_tenant():
    with pytest.raises(HTTPException) as ei:
        SLEEP._check_permission(_req(), "can_unload_models")
    assert ei.value.status_code == 403
    # admin tenant ok
    SLEEP._check_permission(_req(role="system"), "can_unload_models")


def test_bench_denied_for_legacy_tenant():
    with pytest.raises(HTTPException) as ei:
        BENCH._check_permission(_req())
    assert ei.value.status_code == 403
    # admin tenant ok
    BENCH._check_permission(_req(role="admin"))


def test_privileged_set_covers_the_dangerous_perms():
    for p in ("can_load_models", "can_unload_models", "can_admin", "can_benchmark"):
        assert p in MODELS._TENANT_DENIED_PERMISSIONS
    assert "can_infer" not in MODELS._TENANT_DENIED_PERMISSIONS
