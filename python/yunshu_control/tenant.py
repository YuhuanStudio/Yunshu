"""Yunshu Multi-Tenant — API key auth, quotas, and fairness.

Phase 1: Simple API key authentication with per-tenant rate limiting.
Phase 2: RBAC (org/project/api_key three-tier), SLO classes, priority queues.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class TenantTier(Enum):
    FREE = auto()
    PRO = auto()
    ENTERPRISE = auto()


@dataclass
class Quota:
    """Per-tenant rate limits."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    max_concurrent: int = 5
    max_context_tokens: int = 8192
    priority: int = 0  # Higher = higher priority


@dataclass
class Tenant:
    """A single tenant (API key holder)."""

    tenant_id: str
    name: str
    api_key_hash: str  # SHA256 of the API key
    tier: TenantTier = TenantTier.FREE
    quota: Quota = field(default_factory=Quota)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    # Runtime counters
    _request_count: int = 0
    _token_count: int = 0
    _window_start: float = field(default_factory=time.time)
    _active_requests: int = 0

    def check_rate_limit(self) -> bool:
        """Check if tenant is within rate limits. Returns True if allowed."""
        now = time.time()
        # Reset window every 60 seconds
        if now - self._window_start > 60:
            self._request_count = 0
            self._token_count = 0
            self._window_start = now

        if self._request_count >= self.quota.requests_per_minute:
            return False
        if self._token_count >= self.quota.tokens_per_minute:
            return False
        if self._active_requests >= self.quota.max_concurrent:
            return False
        return True

    def record_request(self, tokens: int = 0) -> None:
        self._request_count += 1
        self._token_count += tokens
        self._active_requests += 1

    def finish_request(self) -> None:
        self._active_requests = max(0, self._active_requests - 1)


class TenantManager:
    """Manages API key authentication and tenant lifecycle."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}  # tenant_id → Tenant
        self._key_to_tenant: dict[str, str] = {}  # key_hash → tenant_id

    @staticmethod
    def hash_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        quota: Quota | None = None,
        api_key: str | None = None,
    ) -> tuple[str, str]:
        """Create a new tenant. Returns (tenant_id, api_key)."""
        key = api_key or secrets.token_urlsafe(32)
        key_hash = self.hash_key(key)

        # Default quotas by tier
        if quota is None:
            quotas = {
                TenantTier.FREE: Quota(requests_per_minute=20, tokens_per_minute=50_000, max_concurrent=2),
                TenantTier.PRO: Quota(requests_per_minute=120, tokens_per_minute=500_000, max_concurrent=10, priority=1),
                TenantTier.ENTERPRISE: Quota(requests_per_minute=600, tokens_per_minute=5_000_000, max_concurrent=50, priority=2),
            }
            quota = quotas.get(tier, Quota())

        tenant_id = f"tn-{secrets.token_hex(4)}"
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            api_key_hash=key_hash,
            tier=tier,
            quota=quota,
        )
        self._tenants[tenant_id] = tenant
        self._key_to_tenant[key_hash] = tenant_id
        return tenant_id, key

    def authenticate(self, api_key: str) -> Optional[Tenant]:
        """Authenticate an API key. Returns tenant or None."""
        key_hash = self.hash_key(api_key)
        tenant_id = self._key_to_tenant.get(key_hash)
        if tenant_id is None:
            return None
        tenant = self._tenants.get(tenant_id)
        if tenant is None or not tenant.is_active:
            return None
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> list[dict]:
        return [
            {
                "id": t.tenant_id,
                "name": t.name,
                "tier": t.tier.name,
                "active": t.is_active,
                "quota_rpm": t.quota.requests_per_minute,
            }
            for t in self._tenants.values()
        ]
