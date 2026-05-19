from __future__ import annotations
"""Yunshu Multi-Tenant — API key auth, quotas, fairness, and persistence.

Phase 1: Simple API key authentication with per-tenant rate limiting.
Phase 2: RBAC (org/project/api_key three-tier), SLO classes, priority queues.

Persistence: Tenants saved to JSON file so API keys survive restarts.
"""

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


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
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "max_concurrent": self.max_concurrent,
            "max_context_tokens": self.max_context_tokens,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Quota:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Tenant:
    """A single tenant (API key holder)."""

    tenant_id: str
    name: str
    api_key_hash: str
    tier: TenantTier = TenantTier.FREE
    quota: Quota = field(default_factory=Quota)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    # Runtime counters (not persisted)
    _request_count: int = 0
    _token_count: int = 0
    _window_start: float = field(default_factory=time.monotonic)
    _active_requests: int = 0
    # Lock for thread-safe rate-limit checks (not persisted)
    _rate_limit_lock: threading.Lock = field(default_factory=threading.Lock)

    def check_rate_limit(self) -> bool:
        """Check if tenant is within rate limits. Returns True if allowed."""
        with self._rate_limit_lock:
            now = time.monotonic()
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

    def check_and_record(self, tokens: int = 0) -> bool:
        """Atomically check rate limit and record if allowed. Returns True if allowed."""
        with self._rate_limit_lock:
            now = time.monotonic()
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

            self._request_count += 1
            self._token_count += tokens
            self._active_requests += 1
            return True

    def record_request(self, tokens: int = 0) -> None:
        with self._rate_limit_lock:
            self._request_count += 1
            self._token_count += tokens
            self._active_requests += 1

    def finish_request(self) -> None:
        with self._rate_limit_lock:
            self._active_requests = max(0, self._active_requests - 1)

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "api_key_hash": self.api_key_hash,
            "tier": self.tier.name,
            "quota": self.quota.to_dict(),
            "is_active": self.is_active,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tenant:
        return cls(
            tenant_id=d["tenant_id"],
            name=d["name"],
            api_key_hash=d["api_key_hash"],
            tier=TenantTier[d.get("tier", "FREE")],
            quota=Quota.from_dict(d["quota"]) if "quota" in d else Quota(),
            is_active=d.get("is_active", True),
            created_at=d.get("created_at", time.time()),
        )


class TenantManager:
    """Manages API key authentication and tenant lifecycle with JSON persistence."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._key_to_tenant: dict[str, str] = {}
        self._persist_path = persist_path
        self._lock = threading.Lock()

        if persist_path:
            self._load()

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

        with self._lock:
            self._tenants[tenant_id] = tenant
            self._key_to_tenant[key_hash] = tenant_id
            self._persist()

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

    def list_tenants(self) -> list[dict]:
        with self._lock:
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

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant (key stops working but record kept)."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False
            tenant.is_active = False
            self._persist()
            return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """Permanently delete a tenant."""
        with self._lock:
            tenant = self._tenants.pop(tenant_id, None)
            if tenant is None:
                return False
            self._key_to_tenant.pop(tenant.api_key_hash, None)
            self._persist()
            return True

    def _persist(self) -> None:
        """Save tenants to JSON file."""
        if not self._persist_path:
            return
        try:
            data = {
                "version": 1,
                "tenants": [t.to_dict() for t in self._tenants.values()],
            }
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._persist_path)
            logger.debug(f"Persisted {len(self._tenants)} tenants to {self._persist_path}")
        except Exception as e:
            logger.error(f"Failed to persist tenants: {e}")

    def _load(self) -> None:
        """Load tenants from JSON file."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)

            if data.get("version") != 1:
                logger.warning(f"Unknown tenant store version: {data.get('version')}")
                return

            for td in data.get("tenants", []):
                tenant = Tenant.from_dict(td)
                self._tenants[tenant.tenant_id] = tenant
                self._key_to_tenant[tenant.api_key_hash] = tenant.tenant_id

            logger.info(f"Loaded {len(self._tenants)} tenants from {self._persist_path}")
        except Exception as e:
            logger.error(f"Failed to load tenants: {e}")
