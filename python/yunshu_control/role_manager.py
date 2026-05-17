from __future__ import annotations
"""Yunshu Role-Based Access Control — API key permissions and SLO classes.

Implements a simple RBAC system:
- Roles: admin, developer, user (with escalating permissions)
- Per-role rate limits and model access
- SLO classes: best_effort, standard, premium (affects scheduling priority)
- JSON file persistence for API keys across restarts
"""

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Role(Enum):
    ADMIN = auto()
    DEVELOPER = auto()
    USER = auto()


class SLOClass(Enum):
    BEST_EFFORT = auto()
    STANDARD = auto()
    PREMIUM = auto()


@dataclass
class RolePermissions:
    """Permissions associated with a role."""
    can_load_models: bool = False
    can_unload_models: bool = False
    can_register_models: bool = False
    can_manage_tokens: bool = False
    can_view_admin: bool = False
    can_benchmark: bool = False
    can_manage_models: bool = False  # LoRA adapter management, model settings
    can_view_system: bool = False  # System internals: queue stats, hardware profile
    max_models: int = 1
    allowed_model_patterns: list[str] = field(default_factory=lambda: ["*"])


ROLE_PERMISSIONS = {
    Role.ADMIN: RolePermissions(
        can_load_models=True,
        can_unload_models=True,
        can_register_models=True,
        can_manage_tokens=True,
        can_view_admin=True,
        can_benchmark=True,
        can_manage_models=True,
        can_view_system=True,
        max_models=100,
        allowed_model_patterns=["*"],
    ),
    Role.DEVELOPER: RolePermissions(
        can_load_models=True,
        can_unload_models=True,
        can_register_models=False,
        can_manage_tokens=False,
        can_view_admin=True,
        can_benchmark=True,
        can_manage_models=True,
        can_view_system=True,
        max_models=10,
        allowed_model_patterns=["*"],
    ),
    Role.USER: RolePermissions(
        can_load_models=False,
        can_unload_models=False,
        can_register_models=False,
        can_manage_tokens=False,
        can_view_admin=False,
        can_benchmark=False,
        can_manage_models=False,
        can_view_system=False,
        max_models=1,
        allowed_model_patterns=["*"],
    ),
}


@dataclass
class APIKey:
    """An API key with associated role and SLO class."""
    key_hash: str
    name: str
    role: Role = Role.USER
    slo_class: SLOClass = SLOClass.STANDARD
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_active: bool = True
    created_by: Optional[str] = None
    key_prefix: str = ""  # First 12 chars of raw key for revocation

    # Per-key rate limits (override role defaults)
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None

    def has_permission(self, permission: str) -> bool:
        perms = ROLE_PERMISSIONS.get(self.role)
        if perms is None:
            return False
        return getattr(perms, permission, False)

    def can_access_model(self, model_id: str) -> bool:
        perms = ROLE_PERMISSIONS.get(self.role)
        if perms is None:
            return False
        import fnmatch
        for pattern in perms.allowed_model_patterns:
            if fnmatch.fnmatch(model_id, pattern):
                return True
        return False

    def get_scheduling_priority(self) -> int:
        """Get scheduling priority based on SLO class."""
        priorities = {
            SLOClass.BEST_EFFORT: 0,
            SLOClass.STANDARD: 5,
            SLOClass.PREMIUM: 10,
        }
        return priorities.get(self.slo_class, 0)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class RBACManager:
    """Manages API keys with role-based access control.

    Supports optional JSON file persistence. When persist_path is set,
    keys are automatically saved to disk after any mutation and loaded
    on startup.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._keys: dict[str, APIKey] = {}  # key_hash → APIKey
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path:
            self._load()

    def _load(self) -> None:
        """Load API keys from JSON file."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for entry in data.get("keys", []):
                try:
                    api_key = APIKey(
                        key_hash=entry["key_hash"],
                        name=entry["name"],
                        role=Role[entry["role"]],
                        slo_class=SLOClass[entry["slo_class"]],
                        created_at=entry.get("created_at", time.time()),
                        expires_at=entry.get("expires_at"),
                        is_active=entry.get("is_active", True),
                        created_by=entry.get("created_by"),
                        key_prefix=entry.get("key_prefix", ""),
                        requests_per_minute=entry.get("requests_per_minute"),
                        tokens_per_minute=entry.get("tokens_per_minute"),
                    )
                    self._keys[api_key.key_hash] = api_key
                except (KeyError, ValueError):
                    logger.warning(f"Skipping invalid key entry: {entry.get('name', 'unknown')}")
            logger.info(f"RBAC: loaded {len(self._keys)} keys from {self._persist_path}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"RBAC: failed to load keys from {self._persist_path}: {e}")

    def _save(self) -> None:
        """Save API keys to JSON file."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "keys": [
                    {
                        "key_hash": k.key_hash,
                        "name": k.name,
                        "role": k.role.name,
                        "slo_class": k.slo_class.name,
                        "created_at": k.created_at,
                        "expires_at": k.expires_at,
                        "is_active": k.is_active,
                        "created_by": k.created_by,
                        "key_prefix": k.key_prefix,
                        "requests_per_minute": k.requests_per_minute,
                        "tokens_per_minute": k.tokens_per_minute,
                    }
                    for k in self._keys.values()
                ]
            }
            self._persist_path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning(f"RBAC: failed to save keys to {self._persist_path}: {e}")

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(
        self,
        name: str,
        role: Role = Role.USER,
        slo_class: SLOClass = SLOClass.STANDARD,
        expires_days: Optional[int] = None,
        created_by: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """Create a new API key. Returns (raw_key, APIKey)."""
        raw_key = f"ys_{secrets.token_hex(24)}"
        key_hash = self.hash_key(raw_key)

        expires_at = None
        if expires_days is not None:
            expires_at = time.time() + expires_days * 86400

        api_key = APIKey(
            key_hash=key_hash,
            name=name,
            role=role,
            slo_class=slo_class,
            expires_at=expires_at,
            created_by=created_by,
            key_prefix=raw_key[:12],
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
        self._keys[key_hash] = api_key
        self._save()
        return raw_key, api_key

    def authenticate(self, raw_key: str) -> Optional[APIKey]:
        """Authenticate an API key. Returns APIKey or None."""
        key_hash = self.hash_key(raw_key)
        api_key = self._keys.get(key_hash)
        if api_key is None:
            return None
        if not api_key.is_active:
            return None
        if api_key.is_expired():
            return None
        return api_key

    def revoke_key(self, key_prefix: str) -> int:
        """Revoke keys matching a name or key prefix. Returns count revoked."""
        revoked = 0
        for key_hash, api_key in list(self._keys.items()):
            if api_key.name == key_prefix or api_key.key_prefix.startswith(key_prefix):
                api_key.is_active = False
                revoked += 1
        if revoked:
            self._save()
        return revoked

    def delete_key(self, key_prefix: str) -> int:
        """Delete keys matching a name or key prefix. Returns count deleted."""
        to_delete = [
            h for h, k in self._keys.items()
            if k.name == key_prefix or k.key_prefix.startswith(key_prefix)
        ]
        for h in to_delete:
            del self._keys[h]
        if to_delete:
            self._save()
        return len(to_delete)

    def list_keys(self) -> list[dict]:
        """List all API keys (masked)."""
        return [
            {
                "name": k.name,
                "role": k.role.name,
                "slo_class": k.slo_class.name,
                "is_active": k.is_active,
                "expires_at": k.expires_at,
                "created_by": k.created_by,
            }
            for k in self._keys.values()
        ]

    def get_permissions(self, role: Role) -> RolePermissions:
        return ROLE_PERMISSIONS.get(role, RolePermissions())
