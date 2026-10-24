"""Per-container workspaces with idle expiry and a total-size cap.

A container is a directory ``<root>/<id>/`` (``work/``, ``home/``, ``tmp/``, ``out/``) that the sandbox
may write; its bookkeeping lives outside it in ``<root>/.meta/<id>.json`` so sandboxed code cannot
extend its own lifetime. Ids are validated before they touch a path.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from yunshu_engine import settings

# "container_" ids belong to the Anthropic API, "cntr_" ids to OpenAI's /v1/containers; both name one
# directory kind, so a container made through either API can be used through the other.
_ID = re.compile(r"^(container|cntr)_[0-9a-f]{24}$")
PREFIXES = ("container", "cntr")
MEMORY_LIMITS = {"1g": 1024, "4g": 4096, "16g": 16384, "64g": 65536}
SUBDIRS = ("work", "home", "tmp", "out")
MAX_TOTAL_BYTES = 2 * 1024**3


class ContainerError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # "not_found" | "expired" | "invalid"
        self.message = message


@dataclass(frozen=True)
class Container:
    id: str
    path: Path
    created_at: float
    last_active: float
    expires_at: float
    name: str | None = None
    ttl_s: float = 0.0  # this container's idle lifetime
    # OpenAI memory_limit label: a ceiling under the server's setting
    memory_limit: str | None = None
    # None: the server's allowlist; (): no network
    network_allow: tuple[str, ...] | None = None
    # paths (relative to work/) put there through an API
    user_files: tuple[str, ...] = ()
    owner: str = ""  # the API key / actor that created it ("" = unowned)

    @property
    def work(self) -> Path:
        return self.path / "work"

    @property
    def home(self) -> Path:
        return self.path / "home"

    @property
    def tmp(self) -> Path:
        return self.path / "tmp"

    @property
    def out(self) -> Path:
        return self.path / "out"


def owner_ok(c: Container, actor: str, admin: bool) -> bool:
    """May ``actor`` use this container? Unowned ones are open; admins see everything."""
    return not c.owner or admin or c.owner == actor


def valid_id(value: str) -> bool:
    return bool(_ID.match(value or ""))


def _dir_size(p: Path) -> int:
    total = 0
    for dp, _dn, fn in os.walk(p, followlinks=False):
        for f in fn:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dp, f)).st_size
    return total


class ContainerManager:
    def __init__(
        self,
        root: Path,
        ttl_s: float,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        clock=time.time,
    ):
        self.root = root
        self.ttl_s = ttl_s
        self.max_total_bytes = max_total_bytes
        self.clock = clock
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            os.chmod(self.root, 0o700)
        (self.root / ".meta").mkdir(exist_ok=True, mode=0o700)

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def _meta(self, cid: str) -> Path:
        return self.root / ".meta" / f"{cid}.json"

    def _write_meta(self, cid: str, created: float, last: float, **extra) -> None:
        old = self._read_meta(cid) if extra.pop("keep", True) else {}
        data = {**old, **extra, "created_at": created, "last_active": last}
        tmp = self._meta(cid).with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._meta(cid))

    def _read_meta(self, cid: str) -> dict:
        try:
            m = json.loads(self._meta(cid).read_text())
        except (OSError, ValueError):
            return {}
        return m if isinstance(m, dict) else {}

    def _load(self, cid: str) -> Container:
        try:
            m = json.loads(self._meta(cid).read_text())
        except (OSError, ValueError):
            raise ContainerError(
                "not_found", f"Container '{cid}' was not found."
            ) from None
        if not isinstance(m, dict):
            raise ContainerError("not_found", f"Container '{cid}' was not found.")
        last = float(m.get("last_active", 0))
        # the server's setting is a ceiling: lowering YUNSHU_CODE_EXECUTION_TTL_MIN shortens every
        # container, raising it never lengthens one that asked for less
        asked = float(m.get("ttl_s") or 0)
        ttl = min(asked, self.ttl_s) if asked else self.ttl_s
        net = m.get("network_allow")
        return Container(
            cid,
            self.root / cid,
            float(m.get("created_at", last)),
            last,
            last + ttl,
            name=m.get("name"),
            ttl_s=ttl,
            memory_limit=m.get("memory_limit"),
            network_allow=tuple(net) if isinstance(net, list) else None,
            user_files=tuple(m.get("user_files") or ()),
            owner=str(m.get("owner") or ""),
        )

    # ── lifecycle ────────────────────────────────────────────────────────────
    def create(
        self,
        *,
        prefix: str = "container",
        name: str | None = None,
        ttl_s: float | None = None,
        memory_limit: str | None = None,
        network_allow: list[str] | None = None,
        owner: str = "",
    ) -> Container:
        if prefix not in PREFIXES:
            raise ContainerError("invalid", f"unknown container id prefix '{prefix}'")
        with self._lock:
            self.gc()
            cid = prefix + "_" + secrets.token_hex(12)
            base = self.root / cid
            base.mkdir(mode=0o700)
            for s in SUBDIRS:
                (base / s).mkdir(mode=0o700)
            now = self.clock()
            extra: dict = {}
            if name is not None:
                extra["name"] = name
            if ttl_s is not None:
                extra["ttl_s"] = float(ttl_s)
            if memory_limit is not None:
                extra["memory_limit"] = memory_limit
            if network_allow is not None:
                extra["network_allow"] = list(network_allow)
            if owner:
                extra["owner"] = owner
            self._write_meta(cid, now, now, keep=False, **extra)
            return self._load(cid)

    def list(self, prefix: str | None = None) -> list[Container]:
        """Live containers, oldest first (expired ones are removed on the way)."""
        with self._lock:
            self.gc()
            out = []
            for d in self.root.iterdir():
                if d.name.startswith(".") or not valid_id(d.name):
                    continue
                if prefix and not d.name.startswith(prefix + "_"):
                    continue
                with contextlib.suppress(ContainerError):
                    out.append(self._load(d.name))
            return sorted(out, key=lambda c: (c.created_at, c.id))

    def add_user_file(self, cid: str, rel: str) -> None:
        with self._lock:
            c = self._load(cid)
            if rel not in c.user_files:
                self._write_meta(
                    cid,
                    c.created_at,
                    c.last_active,
                    user_files=[*c.user_files, rel],
                )

    def drop_user_file(self, cid: str, rel: str) -> None:
        with self._lock:
            c = self._load(cid)
            if rel in c.user_files:
                self._write_meta(
                    cid,
                    c.created_at,
                    c.last_active,
                    user_files=[f for f in c.user_files if f != rel],
                )

    def get(self, cid: str, *, touch: bool = True) -> Container:
        if not valid_id(cid):
            raise ContainerError("invalid", f"'{cid}' is not a container id.")
        with self._lock:
            c = self._load(cid)
            if c.expires_at <= self.clock() or not c.path.is_dir():
                self.delete(cid)
                raise ContainerError(
                    "expired",
                    f"Container '{cid}' has expired; send the request without the container parameter.",
                )
            if touch:
                self._write_meta(cid, c.created_at, self.clock())
                c = self._load(cid)
            return c

    def delete(self, cid: str) -> None:
        if not valid_id(cid):
            return
        with self._lock:
            for hook in list(_delete_hooks):
                with contextlib.suppress(Exception):
                    hook(cid)
            shutil.rmtree(self.root / cid, ignore_errors=True)
            with contextlib.suppress(OSError):
                self._meta(cid).unlink()

    def gc(self) -> int:
        """Delete expired containers, orphans, then the oldest until the cap holds. Returns count."""
        removed = 0
        with self._lock:
            now = self.clock()
            live: list[tuple[float, str]] = []
            for d in self.root.iterdir():
                if d.name.startswith(".") or not valid_id(d.name):
                    continue
                try:
                    c = self._load(d.name)
                except ContainerError:
                    self.delete(d.name)  # no bookkeeping: left over from a crash
                    removed += 1
                    continue
                if c.expires_at <= now:
                    self.delete(d.name)
                    removed += 1
                else:
                    live.append((c.last_active, d.name))
            for m in (self.root / ".meta").glob("container_*.json"):
                if not (self.root / m.stem).is_dir():
                    with contextlib.suppress(OSError):
                        m.unlink()
            sizes = {n: _dir_size(self.root / n) for _, n in live}
            total = sum(sizes.values())
            for _last, name in sorted(live):
                if total <= self.max_total_bytes:
                    break
                self.delete(name)
                total -= sizes[name]
                removed += 1
        return removed

    def usage_bytes(self, cid: str) -> int:
        return _dir_size(self.root / cid)


_delete_hooks: list = []  # called with the id before a container's files are removed


def on_delete(hook) -> None:
    if hook not in _delete_hooks:
        _delete_hooks.append(hook)


_mgr: ContainerManager | None = None
_mgr_lock = threading.Lock()


def get_manager() -> ContainerManager:
    """Process-wide manager; rebuilt when the configured root or TTL changes."""
    global _mgr
    root = Path(str(settings.get("YUNSHU_CODE_EXECUTION_ROOT"))).expanduser()
    ttl = float(settings.get("YUNSHU_CODE_EXECUTION_TTL_MIN")) * 60.0
    with _mgr_lock:
        if _mgr is None or _mgr.root != root:
            _mgr = ContainerManager(root, ttl)
        else:
            _mgr.ttl_s = ttl
        return _mgr


def reset_manager() -> None:
    global _mgr
    _mgr = None
