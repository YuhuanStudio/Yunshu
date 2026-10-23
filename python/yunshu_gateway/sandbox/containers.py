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

_ID = re.compile(r"^container_[0-9a-f]{24}$")
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

    def _write_meta(self, cid: str, created: float, last: float) -> None:
        tmp = self._meta(cid).with_suffix(".tmp")
        tmp.write_text(json.dumps({"created_at": created, "last_active": last}))
        os.replace(tmp, self._meta(cid))

    def _load(self, cid: str) -> Container:
        try:
            m = json.loads(self._meta(cid).read_text())
        except (OSError, ValueError):
            raise ContainerError(
                "not_found", f"Container '{cid}' was not found."
            ) from None
        last = float(m.get("last_active", 0))
        return Container(
            cid,
            self.root / cid,
            float(m.get("created_at", last)),
            last,
            last + self.ttl_s,
        )

    # ── lifecycle ────────────────────────────────────────────────────────────
    def create(self) -> Container:
        with self._lock:
            self.gc()
            cid = "container_" + secrets.token_hex(12)
            base = self.root / cid
            base.mkdir(mode=0o700)
            for s in SUBDIRS:
                (base / s).mkdir(mode=0o700)
            now = self.clock()
            self._write_meta(cid, now, now)
            return self._load(cid)

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
