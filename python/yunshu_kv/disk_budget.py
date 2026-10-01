"""One disk budget per SSD-cache root.

Both SSD tiers (the APC tier and the text engine's) keep one directory per checkpoint
namespace under a root. A cap that applies per namespace lets the root grow without bound:
every older fingerprint, every other model leaves its own full-size namespace behind. This
module is the shared policy:

- **global cap**: everything under the root (all namespaces together) is bounded; over the
  cap the least recently used files go first, across namespaces;
- **stale namespaces**: a namespace not touched for ``stale_days`` days, or whose checkpoint
  no longer exists / no longer matches its recorded signature, is removed before anything
  else;
- **free-space reserve**: no write is allowed to leave the volume with less than
  ``max(reserve_pct of the volume, reserve_bytes)`` free, and the effective cap is
  ``min(configured cap, what the reserve leaves the root)``, re-evaluated as space changes;
- **write failures** (ENOSPC, EIO, ...): the write is dropped, one warning is logged, and
  spilling pauses until free space is back; the request that triggered it never sees it.

Nothing here touches what a cache file contains.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUFFIX = ".safetensors"
MARKER = "namespace.json"
ROOT_NS = "."  # files that sit directly under a bucket of the root (no model namespace)
_BUCKET = re.compile(r"^[0-9a-f]$|^hybrid_snapshots$")
# Anything a namespace directory may legitimately hold; a directory with other files is
# never removed wholesale.
_NS_EXTRA = ("index.db", "index.db-wal", "index.db-shm", "index.json", MARKER)
GIB = 1 << 30
# A temp file younger than this may belong to a write in flight.
TMP_GRACE_S = 600.0


def disk_usage(path):  # indirection so tests can simulate a full volume
    return shutil.disk_usage(path)


def _probe(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def reserve_bytes(path: Path, pct: float, min_bytes: int) -> int:
    """Free space that must remain on ``path``'s volume: max(pct of the volume, min_bytes)."""
    total = disk_usage(_probe(path)).total
    return int(max(total * pct / 100.0, min_bytes))


def is_tmp(name: str) -> bool:
    return name.endswith(".tmp") or ".tmp." in name


def is_enospc(exc: BaseException) -> bool:
    import errno

    return isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT)


def sweep_tmp(directory: Path, stem: str | None = None) -> None:
    """Remove temp siblings a failed write may have left (``<stem>.*`` that are temps, or
    for the text tier ``<stem>.safetensors.*.tmp``)."""
    with contextlib.suppress(OSError):
        pattern = f"{stem}.*" if stem else "*"
        for p in directory.glob(pattern):
            if p.is_file() and is_tmp(p.name):
                p.unlink(missing_ok=True)


@dataclass
class _Entry:
    path: Path
    size: int
    mtime: float
    ns: str


@dataclass
class BudgetStatus:
    root: str
    used: int = 0
    namespaces: dict[str, int] = field(default_factory=dict)
    cap: int | None = None
    effective_cap: int | None = None
    free: int = 0
    reserve: int = 0
    paused: bool = False


def write_marker(ns_dir: Path, model_path: str | None, signature: str | None) -> None:
    """Record which checkpoint a namespace was written for, so ``gc`` can tell it is dead."""
    with contextlib.suppress(OSError):
        ns_dir.mkdir(parents=True, exist_ok=True)
        (ns_dir / MARKER).write_text(
            json.dumps({"model_path": model_path, "signature": signature})
        )


def namespace_orphaned(ns_dir: Path) -> str | None:
    """Why this namespace can no longer match a local checkpoint, or None when it can
    (or when that cannot be told: no marker, a hub id)."""
    try:
        marker = json.loads((ns_dir / MARKER).read_text())
        path = marker.get("model_path")
        sig = marker.get("signature")
    except (OSError, ValueError, AttributeError):
        return None
    if not path or not str(path).startswith(("/", "~")):
        return None
    local = Path(str(path)).expanduser()
    if not local.exists():
        return "checkpoint gone"
    if sig:
        from .fingerprint import checkpoint_fingerprint

        try:
            if checkpoint_fingerprint(str(path), digest_size=8) != sig:
                return "checkpoint changed"
        except Exception:
            return None
    return None


def _ns_of(root: Path, p: Path) -> str:
    try:
        parts = p.relative_to(root).parts
    except ValueError:
        return ROOT_NS
    if len(parts) < 2 or _BUCKET.match(parts[0]):
        return ROOT_NS
    return parts[0]


def scan_root(root: Path) -> list[_Entry]:
    out: list[_Entry] = []
    if not root.is_dir():
        return out
    for p in root.rglob(f"*{SUFFIX}"):
        try:
            if p.is_symlink() or not p.is_file() or is_tmp(p.name):
                continue
            st = p.stat()
        except OSError:
            continue
        out.append(_Entry(p, st.st_size, st.st_mtime, _ns_of(root, p)))
    return out


def namespace_last_used(root: Path, ns: str, entries: list[_Entry]) -> float:
    ts = [e.mtime for e in entries if e.ns == ns]
    with contextlib.suppress(OSError):
        ts.append((root / ns).stat().st_mtime)
    return max(ts, default=0.0)


def removable_namespace(ns_dir: Path) -> bool:
    """True when the directory holds only things a cache namespace creates."""
    try:
        for p in ns_dir.rglob("*"):
            if p.is_dir():
                continue
            if p.name.endswith(SUFFIX) or is_tmp(p.name) or p.name in _NS_EXTRA:
                continue
            return False
    except OSError:
        return False
    return True


class DiskBudget:
    """The budget of one cache root; shared by every store that writes under it."""

    SCAN_TTL_S = 15.0
    PRUNE_EVERY_S = 60.0

    def __init__(
        self,
        root: Path | str,
        *,
        cap_bytes: int = 0,
        reserve_pct: float = 10.0,
        reserve_min_bytes: int = 20 * GIB,
        stale_days: float = 7.0,
        recheck_s: float = 30.0,
        label: str = "SSD cache",
    ):
        self.root = Path(root).expanduser()
        self.cap_bytes = int(cap_bytes)
        self.reserve_pct = float(reserve_pct)
        self.reserve_min_bytes = int(reserve_min_bytes)
        self.stale_days = float(stale_days)
        self.recheck_s = float(recheck_s)
        self.label = label
        self._lock = threading.RLock()
        self._entries: list[_Entry] | None = None
        self._scanned = 0.0
        self._paused_at: float | None = None
        self._warned = False
        self._last_prune = float("-inf")
        # namespace -> callable(path) that drops a file of a live store (index + unlink)
        self._owners: dict[str, Callable[[Path], bool]] = {}
        self._protected: dict[str, Callable[[], set[Path]]] = {}
        self.evictions = 0
        self.dropped_writes = 0

    # ── bookkeeping ─────────────────────────────────────────────────────
    def register_owner(
        self,
        ns: str,
        evict: Callable[[Path], bool] | None = None,
        protected: Callable[[], set[Path]] | None = None,
    ) -> None:
        with self._lock:
            if evict is not None:
                self._owners[ns] = evict
            if protected is not None:
                self._protected[ns] = protected

    def _scan(self, force: bool = False) -> list[_Entry]:
        with self._lock:
            now = time.monotonic()
            if force or self._entries is None or now - self._scanned > self.SCAN_TTL_S:
                self._entries = scan_root(self.root)
                self._scanned = now
            return self._entries

    def note_written(self, path: Path, size: int | None = None) -> None:
        with self._lock:
            self._warned = False  # a write landed: the next failure warns again
            if self._entries is None:
                return
            if size is None:
                try:
                    size = path.stat().st_size
                except OSError:
                    return
            self._entries = [e for e in self._entries if e.path != path]
            self._entries.append(
                _Entry(path, int(size), time.time(), _ns_of(self.root, path))
            )

    def used(self) -> int:
        return sum(e.size for e in self._scan())

    def reserve(self) -> int:
        return reserve_bytes(self.root, self.reserve_pct, self.reserve_min_bytes)

    def free(self) -> int:
        return int(disk_usage(_probe(self.root)).free)

    def effective_cap(self) -> int | None:
        """min(configured cap, what the free-space reserve leaves this root). None when
        neither bounds it (cap 0 and the volume cannot be read)."""
        limits = []
        if self.cap_bytes > 0:
            limits.append(self.cap_bytes)
        with contextlib.suppress(OSError):
            limits.append(max(0, self.used() + self.free() - self.reserve()))
        return min(limits) if limits else None

    # ── write admission ─────────────────────────────────────────────────
    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def allow_write(self, size: int) -> bool:
        """May a file of ``size`` bytes be written now? False while paused (until free
        space is back) or when it would eat into the reserve."""
        try:
            free, reserve = self.free(), self.reserve()
        except OSError:
            return not self.paused
        ok = free - int(size) >= reserve
        with self._lock:
            if self._paused_at is not None:
                if ok and time.monotonic() - self._paused_at >= self.recheck_s:
                    self._paused_at = None
                    logger.info("%s: space recovered, spilling resumed", self.label)
                else:
                    ok = False
            if not ok:
                self.dropped_writes += 1
                if not self._warned and self._paused_at is None:
                    self._warned = True
                    logger.warning(
                        "%s: write skipped, %.1f GiB free would fall under the %.1f GiB "
                        "reserve; checkpoints are not spilled until space is back",
                        self.label,
                        free / GIB,
                        reserve / GIB,
                    )
        return ok

    def record_failure(self, exc: BaseException) -> None:
        """A write failed: pause spilling (one warning, no traceback)."""
        with self._lock:
            self.dropped_writes += 1
            self._paused_at = time.monotonic()
            if not self._warned:
                self._warned = True
                logger.warning(
                    "%s: write failed (%s); dropped, spilling paused until space is back",
                    self.label,
                    exc,
                )

    # ── eviction ────────────────────────────────────────────────────────
    def _drop(self, e: _Entry) -> bool:
        owner = self._owners.get(e.ns)
        try:
            if owner is not None:
                if not owner(e.path):
                    return False
            else:
                e.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("%s: cannot remove %s (%s)", self.label, e.path.name, exc)
            return False
        self.evictions += 1
        return True

    def prune_stale(
        self, *, now: float | None = None, keep: AbstractSet[str] = frozenset()
    ):
        """Remove namespaces unused for ``stale_days`` or orphaned; returns
        ``[(namespace, reason, bytes)]``."""
        now = time.time() if now is None else now
        removed = []
        with self._lock:
            entries = self._scan(force=True)
            names = {e.ns for e in entries}
            with contextlib.suppress(OSError):
                names.update(
                    p.name
                    for p in self.root.iterdir()
                    if p.is_dir() and not _BUCKET.match(p.name)
                )
            for ns in sorted(names - {ROOT_NS} - set(keep)):
                d = self.root / ns
                reason = namespace_orphaned(d)
                if reason is None and self.stale_days > 0:
                    last = namespace_last_used(self.root, ns, entries)
                    if now - last >= self.stale_days * 86400:
                        reason = f"unused {self.stale_days:g}+ days"
                if reason is None or not d.is_dir() or not removable_namespace(d):
                    continue
                size = sum(e.size for e in entries if e.ns == ns)
                shutil.rmtree(d, ignore_errors=True)
                removed.append((ns, reason, size))
                self.evictions += sum(1 for e in entries if e.ns == ns)
            if removed:
                gone = {r[0] for r in removed}
                self._entries = [e for e in entries if e.ns not in gone]
                logger.info(
                    "%s: removed %d stale namespace(s), %.1f GiB",
                    self.label,
                    len(removed),
                    sum(r[2] for r in removed) / GIB,
                )
        return removed

    def enforce(
        self,
        *,
        keep: AbstractSet[str] = frozenset(),
        protect: AbstractSet[Path] = frozenset(),
    ):
        """Stale namespaces first, then least recently used files across all namespaces
        until the root is within its effective cap. Returns bytes freed."""
        with self._lock:
            before = self.used()
            if time.monotonic() - self._last_prune >= self.PRUNE_EVERY_S:
                self._last_prune = time.monotonic()
                self.prune_stale(keep=keep)
            cap = self.effective_cap()
            if cap is None:
                return before - self.used()
            entries = self._scan()
            total = sum(e.size for e in entries)
            if total > cap:
                busy = set(protect)
                for fn in self._protected.values():
                    with contextlib.suppress(Exception):
                        busy |= fn()
                gone: set[Path] = set()
                for e in sorted(entries, key=lambda e: e.mtime):
                    if total <= cap:
                        break
                    if e.path in busy:
                        continue
                    if self._drop(e):
                        total -= e.size
                        gone.add(e.path)
                if gone:
                    self._entries = [x for x in entries if x.path not in gone]
            return before - total

    def status(self) -> BudgetStatus:
        entries = self._scan(force=True)
        ns: dict[str, int] = {}
        for e in entries:
            ns[e.ns] = ns.get(e.ns, 0) + e.size
        try:
            free, reserve = self.free(), self.reserve()
        except OSError:
            free = reserve = 0
        return BudgetStatus(
            root=str(self.root),
            used=sum(ns.values()),
            namespaces=ns,
            cap=self.cap_bytes or None,
            effective_cap=self.effective_cap(),
            free=free,
            reserve=reserve,
            paused=self.paused,
        )


_REGISTRY: dict[str, DiskBudget] = {}
_REG_LOCK = threading.Lock()


def budget_for(root: Path | str, *, cap_bytes: int, label: str) -> DiskBudget:
    """The shared budget of ``root`` (one per root per process), configured from settings."""
    from yunshu_engine import settings

    key = str(Path(root).expanduser().resolve())
    with _REG_LOCK:
        b = _REGISTRY.get(key)
        if b is None:
            b = DiskBudget(
                root,
                cap_bytes=cap_bytes,
                reserve_pct=float(settings.get("YUNSHU_CACHE_RESERVE_PCT")),
                reserve_min_bytes=int(
                    float(settings.get("YUNSHU_CACHE_RESERVE_GB")) * GIB
                ),
                stale_days=float(settings.get("YUNSHU_CACHE_STALE_DAYS")),
                label=label,
            )
            _REGISTRY[key] = b
        else:
            b.cap_bytes = int(cap_bytes)
        return b
