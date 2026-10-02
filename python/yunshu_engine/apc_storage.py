# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""N-level storage for the APC checkpoints: internal SSD -> external SSD -> HDD -> NAS -> recompute.

The first directory is the primary store (``SpillDiskStore``: upstream safetensors files, the async
writer, the root-wide ``DiskBudget``). Further directories (``YUNSHU_VLM_APC_DISK_TIERS``) are lower
tiers, each on its own volume with its own budget and free-space reserve. What the module adds:

- **Device profiling** (:func:`probe_device`): sequential read / write bandwidth and small-read
  latency are measured on the volume (``F_NOCACHE``, ``fsync``) and cached per mount point with a
  re-probe interval. Nothing assumes a speed; a tier without a profile is skipped.
- **Ordering** by measured read bandwidth, fastest first.
- **Placement**: a spill goes to the primary store. A background mover (never on the request path)
  demotes least-recently-used checkpoints of this model downwards when a tier passes its budget, to
  the first lower tier that is available, has room, and in which a restore still beats a re-prefill.
  A checkpoint no tier can hold is left to the budgets' LRU eviction (recompute).
- **Cost model**: a candidate checkpoint is used only if ``latency + bytes / read_bw`` (+ decode)
  is below re-prefilling the tokens it saves at the observed prefill speed; the cheapest of all
  tiers' candidates (restore + prefill of the rest) wins.
- **Encoding per tier** (``auto``): a lower tier stores a file raw, or zstd-compressed (lossless,
  byte-plane shuffle for 2-byte tensors) when the measured gain in effective read bandwidth pays
  for the decode time. Raw on fast tiers.
- **Availability and remount**: a tier whose volume is gone (unplugged, NAS down; stat timeout) is
  skipped; when it returns it is re-scanned and every file is validated again (header, size,
  checksums); an unreadable or corrupt file is deleted and the lookup recomputes.

Everything here works on files; no MLX array is touched off the caller's thread.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import shutil
import struct
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .apc_manager import SpillDiskStore

logger = logging.getLogger(__name__)

MIB = 1 << 20
GIB = 1 << 30
CHUNK = 4 * MIB
PROBE_MB = 128
PROFILE_TTL_S = 24 * 3600.0
ENCODED_SUFFIX = ".yscx"
RAW_SUFFIX = ".safetensors"
_MAGIC = b"YSCX1\0\0\0"
_TRAILER = struct.Struct("<QI8s")  # header length, header crc32, magic
OVERSHOOT_DELETE = (
    1.5  # over this multiple of its cap a tier deletes instead of waiting for the mover
)
DEFAULT_PREFILL_TPS = (
    1500.0  # until a real prefill has been observed (conservative: high)
)
_TWO_BYTE = {"BF16", "F16"}


def _delete_above(cap_bytes: int, file_bytes: int) -> float:
    """Usage over which a tier deletes a file the mover has not yet demoted: 1.5x its cap, but
    always room for a couple of files even when one file is bigger than the cap."""
    return max(OVERSHOOT_DELETE * cap_bytes, cap_bytes + 2 * file_bytes)


# ── tier specs ───────────────────────────────────────────────────────────
@dataclass
class TierSpec:
    path: Path
    cap_gib: float | None = None
    sim: tuple[float, float] | None = (
        None  # (bytes/s, seconds per op): a simulated slow device
    )

    @classmethod
    def parse(cls, text: str) -> TierSpec:
        """``PATH[@GiB][!sim=MBps/ms]`` (the simulation clause is for tests: it throttles every
        read / write / probe of the tier to that bandwidth and latency)."""
        sim = None
        if "!sim=" in text:
            text, _, s = text.partition("!sim=")
            bw, _, ms = s.partition("/")
            sim = (float(bw) * 1e6, float(ms or 0) / 1e3)
        cap = None
        head, sep, tail = text.rpartition("@")
        if sep and tail.replace(".", "", 1).isdigit():
            text, cap = head, float(tail)
        return cls(Path(text.strip()).expanduser(), cap, sim)


def parse_tiers(text: str | None) -> list[TierSpec]:
    return [TierSpec.parse(p) for p in (text or "").split(",") if p.strip()]


# ── device profile ───────────────────────────────────────────────────────
@dataclass
class DeviceProfile:
    mount: str
    read_bps: float
    write_bps: float
    latency_s: float
    probed_at: float
    simulated: bool = False

    def restore_s(
        self, nbytes: float, decode_bps: float = 0.0, ratio: float = 1.0
    ) -> float:
        """Seconds to bring ``nbytes`` (uncompressed) back from this device."""
        t = self.latency_s + nbytes / max(ratio, 1.0) / max(self.read_bps, 1.0)
        if ratio > 1.0 and decode_bps > 0:
            t += nbytes / decode_bps
        return t


def mount_of(path: Path) -> Path:
    p = Path(path).expanduser()
    while not p.exists() and p != p.parent:
        p = p.parent
    with contextlib.suppress(OSError):
        p = p.resolve()
    while not os.path.ismount(p) and p != p.parent:
        p = p.parent
    return p


def ensure_root(path: Path) -> bool:
    """Create the tier directory when its volume is there. A path under a mount directory
    (``/Volumes/NAME``, ``/mnt/NAME``, ``/media/..``) whose volume is not mounted is never created:
    that would put the cache on the boot disk under the name of a missing device."""
    p = Path(path).expanduser()
    if p.is_dir():
        return True
    parts = p.parts
    for i, part in enumerate(parts[:-1]):
        if part in ("Volumes", "mnt", "media") and i == 1 and len(parts) > 2:
            if not Path(*parts[: i + 2]).is_dir():
                return False
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return True


def device_name(path: Path) -> str:
    m = mount_of(path)
    return "internal" if str(m) == "/" else (m.name or str(m))


def _nocache(fd: int) -> None:
    with contextlib.suppress(Exception):
        import fcntl

        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)


def _pace(
    nbytes: int, t0: float, sim: tuple[float, float] | None, ops: int = 1
) -> None:
    """Sleep so a simulated device takes ``ops * latency + nbytes / bandwidth`` since ``t0``."""
    if sim is None:
        return
    bw, lat = sim
    due = t0 + ops * lat + nbytes / bw
    left = due - time.perf_counter()
    if left > 0:
        time.sleep(left)


def probe_device(
    path: Path, *, mb: int = PROBE_MB, sim: tuple[float, float] | None = None
) -> DeviceProfile:
    """Measure sequential write / read bandwidth and small-read latency of ``path``'s volume."""
    root = Path(path).expanduser()
    if not ensure_root(root):
        raise OSError(f"{root}: volume not mounted")
    d = root / f".yunshu-probe-{os.getpid()}-{threading.get_ident()}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        block = os.urandom(
            8 * MIB
        )  # incompressible, so a compressing filesystem cannot flatter it
        n = max(1, mb // 8)
        big = d / "big"
        t0 = time.perf_counter()
        fd = os.open(big, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            _nocache(fd)
            for _ in range(n):
                os.write(fd, block)
            os.fsync(fd)
        finally:
            os.close(fd)
        _pace(n * 8 * MIB, t0, sim)
        write_bps = n * 8 * MIB / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        fd = os.open(big, os.O_RDONLY)
        try:
            _nocache(fd)
            got = 0
            while True:
                b = os.read(fd, 8 * MIB)
                if not b:
                    break
                got += len(b)
        finally:
            os.close(fd)
        _pace(got, t0, sim)
        read_bps = got / (time.perf_counter() - t0)
        for i in range(8):
            f = d / f"s{i}"
            fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                _nocache(fd)
                os.write(fd, os.urandom(4096))
                os.fsync(fd)
            finally:
                os.close(fd)
        lat = []
        for i in range(8):
            t0 = time.perf_counter()
            fd = os.open(d / f"s{i}", os.O_RDONLY)
            try:
                _nocache(fd)
                os.pread(fd, 4096, 0)
            finally:
                os.close(fd)
            _pace(4096, t0, sim)
            lat.append(time.perf_counter() - t0)
        lat.sort()
        return DeviceProfile(
            str(mount_of(root)),
            read_bps,
            write_bps,
            lat[len(lat) // 2],
            time.time(),
            simulated=sim is not None,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


class ProfileStore:
    """Profiles cached in one JSON file, keyed by mount point."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return dict(json.loads(self.path.read_text()))
        except (OSError, ValueError):
            return {}

    def get(self, key: str, ttl_s: float = PROFILE_TTL_S) -> DeviceProfile | None:
        row = self._load().get(key)
        if not row:
            return None
        try:
            prof = DeviceProfile(**row)
        except TypeError:
            return None
        return prof if time.time() - prof.probed_at < ttl_s else None

    def put(self, key: str, prof: DeviceProfile) -> None:
        with self._lock:
            data = self._load()
            data[key] = asdict(prof)
            with contextlib.suppress(OSError):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=1))
                os.replace(tmp, self.path)


def profile_for(
    spec: TierSpec,
    store: ProfileStore | None,
    *,
    ttl_s: float = PROFILE_TTL_S,
    force: bool = False,
) -> DeviceProfile | None:
    """The cached profile of ``spec``'s volume, or a fresh probe (None when the probe fails)."""
    key = (
        str(mount_of(spec.path))
        + ("|sim" if spec.sim else "")
        + f"|{spec.path}" * bool(spec.sim)
    )
    if store is not None and not force:
        prof = store.get(key, ttl_s)
        if prof is not None:
            return prof
    try:
        prof = probe_device(spec.path, sim=spec.sim)
    except OSError as e:
        logger.warning("APC storage: cannot probe %s (%s); tier skipped", spec.path, e)
        return None
    if store is not None:
        store.put(key, prof)
    return prof


# ── encoded container ────────────────────────────────────────────────────
def _segments(path: Path) -> list[tuple[int, int, bool]]:
    """Cover a safetensors file with (offset, length, two_byte_plane) segments by tensor."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    start = 8 + n
    header.pop("__metadata__", None)
    spans = []
    for ent in header.values():
        if not isinstance(ent, dict) or "data_offsets" not in ent:
            continue
        a, b = ent["data_offsets"]
        spans.append(
            (start + int(a), start + int(b), str(ent.get("dtype")) in _TWO_BYTE)
        )
    spans.sort()
    out, pos = [], 0
    for a, b, plane in spans:
        if a > pos:
            out.append((pos, a - pos, False))
        if b > a:
            out.append((a, b - a, plane and (b - a) % 2 == 0))
        pos = max(pos, b)
    if pos < size:
        out.append((pos, size - pos, False))
    return out


def _zc(level: int):
    import zstandard

    return zstandard.ZstdCompressor(level=level, write_checksum=True)


def _comp(args) -> bytes:
    data, plane, level = args
    if plane:
        u = np.frombuffer(data, dtype=np.uint8).reshape(-1, 2)
        data = np.concatenate([u[:, 0], u[:, 1]]).tobytes()
    return bytes(_zc(level).compress(data))


def _decomp(args) -> bytes:
    import zstandard

    blob, plane, n = args
    raw = zstandard.ZstdDecompressor().decompress(blob, max_output_size=n)
    if plane:
        half = len(raw) // 2
        out = np.empty((half, 2), dtype=np.uint8)
        arr = np.frombuffer(raw, dtype=np.uint8)
        out[:, 0] = arr[:half]
        out[:, 1] = arr[half:]
        raw = out.tobytes()
    return bytes(raw)


def sample_ratio(path: Path, level: int = 1) -> tuple[float, float]:
    """(compression ratio, single-thread decode bytes/s) from a 4 MiB sample in the middle of the
    file's tensor data, shuffled the way the container would."""
    size = path.stat().st_size
    segs = [s for s in _segments(path) if s[1] >= CHUNK] or [(0, size, False)]
    off, ln, plane = max(segs, key=lambda s: s[1])
    take = min(CHUNK, ln)
    with open(path, "rb") as f:
        f.seek(off + (ln - take) // 2 // 2 * 2)
        data = f.read(take)
    blob = _comp((data, plane and len(data) % 2 == 0, level))
    t0 = time.perf_counter()
    _decomp((blob, plane and len(data) % 2 == 0, len(data)))
    dt = max(time.perf_counter() - t0, 1e-6)
    return len(data) / max(len(blob), 1), len(data) / dt


def encode_file(
    src: Path,
    dst: Path,
    meta: dict,
    *,
    level: int = 1,
    threads: int = 8,
    sim: tuple[float, float] | None = None,
) -> int:
    """Write ``src`` as a zstd container at ``dst`` (temp + atomic rename); returns bytes written."""
    segs = _segments(src)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    chunks: list[list[int]] = []  # [offset, original length, stored length, plane]
    t0 = time.perf_counter()
    with (
        open(src, "rb") as fin,
        open(tmp, "wb") as fout,
        ThreadPoolExecutor(threads) as ex,
    ):
        fout.write(_MAGIC)
        _nocache(fout.fileno())
        for off, ln, plane in segs:
            fin.seek(off)
            left = ln
            while left > 0:
                batch: list[Any] = []
                while left > 0 and len(batch) < threads * 2:
                    n = min(CHUNK, left)
                    batch.append((fin.read(n), plane, level))
                    left -= n
                for (raw, _, _), blob in zip(batch, ex.map(_comp, batch), strict=True):
                    fout.write(blob)
                    chunks.append([len(raw), len(blob), int(plane)])
        header = json.dumps(
            {
                "orig_size": src.stat().st_size,
                "chunks": chunks,
                "meta": meta,
                "codec": "zstd",
            },
            separators=(",", ":"),
        ).encode()
        fout.write(header)
        fout.write(_TRAILER.pack(len(header), zlib.crc32(header), _MAGIC))
        fout.flush()
        os.fsync(fout.fileno())
        written = fout.tell()
    _pace(written, t0, sim)
    os.replace(tmp, dst)
    return written


def read_container_header(path: Path) -> dict | None:
    """The header of an encoded file, or None when it is not a complete valid container."""
    try:
        size = path.stat().st_size
        if size < len(_MAGIC) + _TRAILER.size:
            return None
        with open(path, "rb") as f:
            if f.read(len(_MAGIC)) != _MAGIC:
                return None
            f.seek(size - _TRAILER.size)
            hlen, crc, magic = _TRAILER.unpack(f.read(_TRAILER.size))
            if magic != _MAGIC or hlen <= 0 or hlen > size:
                return None
            f.seek(size - _TRAILER.size - hlen)
            raw = f.read(hlen)
        if zlib.crc32(raw) != crc:
            return None
        head = json.loads(raw)
        stored = sum(c[1] for c in head["chunks"])
        if len(_MAGIC) + stored + hlen + _TRAILER.size != size:
            return None
        return dict(head)
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return None


def decode_file(src: Path, dst: Path, head: dict, *, threads: int = 8) -> None:
    """Rebuild the original file at ``dst``. Raises on any integrity failure."""
    with (
        open(src, "rb") as fin,
        open(dst, "wb") as fout,
        ThreadPoolExecutor(threads) as ex,
    ):
        fin.seek(len(_MAGIC))
        chunks = head["chunks"]
        i = 0
        while i < len(chunks):
            batch_meta = chunks[i : i + threads * 2]
            batch = [(fin.read(c[1]), bool(c[2]), c[0]) for c in batch_meta]
            for raw in ex.map(_decomp, batch):
                fout.write(raw)
            i += len(batch_meta)
    if dst.stat().st_size != int(head["orig_size"]):
        raise ValueError("decoded size mismatch")


# ── a lower tier ─────────────────────────────────────────────────────────
@dataclass
class _Entry:
    path: Path
    size: int  # stored bytes
    orig: int  # restored (raw) bytes
    tokens: tuple
    extra_hash: int
    encoded: bool
    trimmable: bool = False
    ratio: float = 1.0


def _hex_for(cache_hash: int) -> str:
    from mlx_vlm.apc import DiskBlockStore

    return str(DiskBlockStore._exact_id_for(cache_hash))


class FileTier:
    """One lower storage tier: a directory on its own volume holding this model's checkpoints."""

    def __init__(
        self,
        spec: TierSpec,
        namespace: str,
        profile: DeviceProfile | None,
        *,
        cap_bytes: int,
        budget: Any = None,
        name: str | None = None,
        encoding: str = "auto",
        is_last: bool = True,
    ):
        self.spec = spec
        self.root = spec.path
        self.dir = spec.path / namespace
        self.name = name or device_name(spec.path)
        self.fixed_name = name is not None
        self.profile = profile
        self.cap_bytes = int(cap_bytes)
        self.budget = budget
        self.encoding = encoding
        self.is_last = is_last
        self.index: dict[int, _Entry] = {}
        self._lock = threading.RLock()
        self._avail = (0.0, False)
        self._check: Any = None
        self._last_ok = 0.0
        self._stat_pool = ThreadPoolExecutor(
            1, thread_name_prefix=f"apc-stat-{self.name}"
        )
        self.hits = 0
        self.hit_bytes = 0
        self.skipped = 0  # lookups that found the tier unavailable
        self.demoted_in = 0
        self.invalidated = 0
        self.decode_bps = 0.0
        self._was_available = False
        if budget is not None:
            budget.register_owner(namespace, self._evict_path, self._busy_paths)
        self._moving: set[Path] = set()
        self.owner: Any = None  # the TieredDiskStore this tier belongs to
        self.reprofile: Any = (
            None  # callable() -> DeviceProfile | None, set by the engine
        )
        self._ns = namespace
        self.rescan()

    # availability: a hung NAS mount must not hang the engine, and a volume that is merely busy
    # (a multi-GB copy with fsync in flight makes even a stat slow) must not look unplugged
    BUSY_GRACE_S = 120.0
    CHECK_TIMEOUT_S = 2.0

    def available(self) -> bool:
        now = time.monotonic()
        t, ok = self._avail
        if now - t < 5.0:
            return ok
        gone = False
        pending = self._check
        if pending is None:
            pending = self._check = self._stat_pool.submit(
                lambda: ensure_root(self.root) and os.access(self.root, os.W_OK)
            )
        try:
            ok = bool(pending.result(timeout=self.CHECK_TIMEOUT_S))
            self._check = None
            self._last_ok = now if ok else self._last_ok
            gone = not ok
        except FutureTimeout:
            # still checking: keep the last known state while it was answering recently
            ok = self._avail[1] and now - self._last_ok < self.BUSY_GRACE_S
            gone = not ok
        except OSError:
            self._check = None
            ok, gone = False, True
        self._avail = (now, bool(ok))
        if gone:
            self._was_available = False
        elif ok and not self._was_available:
            self._was_available = True
            self.rescan()  # (re)mounted: validate what is there again
            stale = (
                self.profile is None
                or time.time() - self.profile.probed_at > PROFILE_TTL_S
            )
            if stale and callable(self.reprofile):
                try:
                    self.profile = self.reprofile() or self.profile
                except Exception:
                    logger.warning(
                        "APC storage %s: probe failed", self.name, exc_info=True
                    )
        return self._avail[1]

    def _busy_paths(self) -> set[Path]:
        return set(self._moving)

    def _evict_path(self, path: Path) -> bool:
        """The budget wants this file gone. A tier that has a lower tier to give it to returns
        False (the mover demotes it instead of the budget deleting it) unless the tier is far
        over its cap, i.e. the mover is stuck."""
        owner = self.owner
        with self._lock:
            h, e = next(
                ((k, v) for k, v in self.index.items() if v.path == path), (None, None)
            )
        if owner is not None and e is not None and not self.is_last:
            if self.used() < _delete_above(self.cap_bytes, e.size):
                pos = owner.lower.index(self) + 1
                if owner._target_for(pos, e.orig, len(e.tokens)) is not None:
                    owner._wake.set()
                    return False
        with self._lock:
            if h is not None:
                self.index.pop(h, None)
        self._unlink(path)
        return True

    def rescan(self) -> None:
        """Rebuild the index from the files, dropping everything that fails validation."""
        from mlx_vlm.apc import _read_safetensors_metadata

        with self._lock:
            self.index.clear()
        if not self.dir.is_dir():
            return
        for p in list(self.dir.iterdir()):
            try:
                if p.name.endswith(".tmp") or ".tmp." in p.name:
                    if time.time() - p.stat().st_mtime > 600:
                        p.unlink(missing_ok=True)
                    continue
                if p.suffix not in (
                    RAW_SUFFIX,
                    ENCODED_SUFFIX,
                ) or not p.stem.startswith("exact_"):
                    continue
                with open(
                    p, "rb"
                ) as probe:  # an I/O error is the volume's, not the file's
                    probe.read(1)
                if p.suffix == ENCODED_SUFFIX:
                    head = read_container_header(p)
                    meta = (head or {}).get("meta")
                    encoded, orig = True, int((head or {}).get("orig_size", 0))
                else:
                    meta = _read_safetensors_metadata(p)
                    encoded, orig = False, p.stat().st_size
                if not meta or meta.get("layout") != "exact_cache_v1":
                    self._invalid(p, "unreadable header")
                    continue
                if not encoded and not self._raw_complete(p):
                    self._invalid(p, "torn or truncated")
                    continue
                tokens = tuple(
                    int(x) for x in meta.get("token_ids", "").split(",") if x
                )
                self.index[int(meta["cache_hash"])] = _Entry(
                    p,
                    p.stat().st_size,
                    orig,
                    tokens,
                    int(meta.get("extra_hash", "0")),
                    encoded,
                    meta.get("prefix_trimmable") == "1",
                    orig / max(p.stat().st_size, 1),
                )
            except OSError as e:
                # the volume failed under us (a network glitch, an unplug): keep every file and
                # treat the tier as unavailable until the next check validates it again
                logger.warning(
                    "APC storage %s: scan interrupted (%s); tier unavailable",
                    self.name,
                    e,
                )
                self._avail = (time.monotonic(), False)
                self._was_available = False
                return
            except (ValueError, KeyError, TypeError):
                self._invalid(p, "unreadable")

    @staticmethod
    def _raw_complete(path: Path) -> bool:
        from mlx_vlm.apc import _read_safetensors_header

        parsed = _read_safetensors_header(path)
        if parsed is None:
            return False
        entries, _meta, start = parsed
        end = max(
            (
                int(e["data_offsets"][1])
                for e in entries.values()
                if "data_offsets" in e
            ),
            default=0,
        )
        return bool(path.stat().st_size >= start + end)

    def _invalid(self, path: Path, why: str) -> None:
        logger.warning("APC storage %s: %s %s, dropped", self.name, path.name, why)
        self.invalidated += 1
        with contextlib.suppress(OSError):
            path.unlink()

    def used(self) -> int:
        with self._lock:
            return sum(e.size for e in self.index.values())

    def find(self, tokens: tuple, extra_hash: int, max_len: int, min_len: int):
        """Longest stored prefix: (hash, length, entry) or None."""
        best = None
        with self._lock:
            items = list(self.index.items())
        for h, e in items:
            n = len(e.tokens)
            if e.extra_hash != extra_hash or not min_len < n <= max_len:
                continue
            if best is not None and n <= best[1]:
                continue
            if tokens[:n] == e.tokens:
                best = (h, n, e)
        return best

    # observed restore speed (bytes / s, EMA): reading a checkpoint back through the loader
    # (tensor by tensor, then copied into the cache) is slower than the sequential probe
    eff_bps = 0.0

    def note_restore(self, orig_bytes: int, seconds: float) -> None:
        if orig_bytes < (64 << 20) or seconds <= 0:
            return
        bps = orig_bytes / seconds
        self.eff_bps = bps if not self.eff_bps else 0.7 * self.eff_bps + 0.3 * bps

    def restore_s(self, e: _Entry) -> float | None:
        if self.profile is None:
            return None
        if self.eff_bps:
            floor = e.orig / self.eff_bps
            return max(
                floor,
                self.profile.restore_s(
                    e.orig, self.decode_bps, e.ratio if e.encoded else 1.0
                ),
            )
        return self.profile.restore_s(
            e.orig, self.decode_bps, e.ratio if e.encoded else 1.0
        )

    def drop(self, h: int) -> None:
        with self._lock:
            e = self.index.pop(h, None)
        if e is not None:
            self._unlink(e.path)

    def _unlink(self, path: Path) -> None:
        """Delete a file; a file somebody is reading right now goes when the reader is done."""
        owner = self.owner
        if owner is not None:
            owner._unlink_when_free(path)
            return
        with contextlib.suppress(OSError):
            path.unlink()

    def touch(self, h: int) -> None:
        e = self.index.get(h)
        if e is not None:
            with contextlib.suppress(OSError):
                os.utime(e.path)

    def put(self, src: Path, cache_hash: int, meta: dict, orig: int) -> bool:
        """Copy (or encode) ``src`` here. False when this tier cannot take it now."""
        if not self.available():
            return False
        encode = False
        ratio = 1.0
        if self.encoding != "raw" and self.profile is not None:
            try:
                ratio, dec = sample_ratio(src)
            except Exception:
                ratio, dec = 1.0, 0.0
            self.decode_bps = max(
                self.decode_bps, dec * 4
            )  # 4 chunk-parallel threads, conservative
            raw_t = self.profile.restore_s(orig)
            enc_t = self.profile.restore_s(orig, self.decode_bps, ratio)
            encode = self.encoding == "zstd" or (ratio >= 1.1 and enc_t < 0.9 * raw_t)
        if encode and importlib.util.find_spec("zstandard") is None:
            encode, ratio = False, 1.0  # the "compression" extra is not installed: raw
        size_est = int(orig / (ratio if encode else 1.0))
        if self.budget is not None and not self.budget.allow_write(size_est):
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        hexid = _hex_for(cache_hash)
        dst = self.dir / (hexid + (ENCODED_SUFFIX if encode else RAW_SUFFIX))
        self._moving.add(dst)
        sim = self.spec.sim
        try:
            if encode:
                written = encode_file(src, dst, meta, sim=sim)
            else:
                tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
                t0 = time.perf_counter()
                with open(src, "rb") as fi, open(tmp, "wb") as fo:
                    _nocache(fo.fileno())
                    while True:
                        b = fi.read(8 * MIB)
                        if not b:
                            break
                        fo.write(b)
                    fo.flush()
                    os.fsync(fo.fileno())
                written = tmp.stat().st_size
                _pace(written, t0, sim)
                os.replace(tmp, dst)
        except OSError as e:
            if self.budget is not None:
                self.budget.record_failure(e)
            with contextlib.suppress(OSError):
                dst.unlink()
            for t in self.dir.glob(dst.name + ".tmp.*"):
                with contextlib.suppress(OSError):
                    t.unlink()
            return False
        finally:
            self._moving.discard(dst)
        tokens = tuple(int(x) for x in meta.get("token_ids", "").split(",") if x)
        with self._lock:
            self.index[int(cache_hash)] = _Entry(
                dst,
                written,
                orig,
                tokens,
                int(meta.get("extra_hash", "0")),
                encode,
                meta.get("prefix_trimmable") == "1",
                orig / max(written, 1),
            )
        if self.budget is not None:
            self.budget.note_written(dst)
            try:
                self.budget.enforce(keep={self._ns})
            except Exception:
                logger.warning(
                    "APC storage %s: budget enforcement failed",
                    self.name,
                    exc_info=True,
                )
        self.demoted_in += 1
        return True

    def raw_path_for_load(self, h: int, scratch: Path) -> tuple[Path, Path | None]:
        """A path the upstream loader can read (a decoded temp file for an encoded entry) and the
        temp to remove afterwards. Raises on corruption."""
        e = self.index[h]
        if not e.encoded:
            return e.path, None
        head = read_container_header(e.path)
        if head is None:
            raise ValueError("container header invalid")
        tmp = scratch / f"{_hex_for(h)}.tmp.{os.getpid()}.{threading.get_ident()}"
        t0 = time.perf_counter()
        try:
            decode_file(e.path, tmp, head)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        _pace(e.size, t0, self.spec.sim)
        return tmp, tmp

    def snapshot(self) -> dict:
        p = self.profile
        return {
            "name": self.name,
            "path": str(self.root),
            "available": self._avail[1],
            "used_bytes": self.used(),
            "cap_bytes": self.cap_bytes,
            "entries": len(self.index),
            "read_bps": p.read_bps if p else None,
            "write_bps": p.write_bps if p else None,
            "latency_ms": p.latency_s * 1e3 if p else None,
            "hits": self.hits,
            "hit_bytes": self.hit_bytes,
            "skipped_unavailable": self.skipped,
            "effective_read_bps": self.eff_bps or None,
            "demoted_in": self.demoted_in,
            "invalidated": self.invalidated,
            "simulated": bool(self.spec.sim),
        }


# ── the tiered store ─────────────────────────────────────────────────────
class TieredDiskStore(SpillDiskStore):
    """The primary store plus ordered lower tiers; one ``disk`` for the APC manager."""

    prefill_tps: float = DEFAULT_PREFILL_TPS
    prefill_observed = False
    eff_bps = 0.0  # observed restore speed of the primary store (EMA, bytes / s)
    cost_rejected = (
        0  # candidates found but not worth restoring (slower than re-prefilling)
    )

    def _primary_restore_s(self, nbytes: int) -> float | None:
        if self.profile is None:
            return None
        t = self.profile.restore_s(nbytes)
        return max(t, nbytes / self.eff_bps) if self.eff_bps else t

    def __init__(self, *args: Any, name: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.name = name or device_name(self.dir.parent)
        self.fixed_name = name is not None
        self.profile: DeviceProfile | None = None
        self.sim: tuple[float, float] | None = None
        self.lower: list[FileTier] = []
        self._where: dict[int, FileTier] = {}
        self._lease_lock = threading.Lock()
        self._leases: Counter = Counter()
        self._deferred: set[Path] = set()
        self.last_device: str | None = None
        self.device_hits: Counter = Counter()
        self.soft_cap_bytes = 0
        self._mover: threading.Thread | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.moved = 0
        self.move_failures = 0
        self.dropped_not_worth = 0
        self.primary_hits = 0
        self.primary_hit_bytes = 0

    # ── configuration ──────────────────────────────────────────────────
    def add_lower(self, tier: FileTier) -> None:
        """Add a lower tier; the lower tiers stay ordered by measured read bandwidth. Tiers on
        the same volume (the primary store included) are told apart by their directory name."""
        tier.owner = self
        self.lower.append(tier)
        self.lower.sort(key=lambda t: -(t.profile.read_bps if t.profile else 0.0))
        for i, t in enumerate(self.lower):
            t.is_last = i == len(self.lower) - 1
        self._relabel()

    def _relabel(self) -> None:
        entries = [(self, self.dir.parent)] + [(t, t.root) for t in self.lower]
        base = {id(o): device_name(path) for o, path in entries}
        count = Counter(base.values())
        for o, path in entries:
            if not getattr(o, "fixed_name", False):  # an explicit name is kept
                o.name = base[id(o)] + (
                    f"/{path.name}" if count[base[id(o)]] > 1 else ""
                )

    def start_mover(self) -> None:
        if self._mover is None and self.lower:
            self._mover = threading.Thread(
                target=self._mover_loop, daemon=True, name="apc-tier-mover"
            )
            self._mover.start()

    def observe_prefill(self, tokens: int, seconds: float) -> None:
        """A real prefill: ``tokens`` computed in ``seconds`` (EMA; drives the cost model)."""
        if tokens < 256 or seconds <= 0:
            return
        tps = tokens / seconds
        if not self.prefill_observed:
            self.prefill_tps, self.prefill_observed = tps, True
        else:
            self.prefill_tps = 0.8 * self.prefill_tps + 0.2 * tps

    # ── lookup ─────────────────────────────────────────────────────────
    def _worth(self, restore_s: float | None, gained_tokens: int) -> bool:
        if restore_s is None:
            return True  # no profile for the primary store: the previous behaviour
        return restore_s < gained_tokens / max(self.prefill_tps, 1.0)

    def find_exact_prefix(
        self,
        token_ids,
        *,
        extra_hash=0,
        max_prefix_tokens=None,
        min_prefix_tokens=0,
        block_size=16,
    ):
        tokens = tuple(int(t) for t in token_ids)
        max_len = len(tokens) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        cands = []  # (cost_s, hash, length, tier)
        r0 = super().find_exact_prefix(
            token_ids,
            extra_hash=extra_hash,
            max_prefix_tokens=max_prefix_tokens,
            min_prefix_tokens=min_prefix_tokens,
            block_size=block_size,
        )
        if r0 is not None:
            h, n = r0
            rs = self._primary_restore_s(self.exact_cache_bytes(h))
            if self._worth(rs, n - min_prefix_tokens):
                cands.append(
                    ((rs or 0.0) + (len(tokens) - n) / self.prefill_tps, h, n, None)
                )
            else:
                self._rejected(self.name, n, rs)
        for tier in self.lower:
            if not tier.available():
                tier.skipped += 1
                continue
            hit = tier.find(tokens, extra_hash, max_len, min_prefix_tokens)
            if hit is None:
                continue
            h, n, e = hit
            rs = tier.restore_s(e)
            if rs is None or not self._worth(rs, n - min_prefix_tokens):
                self._rejected(tier.name, n, rs)
                continue
            cands.append((rs + (len(tokens) - n) / self.prefill_tps, h, n, tier))
        if not cands:
            return None
        _cost, h, n, tier = min(cands, key=lambda c: (c[0], -c[2]))
        if tier is not None:
            self._where[h] = tier
        else:
            self._where.pop(h, None)
        return h, n

    # ── leases: a file that is being read is not deleted under the reader ─────────────
    def _lease(self, path: Path) -> None:
        with self._lease_lock:
            self._leases[path] += 1

    def _release(self, path: Path) -> None:
        with self._lease_lock:
            self._leases[path] -= 1
            if self._leases[path] > 0:
                return
            del self._leases[path]
            late = path in self._deferred
            self._deferred.discard(path)
        if late:
            with contextlib.suppress(OSError):
                path.unlink()

    def _unlink_when_free(self, path: Path) -> None:
        with self._lease_lock:
            if self._leases.get(path, 0) > 0:
                self._deferred.add(path)
                return
        with contextlib.suppress(OSError):
            path.unlink()

    @contextlib.contextmanager
    def _leased(self, path: Path):
        self._lease(path)
        try:
            yield
        finally:
            self._release(path)

    def _rejected(self, name: str, n: int, restore_s: float | None) -> None:
        self.cost_rejected += 1
        now = time.monotonic()
        if now - self._last_reject_log > 1.0:
            self._last_reject_log = now
            logger.info(
                "APC storage: %d-token checkpoint on %s not restored (restore %s s, prefill %.0f tok/s)",
                n,
                name,
                "?" if restore_s is None else f"{restore_s:.1f}",
                self.prefill_tps,
            )

    _last_reject_log = 0.0

    def exact_cache_bytes(self, cache_hash: int) -> int:
        tier = self._where.get(cache_hash)
        if tier is not None:
            e = tier.index.get(cache_hash)
            return e.orig if e is not None else 0
        return int(super().exact_cache_bytes(cache_hash))

    def load_exact_cache(self, cache_hash, **kwargs):
        tier = self._where.get(cache_hash)
        t0 = time.perf_counter()
        if tier is None:
            with self._index_lock:
                src = self._exact_index.get(cache_hash)
            if src is not None:
                with self._leased(src):
                    got = super().load_exact_cache(cache_hash, **kwargs)
            else:
                got = super().load_exact_cache(cache_hash, **kwargs)
            if got is None:  # moved down between the lookup and the read
                for t in self.lower:
                    if cache_hash in t.index and t.available():
                        tier = t
                        break
            if tier is None:
                if got is not None:
                    nbytes = self.exact_cache_bytes(cache_hash)
                    if nbytes >= (64 << 20):
                        bps = nbytes / max(time.perf_counter() - t0, 1e-6)
                        self.eff_bps = (
                            bps if not self.eff_bps else 0.7 * self.eff_bps + 0.3 * bps
                        )
                    self.last_device = self.name
                    self.device_hits[self.name] += 1
                    self.primary_hits += 1
                    self.primary_hit_bytes += self.exact_cache_bytes(cache_hash)
                return got
        got = self._load_lower(tier, cache_hash, **kwargs)
        if got is not None:
            self.last_device = tier.name
            self.device_hits[tier.name] += 1
            tier.hits += 1
            e = tier.index.get(cache_hash)
            tier.hit_bytes += e.orig if e else 0
            tier.touch(cache_hash)
        sim = tier.spec.sim
        if sim is not None and got is not None:  # the file is really on a fast disk
            e = tier.index.get(cache_hash)
            _pace(e.size if e else 0, t0, sim)
        if got is not None:
            e = tier.index.get(cache_hash)
            if e is not None:
                tier.note_restore(e.orig, time.perf_counter() - t0)
        return got

    def _load_lower(
        self,
        tier: FileTier,
        h: int,
        *,
        prefix_len=None,
        min_capacity_tokens=None,
        **_kw,
    ):
        e = tier.index.get(h)
        if e is None:
            return None
        with self._leased(e.path):
            return self._load_lower_leased(tier, h, e, prefix_len, min_capacity_tokens)

    def _load_lower_leased(self, tier, h, e, prefix_len, min_capacity_tokens):
        """Read one checkpoint back from a lower tier. Fail closed, but delete only what is
        provably bad: a file that vanished (superseded, moved) is a miss, an I/O error is a miss
        that keeps the file, a file that is torn or fails validation is removed."""
        tmp = None
        try:
            try:
                path, tmp = tier.raw_path_for_load(h, self.dir)
            except OSError:
                return None
            except Exception as exc:
                tier._invalid(e.path, f"decode failed: {type(exc).__name__}")
                tier.drop(h)
                return None
            loaded = self._load_exact_cache_file(
                path, min_capacity_tokens=min_capacity_tokens, prefix_len=prefix_len
            )
        except OSError:
            return None
        except Exception as exc:
            tier._invalid(e.path, f"load raised {type(exc).__name__}")
            tier.drop(h)
            return None
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                with self._header_cache_lock:
                    self._header_cache.pop(tmp, None)
        if loaded is None:
            if not e.path.exists():
                with tier._lock:
                    tier.index.pop(h, None)  # gone under us: a miss
            elif not e.encoded and not tier._raw_complete(e.path):
                tier._invalid(e.path, "torn or truncated")
                tier.drop(h)
            else:
                logger.warning(
                    "APC storage %s: %s could not be read back; kept, lookup recomputes",
                    tier.name,
                    e.path.name,
                )
            return None
        if self.validator is not None:
            try:
                ok = bool(self.validator(loaded[0], loaded[2]))
            except Exception:
                ok = False
            if not ok:
                tier._invalid(e.path, "structure does not match the model")
                tier.drop(h)
                return None
        return loaded

    # ── supersede across tiers ─────────────────────────────────────────
    def exact_prefixes_of(self, tokens, extra_hash, exclude=None) -> list:
        out = super().exact_prefixes_of(tokens, extra_hash, exclude)
        for tier in self.lower:
            with tier._lock:
                items = list(tier.index.items())
            for h, e in items:
                if (
                    h != exclude
                    and e.extra_hash == extra_hash
                    and 0 < len(e.tokens) < len(tokens)
                ):
                    if tokens[: len(e.tokens)] == e.tokens:
                        out.append(h)
        return out

    def drop_exact(self, cache_hash) -> bool:
        for tier in self.lower:
            if cache_hash in tier.index:
                tier.drop(cache_hash)
                return True
        with self._index_lock:
            path = self._exact_index.get(cache_hash)
        if path is None:
            return False
        ok = self._remove_primary(path)
        if ok:
            self._tok_cache().pop(cache_hash, None)
        return ok

    # ── placement ──────────────────────────────────────────────────────
    def _evict_path(self, path) -> bool:
        """The root's budget wants this file gone: with a lower tier that can take it, the mover
        demotes it instead (return False = busy); otherwise (or when the mover is stuck, the
        store far over its cap) it is deleted, i.e. recomputed later."""
        if self.lower and self.soft_cap_bytes and path not in self._in_flight_paths():
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            if self._root_used() < _delete_above(self.soft_cap_bytes, size):
                meta = self._meta_of(path)
                if meta is not None:
                    ntok = meta.get("token_ids", "").count(",") + 1
                    if self._target_for(0, size, ntok) is not None:
                        self._wake.set()
                        return False
        return self._remove_primary(path)

    def _remove_primary(self, path) -> bool:
        """Delete one primary-store file (index and all); a file being read goes when the reader
        is done. False when it is still being written."""
        if path in self._in_flight_paths():
            return False
        if self._leases.get(path, 0) > 0:
            with self._lease_lock:
                self._deferred.add(path)
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self._drop_index_for_path(path)
            self._disk_bytes = max(0, self._disk_bytes - size)
            self.evictions += 1
            return True
        return bool(super()._evict_path(path))

    def _write_payload(self, shard_id, block_hashes, payload) -> bool:
        ok = super()._write_payload(shard_id, block_hashes, payload)
        if ok:
            h = getattr(payload, "cache_hash", None)
            if h is not None:
                for t in (
                    self.lower
                ):  # the fresh copy is the fastest one: drop slower duplicates
                    if h in t.index:
                        t.drop(h)
            self._wake.set()
        return ok

    def _own_files(self):
        """(mtime, size, hash, path) of this model's primary-store files, oldest first."""
        out = []
        with self._index_lock:
            items = list(self._exact_index.items())
        busy = self._in_flight_paths()
        for h, p in items:
            if p in busy:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((st.st_mtime, st.st_size, h, p))
        out.sort()
        return out

    def _meta_of(self, path: Path) -> dict | None:
        from mlx_vlm.apc import _read_safetensors_metadata

        m = _read_safetensors_metadata(path)
        return m if m and m.get("layout") == "exact_cache_v1" else None

    def _target_for(self, start: int, orig: int, tokens: int):
        """First lower tier after position ``start`` that can hold and usefully serve the entry."""
        for t in self.lower[start:]:
            if not t.available() or t.profile is None:
                continue
            if t.profile.restore_s(orig) >= tokens / max(self.prefill_tps, 1.0):
                continue  # even raw this tier would be slower than re-prefilling
            return t
        return None

    def rebalance(self) -> int:
        """One mover pass: demote LRU checkpoints out of tiers over their soft cap. Returns moves."""
        moved = 0
        if not self.lower:
            return 0
        # primary store -> first lower tier
        used = self._root_used()
        if self.soft_cap_bytes and used > self.soft_cap_bytes:
            for _mt, size, h, path in self._own_files():
                if used <= self.soft_cap_bytes or self._stop.is_set():
                    break
                meta = self._meta_of(path)
                if meta is None:
                    continue
                ntok = meta.get("token_ids", "").count(",") + 1
                tier = self._target_for(0, size, ntok)
                if tier is None:
                    self.dropped_not_worth += 1
                    continue
                with self._leased(path):
                    moved_ok = tier.put(path, h, meta, size)
                if moved_ok:
                    self._evict_primary(h, path)
                    used -= size
                    moved += 1
                    self.moved += 1
                else:
                    self.move_failures += 1
        # lower tiers cascade downwards
        for i, tier in enumerate(self.lower[:-1]):
            while tier.used() > tier.cap_bytes and not self._stop.is_set():
                with tier._lock:
                    victims = sorted(
                        tier.index.items(), key=lambda kv: self._mtime(kv[1].path)
                    )
                if not victims:
                    break
                progressed = False
                for h, e in victims:
                    if tier.used() <= tier.cap_bytes:
                        break
                    nxt = self._target_for(i + 1, e.orig, len(e.tokens))
                    if nxt is None:
                        break
                    meta = self._meta_for_entry(e)
                    if meta is None:
                        tier.drop(h)
                        continue
                    tmp_raw, cleanup = self._raw_copy(tier, h)
                    try:
                        if tmp_raw is None:
                            ok = False
                        else:
                            with self._leased(e.path):
                                ok = nxt.put(tmp_raw, h, meta, e.orig)
                    finally:
                        if cleanup is not None:
                            with contextlib.suppress(OSError):
                                cleanup.unlink()
                    if ok:
                        tier.drop(h)
                        moved += 1
                        self.moved += 1
                        progressed = True
                    else:
                        self.move_failures += 1
                if not progressed:
                    break
        return moved

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    def _meta_for_entry(self, e: _Entry) -> dict | None:
        if e.encoded:
            head = read_container_header(e.path)
            return (head or {}).get("meta")
        return self._meta_of(e.path)

    def _raw_copy(self, tier: FileTier, h: int):
        try:
            return tier.raw_path_for_load(h, self.dir)
        except Exception:
            tier.drop(h)
            return None, None

    def _root_used(self) -> int:
        b = self.budget
        if b is not None:
            with contextlib.suppress(Exception):
                return int(b.used())
        return int(self._disk_bytes)

    def _evict_primary(self, h: int, path: Path) -> None:
        if path in self._in_flight_paths():
            return
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        self._drop_index_for_path(path)
        path.unlink(missing_ok=True)
        self._disk_bytes = max(0, self._disk_bytes - size)

    def _mover_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=10.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.rebalance()
            except Exception:
                logger.warning("APC storage: mover pass failed", exc_info=True)

    def settle(self, timeout: float = 30.0) -> None:
        """Run mover passes inline until nothing moves (tests, shutdown)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.flush()
            if not self.rebalance():
                return

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._mover is not None:
            self._mover.join(timeout=30)
        for t in self.lower:
            t._stat_pool.shutdown(wait=False)
        super().close()

    def snapshot(self) -> list[dict]:
        primary = {
            "name": self.name,
            "path": str(self.dir.parent),
            "available": True,
            "used_bytes": self._root_used(),
            "cap_bytes": self.soft_cap_bytes or self.max_bytes or 0,
            "entries": len(self._exact_index),
            "read_bps": self.profile.read_bps if self.profile else None,
            "write_bps": self.profile.write_bps if self.profile else None,
            "latency_ms": self.profile.latency_s * 1e3 if self.profile else None,
            "hits": self.primary_hits,
            "hit_bytes": self.primary_hit_bytes,
            "effective_read_bps": self.eff_bps or None,
            "prefill_tps": round(self.prefill_tps, 1),
            "cost_rejected": self.cost_rejected,
            "simulated": bool(self.sim),
        }
        return [primary, *(t.snapshot() for t in self.lower)]


__all__ = [
    "DeviceProfile",
    "FileTier",
    "ProfileStore",
    "TierSpec",
    "TieredDiskStore",
    "decode_file",
    "device_name",
    "encode_file",
    "parse_tiers",
    "probe_device",
    "profile_for",
]
