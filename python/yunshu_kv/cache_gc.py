"""Offline integrity check and garbage collection for the KV / APC SSD caches.

Both SSD tiers (the APC tier ``<dir>/<namespace>/{shard,exact}_<hex>.safetensors`` and the
text engine's ``<dir>/<bucket>/<hex>.safetensors``) already drop what they cannot read when
a server starts. What they do not do: notice a file cut short after a crash (the header
parses, the payload is missing), delete entries written by an older cache format, or collect
temp files from writes that never finished, so those pile up outside the size cap. This
module finds them, and enforces the cap on what remains.

Only files it can positively identify as cache files are ever removed; any other file in the
directory is left alone. :func:`scan` is a dry run unless ``apply=True``.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

SUFFIX = ".safetensors"
# Matches yunshu_engine.ssd_kv_cache._READABLE_VERSIONS (the text tier's format marker).
READABLE_TEXT_VERSIONS = frozenset({"1"})
MAX_HEADER = 100 * 1024 * 1024
# A temp file younger than this may belong to a write in flight.
TMP_GRACE_S = 600.0

_APC_NAME = re.compile(r"^(shard|exact)_[0-9a-f]{32}$")
_HEX = re.compile(r"^[0-9a-f]+$")


@dataclass
class Finding:
    path: str
    reason: str  # tmp-orphan | truncated | corrupt | old-format | over-cap
    bytes: int


@dataclass
class Report:
    root: str
    files: int = 0  # cache files seen (valid or not)
    bytes: int = 0  # bytes of valid cache files before the cap
    unrecognized: int = 0  # *.safetensors this module does not own: never touched
    findings: list[Finding] = field(default_factory=list)
    applied: bool = False
    freed: int = 0
    kept_bytes: int = 0
    cap_bytes: int | None = None

    def by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.reason] = out.get(f.reason, 0) + 1
        return out


def check_file(path: Path) -> tuple[str | None, dict]:
    """``(problem, metadata)`` for one safetensors file; problem is None when it is intact.

    Intact means: a header that parses, and a file length equal to header + the end of the
    last tensor (so a file cut short, or with trailing garbage, is reported).
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return "truncated", {}
            (hlen,) = struct.unpack("<Q", raw)
            if hlen <= 0 or hlen > MAX_HEADER:
                return "corrupt", {}
            if 8 + hlen > size:
                return "truncated", {}
            header = json.loads(f.read(hlen))
    except (OSError, ValueError, struct.error):
        return "corrupt", {}
    if not isinstance(header, dict):
        return "corrupt", {}
    meta = header.get("__metadata__") or {}
    end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            end = max(end, int(info["data_offsets"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return "corrupt", {}
    expected = 8 + hlen + end
    if size < expected:
        return "truncated", meta
    if size > expected:
        return "corrupt", meta
    return None, meta


def _is_tmp(name: str) -> bool:
    return name.endswith(".tmp") or ".tmp." in name


def scan(
    root: Path,
    *,
    max_bytes: int | None = None,
    apply: bool = False,
    tmp_grace_s: float = TMP_GRACE_S,
    now: float | None = None,
) -> Report:
    """Check every cache file under ``root``; with ``apply`` remove the bad ones and trim to the cap."""
    now = time.time() if now is None else now
    rep = Report(root=str(root), applied=apply, cap_bytes=max_bytes)
    if not root.is_dir():
        return rep
    valid: list[tuple[float, int, Path]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if _is_tmp(p.name):
            if now - st.st_mtime >= tmp_grace_s:
                rep.findings.append(Finding(str(p), "tmp-orphan", st.st_size))
            continue
        if p.suffix != SUFFIX:
            continue
        apc = bool(_APC_NAME.match(p.stem))
        text = len(p.parent.name) == 1 and bool(_HEX.match(p.stem)) and not apc
        if not (apc or text):
            rep.unrecognized += 1
            continue
        rep.files += 1
        problem, meta = check_file(p)
        if problem is None and text:
            version = meta.get("yunshu_cache_version", "unknown")
            if version not in READABLE_TEXT_VERSIONS:
                problem = "old-format"
        if problem is not None:
            rep.findings.append(Finding(str(p), problem, st.st_size))
            continue
        rep.bytes += st.st_size
        valid.append((st.st_mtime, st.st_size, p))
    kept = rep.bytes
    if max_bytes is not None and max_bytes > 0 and kept > max_bytes:
        for _mtime, size, p in sorted(valid):  # oldest first
            if kept <= max_bytes:
                break
            rep.findings.append(Finding(str(p), "over-cap", size))
            kept -= size
    rep.kept_bytes = kept
    if apply:
        for f in rep.findings:
            try:
                os.unlink(f.path)
                rep.freed += f.bytes
            except OSError:
                pass
    return rep
