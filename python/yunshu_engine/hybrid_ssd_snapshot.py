"""Whole-cache boundary-snapshot persistence for HYBRID models.

Standard models spill 64-token KV blocks to SSD (ssd_kv_cache.py) and reassemble
them on reuse. HYBRID models (Qwen3.5 etc.) can't: their linear-attention layers
are backed by ArraysCache — a fixed-size recurrent state that isn't decomposable
into per-token blocks. So for hybrid we persist the ENTIRE cache snapshot at a
block boundary (all layers) as one safetensors file keyed by the prefix-token
hash, and reuse it whole with trim=0 (never slicing the recurrent state). We key
the snapshot by prefix hash so it plugs into the prefix cache's get_no_trim() path.

Layer serialization:
  - KVCache layer        -> tensors l{i}_k, l{i}_v ; meta l{i}=kv, l{i}o=offset
  - ArraysCache layer    -> tensors l{i}_a{j} for each non-None state array ;
                            meta l{i}=arr, l{i}n=count, l{i}m=non-None-index-csv
On load the same object types are rebuilt (KVCache keys/values/offset, ArraysCache
.cache list), giving a cache identical to the in-RAM boundary snapshot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import struct
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class HybridSnapshotStore:
    """Disk store of full hybrid cache snapshots, keyed by prefix-token hash."""

    def __init__(self, cache_dir: str):
        self._dir = Path(os.path.expanduser(cache_dir)) / "hybrid_snapshots"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # In-RAM index: key(hex) -> token_count, so get_no_trim can find the
        # longest stored boundary without re-hashing/stat-ing every candidate.
        self._index: dict[str, int] = {}
        self._scan_existing()

    def _scan_existing(self) -> None:
        for sub in self._dir.iterdir() if self._dir.exists() else []:
            if not sub.is_dir():
                continue
            for f in sub.glob("*.safetensors"):
                if ".tmp." in f.name:
                    # leftover atomic-write temp from a crash mid-save — clean it up,
                    # don't index it (its stem isn't a real key).
                    with contextlib.suppress(OSError):
                        f.unlink()
                    continue
                self._index[f.stem] = 0  # token_count filled lazily on load

    def _path(self, key_hex: str) -> Path:
        sub = self._dir / key_hex[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key_hex}.safetensors"

    def has(self, key: bytes) -> bool:
        return key.hex() in self._index

    def _read_header_tok(self, key_hex: str) -> int:
        """Read ONLY the token-count from a snapshot's safetensors header (cheap —
        a few hundred bytes, no tensor load). _scan_existing indexes
        disk-discovered snapshots with token_count=0 (lazy), so after a process
        restart the real count is unknown until a full load. candidate_token_counts
        needs the real count to probe the EXACT stored boundary, so resolve it from
        the header here. Returns 0 on any failure."""
        try:
            path = self._path(key_hex)
            with open(path, "rb") as fh:
                _n = struct.unpack("<Q", fh.read(8))[0]
                if _n <= 0 or _n > (1 << 24):  # sane header bound
                    return 0
                hdr = json.loads(fh.read(_n).decode("utf-8"))
            return int(hdr.get("__metadata__", {}).get("tok", "0"))
        except Exception:
            return 0

    def candidate_token_counts(self) -> list[int]:
        """Distinct stored snapshot token-counts, DESCENDING (longest first), so the
        prefix-cache probe can try the EXACT stored boundaries instead of guessing
        64-aligned ones (the snapshots are keyed by the hash of the FULL unaligned
        prompt, so an aligned probe only matched when len % block == 0). Resolves
        any lazily-0 disk-discovered counts via the cheap header read above."""
        with self._lock:
            items = list(self._index.items())
        counts: set[int] = set()
        for key_hex, tok in items:
            if tok <= 0:
                tok = self._read_header_tok(key_hex)
                if tok > 0:
                    with self._lock:
                        # only fill if still present + still unknown (avoid clobbering
                        # a real count a concurrent load() just wrote)
                        if self._index.get(key_hex, -1) == 0:
                            self._index[key_hex] = tok
            if tok > 0:
                counts.add(tok)
        return sorted(counts, reverse=True)

    def save(self, key: bytes, cache_list: list, token_count: int) -> None:
        """Serialize a whole boundary snapshot (all layers) to one file."""
        tensors: dict[str, mx.array] = {}
        meta: dict[str, str] = {
            "n": str(len(cache_list)),
            "tok": str(token_count),
            "q": "i8",
        }

        def _put(name: str, arr) -> None:
            """int8-quantize one tensor (per-tensor symmetric scale) into the
            output, recording its scale + original dtype in metadata. int8 ≈ 4x
            smaller on disk than bf16/fp32, matching the standard per-block SSD
            path; lossy-by-design (SSD tier accepts it)."""
            a = mx.array(arr)
            orig_dtype = str(a.dtype).split(".")[-1]
            af = a.astype(mx.float32)
            amax = mx.max(mx.abs(af))
            mx.eval(amax)
            s = float(amax) / 127.0
            if not (s > 0):
                s = 1.0
            q = mx.clip(mx.round(af / s), -127, 127).astype(mx.int8)
            tensors[name] = q
            meta[f"{name}s"] = repr(s)
            meta[f"{name}d"] = orig_dtype

        try:
            for i, c in enumerate(cache_list):
                k = getattr(c, "keys", None)
                v = getattr(c, "values", None)
                if k is not None and v is not None and not isinstance(k, (tuple, list)):
                    _put(f"l{i}_k", k)
                    _put(f"l{i}_v", v)
                    meta[f"l{i}"] = "kv"
                    meta[f"l{i}o"] = str(int(getattr(c, "offset", k.shape[-2])))
                else:
                    # ArraysCache (recurrent state) — serialize its .state/.cache list.
                    state = getattr(c, "state", None)
                    if state is None:
                        state = getattr(c, "cache", None)
                    state = list(state) if state is not None else []
                    nonnull = []
                    for j, arr in enumerate(state):
                        if arr is not None and hasattr(arr, "shape"):
                            _put(f"l{i}_a{j}", arr)
                            nonnull.append(j)
                    meta[f"l{i}"] = "arr"
                    meta[f"l{i}n"] = str(len(state))
                    meta[f"l{i}m"] = ",".join(str(x) for x in nonnull)
            if not tensors:
                return
            path = self._path(key.hex())
            # atomic write — save to a temp file then os.replace, so a
            # crash or concurrent load() never sees a torn/truncated .safetensors (which
            # would then fail validation forever → permanent miss + disk leak, since the
            # stale index entry is only pruned when the file is ABSENT, not corrupt).
            # The temp name MUST end in `.safetensors`.
            # mx.save_safetensors APPENDS `.safetensors` when the name lacks it, so
            # `{path}.tmp.{pid}` was written as `{path}.tmp.{pid}.safetensors` and the
            # subsequent os.replace({path}.tmp.{pid}, …) raised FileNotFoundError —
            # swallowed by the except → EVERY hybrid snapshot save silently failed
            # (the whole-snapshot SSD tier was dead: has()→False →
            # every restore fell back to a full prefill). Keep the extension.
            tmp_path = f"{path}.{os.getpid()}.tmp.safetensors"
            mx.save_safetensors(tmp_path, tensors, meta)
            os.replace(tmp_path, str(path))
            with self._lock:
                self._index[key.hex()] = token_count
        except Exception:
            logger.debug("hybrid snapshot save failed", exc_info=True)

    def load(self, key: bytes) -> tuple[list | None, int]:
        """Rebuild the cache_list from disk. Returns (cache_list, token_count)."""
        key_hex = key.hex()
        if key_hex not in self._index:
            return None, 0
        path = self._path(key_hex)
        if not path.exists():
            with self._lock:
                self._index.pop(key_hex, None)
            return None, 0
        try:
            from mlx_lm.models.cache import ArraysCache, KVCache

            _DT = {
                "bfloat16": mx.bfloat16,
                "float16": mx.float16,
                "float32": mx.float32,
            }
            arrays, meta = mx.load(str(path), return_metadata=True)
            quant = meta.get("q") == "i8"

            def _get(name):
                a = arrays[name]
                if quant:
                    s = float(meta.get(f"{name}s", "1.0"))
                    dt = _DT.get(meta.get(f"{name}d", "bfloat16"), mx.bfloat16)
                    return (a.astype(mx.float32) * s).astype(dt)
                return a

            n = int(meta.get("n", "0"))
            token_count = int(meta.get("tok", "0"))
            out: list[Any] = []
            for i in range(n):
                kind = meta.get(f"l{i}", "kv")
                if kind == "kv":
                    c = KVCache()
                    c.keys = _get(f"l{i}_k")
                    c.values = _get(f"l{i}_v")
                    c.offset = int(meta.get(f"l{i}o", c.keys.shape[-2]))
                    out.append(c)
                else:
                    size = int(meta.get(f"l{i}n", "0"))
                    nonnull = [
                        int(x) for x in meta.get(f"l{i}m", "").split(",") if x != ""
                    ]
                    state: list[Any] = [None] * size
                    for j in nonnull:
                        state[j] = _get(f"l{i}_a{j}")
                    ac = ArraysCache(size)
                    ac.cache = state
                    out.append(ac)
            with self._lock:
                self._index[key_hex] = token_count
            return out, token_count
        except Exception:
            logger.debug("hybrid snapshot load failed", exc_info=True)
            return None, 0
