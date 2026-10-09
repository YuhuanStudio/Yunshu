"""Persistent metrics history and request log (SQLite, WAL), recorded from startup.

The console's charts and request list must not depend on the console having been open. The
sampler in ``history.py`` reads cheap in-process counters once a second and hands each row to
this store; a background thread writes them in batches (default every 10 s), so nothing here runs
on the generation thread and a sample costs one list append.

Resolutions (multi-resolution, each table keeps its own window):

====== ========== ==========================
table  resolution retention
====== ========== ==========================
m1     1 s        1 hour
m10    10 s       24 hours
m60    1 min      ``retention_days`` (30)
====== ========== ==========================

``m10`` and ``m60`` are rolled up from the finer table with SQL ``AVG`` (``MAX`` for peaks), for
the buckets a flush touched, so a restart never loses or doubles a partial bucket. A gap in time
(the process was down) stays a gap: rows exist only where a sample was taken, and
:meth:`HistoryStore.read` reports the gaps instead of interpolating.

The request log keeps metadata only, built from a fixed whitelist: id, model, route, status,
timestamps, token counts, cached tokens, TTFT, decode speed, finish reason, the API key's *name*.
Prompts, outputs, headers and client identity cannot reach it: unknown keys are dropped.

The file is bounded: retention by age, and a byte cap that trims the oldest rows when exceeded.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TIERS = (("m1", 1, 3600), ("m10", 10, 86400), ("m60", 60, None))
TIER_RES = {name: res for name, res, _ in TIERS}
MAX_FIELDS = frozenset({"peak_gb"})  # rolled up with MAX, every other field with AVG

REQUEST_COLUMNS = (
    "request_id",
    "t",
    "t_start",
    "model",
    "path",
    "stream",
    "status",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "prefill_tps",
    "decode_tps",
    "ttft_ms",
    "queue_wait_ms",
    "key_name",
    "cancelled",
)
_TEXT = {"request_id", "model", "path", "finish_reason", "key_name"}
_INT = {
    "stream",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cancelled",
}
_LABEL = re.compile(r"^[A-Za-z0-9._:/+\- ]{1,96}$")


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def request_row(entry: dict) -> dict[str, Any] | None:
    """The whitelisted, typed row for one finished request; None without a usable id and time."""
    rid = entry.get("request_id")
    t = _num(entry.get("t"))
    if not isinstance(rid, str) or not _LABEL.fullmatch(rid) or t is None:
        return None
    out: dict[str, Any] = {}
    for col in REQUEST_COLUMNS:
        v = entry.get(col)
        if col == "key_name":
            # a name the user typed for the key: any printable text, short, never a secret field
            out[col] = v[:64] if isinstance(v, str) and v.isprintable() else None
        elif col in _TEXT:
            out[col] = v if isinstance(v, str) and _LABEL.fullmatch(v) else None
        elif col in _INT:
            n = _num(v) if not isinstance(v, bool) else float(v)
            out[col] = None if n is None else int(n)
        else:
            out[col] = _num(v)
    out["request_id"], out["t"] = rid, t
    return out


class HistoryStore:
    def __init__(
        self,
        path: Path | str,
        fields: tuple[str, ...],
        *,
        retention_days: float = 30.0,
        max_bytes: int = 64 * 1024 * 1024,
        flush_s: float = 10.0,
        clock=time.time,
    ) -> None:
        self.path = Path(path)
        self.fields = tuple(fields)
        self.retention_days = float(retention_days)
        self.max_bytes = int(max_bytes)
        self.flush_s = float(flush_s)
        self._clock = clock
        self._lock = (
            threading.RLock()
        )  # one connection; the writer thread and readers share it
        self._pending: list[tuple[float, dict]] = []
        self._pending_requests: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0
        self.last_error: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5)
        self._init()

    # ── schema ──────────────────────────────────────────────────────────
    def _init(self) -> None:
        with self._lock:
            db = self._db
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            cols = ", ".join(f'"{f}" REAL' for f in self.fields)
            for name, _res, _keep in TIERS:
                db.execute(
                    f"CREATE TABLE IF NOT EXISTS {name} (t INTEGER PRIMARY KEY, {cols})"
                )
                existing = {r[1] for r in db.execute(f"PRAGMA table_info({name})")}
                for f in self.fields:  # a newer build may have added fields
                    if f not in existing:
                        db.execute(f'ALTER TABLE {name} ADD COLUMN "{f}" REAL')
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests ("
                "request_id TEXT NOT NULL, t REAL NOT NULL, t_start REAL, model TEXT, path TEXT,"
                " stream INTEGER, status INTEGER, finish_reason TEXT, prompt_tokens INTEGER,"
                " completion_tokens INTEGER, cached_tokens INTEGER, prefill_tps REAL,"
                " decode_tps REAL, ttft_ms REAL, queue_wait_ms REAL, key_name TEXT,"
                " cancelled INTEGER, PRIMARY KEY (t, request_id))"
            )
            db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
                (str(SCHEMA_VERSION),),
            )
            db.commit()

    # ── intake (cheap: called from the sampler / request completion) ─────
    def add_sample(self, t: float, row: dict[str, float | None]) -> None:
        with self._lock:
            self._pending.append((t, row))

    def add_request(self, entry: dict) -> None:
        row = request_row(entry)
        if row is not None:
            with self._lock:
                self._pending_requests.append(row)

    # ── writing ─────────────────────────────────────────────────────────
    def flush(self) -> int:
        """Write what is pending, roll up the touched buckets, prune. Returns rows written."""
        with self._lock:
            samples, self._pending = self._pending, []
            requests, self._pending_requests = self._pending_requests, []
            if not samples and not requests:
                return 0
            try:
                db = self._db
                names = ", ".join(f'"{f}"' for f in self.fields)
                marks = ", ".join("?" for _ in self.fields)
                db.executemany(
                    f"INSERT OR REPLACE INTO m1 (t, {names}) VALUES (?, {marks})",
                    [
                        (int(t), *[_num(row.get(f)) for f in self.fields])
                        for t, row in samples
                    ],
                )
                for name, res, src in (("m10", 10, "m1"), ("m60", 60, "m10")):
                    buckets = sorted({int(t) // res * res for t, _ in samples})
                    for b in buckets:
                        self._rollup(name, src, b, res)
                if requests:
                    cols = ", ".join(REQUEST_COLUMNS)
                    q = ", ".join("?" for _ in REQUEST_COLUMNS)
                    db.executemany(
                        f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({q})",
                        [tuple(r[c] for c in REQUEST_COLUMNS) for r in requests],
                    )
                self._prune()
                db.commit()
            except sqlite3.Error as exc:
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("history store write failed: %s", exc)
                with contextlib.suppress(sqlite3.Error):
                    self._db.rollback()
                return 0
            return len(samples) + len(requests)

    def _rollup(self, dst: str, src: str, bucket: int, res: int) -> None:
        aggs = ", ".join(
            f'{"MAX" if f in MAX_FIELDS else "AVG"}("{f}")' for f in self.fields
        )
        names = ", ".join(f'"{f}"' for f in self.fields)
        self._db.execute(
            f"INSERT OR REPLACE INTO {dst} (t, {names}) "
            f"SELECT {int(bucket)}, {aggs} FROM {src} WHERE t >= ? AND t < ? GROUP BY 1",
            (bucket, bucket + res),
        )

    def _prune(self) -> None:
        now = self._clock()
        db = self._db
        for name, _res, keep in TIERS:
            horizon = keep if keep is not None else self.retention_days * 86400
            db.execute(f"DELETE FROM {name} WHERE t < ?", (int(now - horizon),))
        db.execute(
            "DELETE FROM requests WHERE t < ?", (now - self.retention_days * 86400,)
        )
        # byte cap: drop the oldest requests, then the oldest 1-minute rows, until it fits
        for _ in range(8):
            if self._size() <= self.max_bytes:
                break
            n = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if n:
                db.execute(
                    "DELETE FROM requests WHERE rowid IN "
                    "(SELECT rowid FROM requests ORDER BY t ASC LIMIT ?)",
                    (max(1, n // 4),),
                )
            else:
                db.execute(
                    "DELETE FROM m60 WHERE t IN (SELECT t FROM m60 ORDER BY t ASC LIMIT 1440)"
                )
            db.commit()
            db.execute("PRAGMA incremental_vacuum")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _size(self) -> int:
        try:
            page = self._db.execute("PRAGMA page_size").fetchone()[0]
            count = self._db.execute("PRAGMA page_count").fetchone()[0]
            free = self._db.execute("PRAGMA freelist_count").fetchone()[0]
            return int((count - free) * page)
        except sqlite3.Error:
            return 0

    def size_bytes(self) -> int:
        with self._lock:
            return self._size()

    # ── reading ─────────────────────────────────────────────────────────
    def tier_for(
        self, since: float, step: float | None, now: float | None = None
    ) -> str:
        """The finest table that still covers ``since`` and is no finer than ``step`` asks."""
        age = (now if now is not None else self._clock()) - since
        want = "m1"
        if step and step >= 60:
            want = "m60"
        elif step and step >= 10:
            want = "m10"
        order = [t[0] for t in TIERS]
        for name, _res, keep in TIERS:
            if order.index(name) < order.index(want):
                continue
            if keep is None or age <= keep:
                return name
        return "m60"

    def read(
        self,
        since: float | None = None,
        until: float | None = None,
        step: float | None = None,
    ) -> dict[str, Any]:
        self.flush()
        now = self._clock()
        until = now if until is None else until
        since = until - 900 if since is None else since
        tier = self.tier_for(since, step, now)
        res = TIER_RES[tier]
        names = ", ".join(f'"{f}"' for f in self.fields)
        with self._lock:
            rows = self._db.execute(
                f"SELECT t, {names} FROM {tier} WHERE t > ? AND t <= ? ORDER BY t",
                (int(since), int(until)),
            ).fetchall()
        t = [float(r[0]) for r in rows]
        cols = {f: [r[i + 1] for r in rows] for i, f in enumerate(self.fields)}
        out_res = res
        if step and step > res and rows:
            t, cols = _rebucket(t, cols, step, self.fields)
            out_res = step
        gaps = _gaps(t, out_res)
        return {
            "object": "yunshu.metrics_history",
            "tier": tier,
            "resolution_s": out_res,
            "since": since,
            "until": until,
            "fields": list(self.fields),
            "series": {
                "t": t,
                **{
                    f: [None if v is None else round(v, 3) for v in cols[f]]
                    for f in self.fields
                },
            },
            "gaps": gaps,
        }

    def requests_page(
        self, limit: int = 50, before: str | None = None, model: str | None = None
    ) -> dict[str, Any]:
        self.flush()
        boundary = None
        if before:
            try:
                value = json.loads(
                    base64.urlsafe_b64decode(before + "=" * (-len(before) % 4))
                )
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or _num(value[0]) is None
                    or not isinstance(value[1], str)
                ):
                    raise ValueError
                boundary = (float(value[0]), value[1])
            except (ValueError, TypeError):
                raise ValueError("invalid history cursor") from None
        where, args = [], []
        if boundary:
            where.append("(t, request_id) < (?, ?)")
            args += [boundary[0], boundary[1]]
        if model:
            where.append("model = ?")
            args.append(model)
        q = (
            f"SELECT {', '.join(REQUEST_COLUMNS)} FROM requests"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY t DESC, request_id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(q, (*args, limit + 1)).fetchall()
        page = rows[:limit]
        data = []
        for r in page:
            d = dict(zip(REQUEST_COLUMNS, r, strict=True))
            d["stream"] = bool(d["stream"])
            d["cancelled"] = bool(d["cancelled"])
            data.append(d)
        cursor = None
        if len(rows) > limit and page:
            last = page[-1]
            cursor = (
                base64.urlsafe_b64encode(json.dumps([last[1], last[0]]).encode())
                .decode()
                .rstrip("=")
            )
        return {"data": data, "count": len(data), "next_cursor": cursor}

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="yunshu-history-store", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.flush_s):
            self.flush()

    def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)
        self.flush()
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.close()


def _rebucket(t, cols, step, fields):
    buckets: dict[int, list[int]] = {}
    for k, tv in enumerate(t):
        buckets.setdefault(int(tv // step), []).append(k)
    out_t: list[float] = []
    out_c: dict[str, list[float | None]] = {f: [] for f in fields}
    for b in sorted(buckets):
        ks = buckets[b]
        out_t.append(float(b * step))
        for f in fields:
            vals = [cols[f][k] for k in ks if cols[f][k] is not None]
            if not vals:
                out_c[f].append(None)
            else:
                out_c[f].append(max(vals) if f in MAX_FIELDS else sum(vals) / len(vals))
    return out_t, out_c


def _gaps(t: list[float], res: float) -> list[list[float]]:
    """Spans with no samples at all (the process was not running): [from, to] in epoch seconds."""
    out = []
    for a, b in zip(t, t[1:], strict=False):
        if b - a > res * 2.5:
            out.append([a + res, b])
    return out
