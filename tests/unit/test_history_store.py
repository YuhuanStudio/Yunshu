"""The persistent history store: rollups, retention, restart persistence, the size cap, gaps,
request metadata only, and the proof that sampling stays off the generation thread."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

import pytest

from yunshu_gateway import history as history_mod
from yunshu_gateway.history_store import HistoryStore, request_row

FIELDS = ("decode_tps", "peak_gb", "rss_gb")
T0 = 1_800_000_000  # a round second well in the past of any clock the tests set


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def mk(tmp_path: Path, clock: Clock | None = None, **kw) -> HistoryStore:
    return HistoryStore(
        tmp_path / "h.sqlite",
        FIELDS,
        clock=clock or Clock(T0 + 7200),
        flush_s=3600,
        **kw,
    )


def feed(
    store: HistoryStore, start: int, seconds: int, tps=lambda i: float(i % 10)
) -> None:
    for i in range(seconds):
        store.add_sample(
            start + i, {"decode_tps": tps(i), "peak_gb": 10 + i, "rss_gb": 5.0}
        )
    store.flush()


def test_rollups_average_and_peaks_take_the_max(tmp_path):
    s = mk(tmp_path)
    feed(s, T0, 120)  # two full minutes
    m10 = s._db.execute("SELECT t, decode_tps, peak_gb FROM m10 ORDER BY t").fetchall()
    assert len(m10) == 12
    assert m10[0] == (T0, 4.5, 19)  # mean of 0..9, max of 10..19
    m60 = s._db.execute("SELECT t, decode_tps, peak_gb FROM m60 ORDER BY t").fetchall()
    assert len(m60) == 2 and m60[0][0] == T0 // 60 * 60
    s.close()


def test_null_fields_stay_null_and_are_not_zero(tmp_path):
    s = mk(tmp_path)
    for i in range(20):
        s.add_sample(T0 + i, {"decode_tps": None, "peak_gb": 1.0, "rss_gb": None})
    s.flush()
    out = s.read(since=T0 - 1, until=T0 + 100)
    assert all(v is None for v in out["series"]["decode_tps"])
    assert out["series"]["peak_gb"][0] == 1.0
    s.close()


def test_read_picks_the_tier_that_covers_the_range(tmp_path):
    clock = Clock(T0 + 30 * 3600)
    s = mk(tmp_path, clock)
    feed(s, T0 + 3600 * 5, 600)  # 25 hours old: only the 1-minute tier still has it
    clock.t = T0 + 3600 * 5 + 600 + 25 * 3600
    assert s.tier_for(clock.t - 600, None) == "m1"
    assert s.tier_for(clock.t - 7200, None) == "m10"
    assert s.tier_for(clock.t - 2 * 86400, None) == "m60"
    assert s.tier_for(clock.t - 600, step=60) == "m60"
    out = s.read(since=T0 + 3600 * 5 - 1, until=T0 + 3600 * 5 + 700, step=60)
    assert out["tier"] == "m60" and out["resolution_s"] == 60
    s.close()


def test_retention_prunes_each_tier_by_its_own_window(tmp_path):
    clock = Clock(T0 + 600)
    s = mk(tmp_path, clock, retention_days=1)
    feed(s, T0, 600)
    clock.t = T0 + 600 + 2 * 3600  # two hours later: m1 (1 h) is gone, m10 (24 h) stays
    s.add_sample(clock.t, {"decode_tps": 1.0, "peak_gb": 1, "rss_gb": 1})
    s.flush()
    count = lambda tb: s._db.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0]  # noqa: E731
    assert count("m1") == 1
    assert count("m10") >= 60
    clock.t = T0 + 600 + 2 * 86400  # past the retention: even m60 is empty
    s.add_sample(clock.t, {"decode_tps": 1.0, "peak_gb": 1, "rss_gb": 1})
    s.flush()
    assert count("m10") == 1 and count("m60") == 1
    s.close()


def test_survives_a_restart_and_keeps_the_gap_as_a_gap(tmp_path):
    clock = Clock(T0 + 400)
    s = mk(tmp_path, clock)
    feed(s, T0, 60)
    s.close()
    s2 = mk(tmp_path, clock)  # the process came back after the engine was down
    feed(s2, T0 + 300, 60)
    out = s2.read(since=T0 - 1, until=T0 + 400)
    assert len(out["series"]["t"]) == 120
    assert out["gaps"] == [[T0 + 60, T0 + 300]]
    s2.close()


def test_the_size_cap_trims_the_oldest_rows(tmp_path):
    clock = Clock(T0 + 10)
    s = mk(tmp_path, clock, max_bytes=120_000, retention_days=365)
    rows = [
        {
            "request_id": f"req_{i}",
            "t": T0 + i,
            "model": "m",
            "status": 200,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "key_name": "x" * 60,
        }
        for i in range(4000)
    ]
    for r in rows:
        s.add_request(r)
    s.flush()
    assert s.size_bytes() <= 120_000
    page = s.requests_page(limit=5)
    assert page["data"][0]["request_id"] == "req_3999", "the newest survive"
    assert s._db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] < 4000
    s.close()


def test_requests_hold_metadata_only(tmp_path):
    row = request_row(
        {
            "request_id": "req_1",
            "t": T0,
            "model": "org/m",
            "path": "/v1/chat/completions",
            "status": 200,
            "stream": True,
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "cached_tokens": 4,
            "ttft_ms": 120.5,
            "decode_tps": 40.0,
            "finish_reason": "stop",
            "key_name": "laptop",
            # none of these may reach the file:
            "messages": [{"role": "user", "content": "SECRET PROMPT"}],
            "output": "SECRET OUTPUT",
            "authorization": "Bearer SECRET",
            "client": "10.0.0.7",
        }
    )
    assert row is not None
    assert set(row) == set(
        [
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
        ]
    )
    s = mk(tmp_path)
    s.add_request(row)
    s.close()
    raw = (tmp_path / "h.sqlite").read_bytes()
    for wal in (tmp_path / "h.sqlite-wal",):
        if wal.exists():
            raw += wal.read_bytes()
    assert b"SECRET" not in raw and b"10.0.0.7" not in raw
    cols = {
        r[1]
        for r in sqlite3.connect(tmp_path / "h.sqlite").execute(
            "PRAGMA table_info(requests)"
        )
    }
    assert not {"messages", "output", "prompt", "content", "authorization"} & cols


def test_request_rows_reject_unusable_ids_and_odd_values():
    assert request_row({"request_id": "bad id\n", "t": 1}) is None
    assert request_row({"request_id": "ok", "t": float("nan")}) is None
    r = request_row(
        {"request_id": "ok", "t": 1, "model": "a b\x00", "ttft_ms": float("inf")}
    )
    assert r and r["model"] is None and r["ttft_ms"] is None


def test_requests_are_paged_newest_first_with_a_cursor(tmp_path):
    s = mk(tmp_path)
    for i in range(7):
        s.add_request(
            {"request_id": f"r{i}", "t": T0 + i, "model": "m" if i % 2 else "n"}
        )
    first = s.requests_page(limit=3)
    assert [r["request_id"] for r in first["data"]] == ["r6", "r5", "r4"]
    second = s.requests_page(limit=3, before=first["next_cursor"])
    assert [r["request_id"] for r in second["data"]] == ["r3", "r2", "r1"]
    last = s.requests_page(limit=3, before=second["next_cursor"])
    assert [r["request_id"] for r in last["data"]] == ["r0"] and last[
        "next_cursor"
    ] is None
    assert [r["request_id"] for r in s.requests_page(limit=9, model="m")["data"]] == [
        "r5",
        "r3",
        "r1",
    ]
    with pytest.raises(ValueError):
        s.requests_page(before="nonsense")
    s.close()


def test_a_step_larger_than_the_tier_rebuckets(tmp_path):
    s = mk(tmp_path)
    feed(s, T0, 100)
    out = s.read(since=T0 - 1, until=T0 + 100, step=30)
    assert out["resolution_s"] == 30
    assert len(out["series"]["t"]) in (3, 4)
    s.close()


def test_the_writer_thread_flushes_in_the_background(tmp_path):
    s = HistoryStore(tmp_path / "w.sqlite", FIELDS, flush_s=0.05, clock=Clock(T0 + 100))
    s.start()
    s.add_sample(T0 + 1, {"decode_tps": 1.0, "peak_gb": 1, "rss_gb": 1})
    for _ in range(100):
        if s._db.execute("SELECT COUNT(*) FROM m1").fetchone()[0]:
            break
        threading.Event().wait(0.05)
    assert s._db.execute("SELECT COUNT(*) FROM m1").fetchone()[0] == 1
    s.close()


# ── the sampler stays off the generation thread ──────────────────────────


def test_sampling_reads_counters_on_the_calling_thread_only(monkeypatch, tmp_path):
    """sample_row() runs wherever it is called (the gateway event loop in production), calls no
    engine and no executor, and the MLX thread is never entered."""
    seen: set[int] = set()
    from yunshu_gateway import memory_ledger as ml

    real = ml.mlx_counters

    def counters():
        seen.add(threading.get_ident())
        return real()

    monkeypatch.setattr(ml, "mlx_counters", counters)
    from yunshu_engine import mlx_executor

    monkeypatch.setattr(
        mlx_executor,
        "get_mlx_executor",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the sampler touched the MLX executor")
        ),
    )
    row = history_mod.sample_row()
    assert set(row) == set(history_mod.FIELDS)
    assert seen == {threading.get_ident()}


def test_the_history_modules_never_name_the_generation_machinery():
    root = Path(history_mod.__file__).parent
    for name in ("history.py", "history_store.py"):
        text = (root / name).read_text()
        for banned in (
            "mlx_executor",
            "run_in_executor",
            "generation_stream",
            "mx.eval",
            "get_engine",
        ):
            assert banned not in text, f"{name} mentions {banned}"
        assert not re.search(r"\bmx\.(eval|async_eval|synchronize)\b", text)


def test_sample_row_cost_is_microseconds():
    import time

    history_mod.sample_row()  # warm imports
    t = time.perf_counter()
    for _ in range(200):
        history_mod.sample_row()
    per_call_ms = (time.perf_counter() - t) / 200 * 1000
    print(f"HISTORY sample_row: {per_call_ms:.3f} ms/call")
    assert per_call_ms < 5, (
        "a once-a-second sample must stay far under a millisecond budget"
    )


def test_json_payload_is_plain(tmp_path):
    s = mk(tmp_path)
    feed(s, T0, 30)
    json.dumps(s.read(since=T0 - 1, until=T0 + 30))
    s.close()
