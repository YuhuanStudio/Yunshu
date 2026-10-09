"""The console process's recorder: one cheap read a second, outages kept as gaps and events, and a
request cursor that neither skips nor repeats across engine restarts and console restarts."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from yunshu_console.poller import FIELDS, Poller, memory_gib, row_from_status
from yunshu_console.store import HistoryStore

T0 = 1_800_000_000.0


class FakeEngine:
    """Answers the endpoints the recorder reads, from a state the test mutates."""

    def __init__(self) -> None:
        self.up = True
        self.boot_id = "boot-a"
        self.pid = 4242
        self.uptime = 100.0
        self.seq = 0
        self.ring: list[dict] = []  # newest last
        self.models = [{"id": "org/m", "loaded": True}]
        self.load_error: str | None = None
        self.calls: list[str] = []
        self.token: str | None = None
        self.seen_auth: list[str | None] = []

    def finish(self, rid: str, t: float, **kw) -> None:
        self.seq += 1
        self.ring.append(
            {
                "seq": self.seq,
                "request_id": rid,
                "t": t,
                "model": "org/m",
                "path": "/v1/chat/completions",
                "status": 200,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 40,
                "ttft_ms": 150.0,
                "decode_tps": 40.0,
                **kw,
            }
        )

    def restart(self) -> None:
        self.boot_id = "boot-b"
        self.pid += 1
        self.uptime = 1.0
        self.seq = 0
        self.ring = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.path)
        self.seen_auth.append(request.headers.get("authorization"))
        if not self.up:
            raise httpx.ConnectError("refused", request=request)
        if (
            self.token
            and request.headers.get("authorization") != f"Bearer {self.token}"
        ):
            return httpx.Response(401, json={"error": "no"})
        path = request.url.path
        if path == "/v1/yunshu/status":
            return httpx.Response(
                200,
                json={
                    "object": "yunshu.status",
                    "version": "t",
                    "state": "running",
                    "uptime_s": self.uptime,
                    "pid": self.pid,
                    "load_error": self.load_error,
                    "models": self.models,
                    "memory": {
                        "active_bytes": 20 * 1024**3,
                        "cache_bytes": 2 * 1024**3,
                        "peak_bytes": 24 * 1024**3,
                        "pressure": 0.4,
                    },
                    "requests": {
                        "active": 1,
                        "queued": 0,
                        "items": [{"phase": "decode", "tokens_per_second": 50.0}],
                    },
                    "throughput": {"live_decode_tps": 50.0},
                },
            )
        if path == "/v1/yunshu/requests/recent":
            after = request.url.params.get("after_seq")
            rows = [r for r in self.ring if after is None or r["seq"] > int(after)]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": list(reversed(rows)),
                    "boot_id": self.boot_id,
                    "latest_seq": self.seq,
                },
            )
        if path == "/v1/yunshu/host":
            return httpx.Response(
                200,
                json={
                    "telemetry": {
                        "watts": {"gpu": 12.5, "package": 20.0},
                        "gpu": {"frequency_mhz": 1200, "active_ratio": 0.5},
                        "temperature": {"die_max_c": 61},
                    }
                },
            )
        return httpx.Response(404)


class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(
    tmp_path, engine: FakeEngine, clock: Clock, **kw
) -> tuple[Poller, HistoryStore]:
    store = HistoryStore(tmp_path / "c.sqlite", FIELDS, clock=clock, flush_s=3600)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(engine.handler), base_url="http://engine"
    )
    return Poller("http://engine", store, client=client, clock=clock, **kw), store


def run(coro):
    return asyncio.run(coro)


def count(store: HistoryStore, table: str) -> int:
    store.flush()
    return store._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_one_tick_records_a_row_the_requests_and_the_models(tmp_path):
    engine, clock = FakeEngine(), Clock()
    engine.finish("req_1", T0 - 5)
    poller, store = make(tmp_path, engine, clock)
    run(poller.tick())
    assert count(store, "m1") == 1 and count(store, "requests") == 1
    row = store._db.execute(
        "SELECT active_gb, peak_gb, decode_tps, requests_active, gpu_w, die_c FROM m1"
    ).fetchone()
    assert row == (20.0, 24.0, 50.0, 1.0, 12.5, 61.0)
    kinds = [e["kind"] for e in store.events_between(0, T0 + 10)]
    assert kinds == ["model_loaded"]
    assert poller.up is True and poller.engine["models"] == ["org/m"]
    assert set(engine.calls) == {
        "/v1/yunshu/status",
        "/v1/yunshu/requests/recent",
        "/v1/yunshu/host",
    }


def test_the_host_telemetry_is_read_every_five_seconds_not_every_tick(tmp_path):
    engine, clock = FakeEngine(), Clock()
    poller, _ = make(tmp_path, engine, clock)
    for _ in range(4):
        run(poller.tick())
        clock.t += 1
    assert engine.calls.count("/v1/yunshu/host") == 1
    clock.t += 5
    run(poller.tick())
    assert engine.calls.count("/v1/yunshu/host") == 2


def test_an_outage_is_a_gap_and_two_events_with_how_long(tmp_path):
    engine, clock = FakeEngine(), Clock()
    poller, store = make(tmp_path, engine, clock)
    run(poller.tick())
    engine.up = False
    for _ in range(5):
        clock.t += 1
        run(poller.tick())  # five failed reads: still one "unreachable" event
    assert poller.up is False and poller.since == T0 + 1
    engine.up = True
    clock.t += 90
    run(poller.tick())
    clock.t += 1
    run(poller.tick())
    out = store.read(since=T0 - 1, until=T0 + 200)
    kinds = [(e["kind"], e["detail"]) for e in out["events"]]
    assert [k for k, _ in kinds] == [
        "model_loaded",
        "engine_unreachable",
        "engine_reachable",
    ]
    assert kinds[2][1] == {"down_s": 94.0}
    assert out["gaps"] and out["gaps"][0][0] == T0 + 1 and out["gaps"][0][1] == T0 + 95
    assert len(out["series"]["t"]) == 3, (
        "no rows were written while the engine was away"
    )


def test_a_gateway_that_answers_with_an_error_is_an_engine_error_not_unreachable(
    tmp_path,
):
    engine, clock = FakeEngine(), Clock()
    engine.token = "secret"
    poller, store = make(tmp_path, engine, clock)  # no token: the engine says 401
    run(poller.tick())
    assert poller.up is False and poller.error_status == 401
    assert [e["kind"] for e in store.events_between(0, T0 + 1)] == ["engine_error"]


def test_the_token_is_sent_to_the_engine(tmp_path):
    engine, clock = FakeEngine(), Clock()
    engine.token = "secret"
    poller, _ = make(tmp_path, engine, clock, token="secret")
    run(poller.tick())
    assert poller.up is True
    assert set(engine.seen_auth) == {"Bearer secret"}


def test_an_engine_restart_is_an_event_and_a_load_error_is_recorded_once(tmp_path):
    engine, clock = FakeEngine(), Clock()
    poller, store = make(tmp_path, engine, clock)
    run(poller.tick())
    engine.restart()
    engine.models = [{"id": "org/m", "loaded": False}]
    engine.load_error = "out of memory while loading"
    for _ in range(3):
        clock.t += 1
        run(poller.tick())
    kinds = [e["kind"] for e in store.events_between(0, T0 + 10)]
    assert kinds.count("engine_restarted") == 1
    assert kinds.count("load_error") == 1
    assert "model_unloaded" in kinds
    msg = [e for e in store.events_between(0, T0 + 10) if e["kind"] == "load_error"][0]
    assert "out of memory" in msg["detail"]["message"]


def test_the_cursor_neither_skips_nor_repeats(tmp_path):
    engine, clock = FakeEngine(), Clock()
    poller, store = make(tmp_path, engine, clock)
    run(poller.tick())
    for i in range(5):
        engine.finish(f"req_{i}", T0 + i)
    clock.t += 1
    run(poller.tick())
    assert count(store, "requests") == 5
    clock.t += 1
    run(poller.tick())  # nothing new: nothing written again
    assert poller.requests_recorded == 5
    engine.finish("req_5", T0 + 6)
    clock.t += 1
    run(poller.tick())
    store.flush()
    ids = [
        r[0] for r in store._db.execute("SELECT request_id FROM requests ORDER BY t")
    ]
    assert ids == [f"req_{i}" for i in range(6)]
    # the second read after the first request used the cursor, not the whole ring
    assert poller.seq == 6


def test_across_an_engine_restart_new_requests_are_taken_from_the_start_of_the_new_sequence(
    tmp_path,
):
    engine, clock = FakeEngine(), Clock()
    poller, store = make(tmp_path, engine, clock)
    for i in range(3):
        engine.finish(f"a{i}", T0 + i)
    run(poller.tick())
    engine.restart()
    engine.finish("b0", T0 + 10)  # seq 1 again: a plain cursor would have skipped it
    engine.finish("b1", T0 + 11)
    clock.t += 5
    run(poller.tick())
    store.flush()
    ids = {r[0] for r in store._db.execute("SELECT request_id FROM requests")}
    assert ids == {"a0", "a1", "a2", "b0", "b1"}


def test_a_console_restart_rereads_the_ring_and_writes_nothing_twice(tmp_path):
    engine, clock = FakeEngine(), Clock()
    for i in range(4):
        engine.finish(f"r{i}", T0 + i)
    first, store = make(tmp_path, engine, clock)
    run(first.tick())
    store.close()
    second, store2 = make(tmp_path, engine, clock)  # same file, a fresh cursor
    run(second.tick())
    assert count(store2, "requests") == 4


def test_nothing_but_metadata_reaches_the_request_log(tmp_path):
    engine, clock = FakeEngine(), Clock()
    engine.finish(
        "req_x", T0, messages=[{"content": "SECRET PROMPT"}], output="SECRET OUTPUT"
    )
    poller, store = make(tmp_path, engine, clock)
    run(poller.tick())
    store.close()
    blob = b""
    for suffix in ("", "-wal"):
        f = tmp_path / f"c.sqlite{suffix}"
        if f.exists():
            blob += f.read_bytes()
    assert b"SECRET" not in blob


def test_row_from_status_uses_exact_bytes_and_converts_legacy_decimal_gb():
    assert memory_gib({"active_bytes": 2 * 1024**3}, "active") == 2.0
    assert memory_gib({"active_gb": 1.0}, "active") == pytest.approx(0.931, abs=1e-3)
    assert memory_gib({}, "active") is None
    row = row_from_status(
        {
            "memory": {"active_gb": 10.0},
            "requests": {
                "active": 2,
                "queued": 1,
                "items": [
                    {"phase": "decode", "tokens_per_second": 30.0},
                    {"phase": "decode", "tokens_per_second": 20.5},
                    {"phase": "prefill", "tokens_per_second": 900.0},
                ],
            },
            "throughput": {"live_decode_tps": None},
        },
        [
            {
                "ttft_ms": 100.0,
                "prompt_tokens": 600,
                "completion_tokens": 60,
                "cached_tokens": 120,
            }
        ],
    )
    assert row["decode_tps"] == 50.5 and row["prefill_tps"] == 900.0
    assert row["active_gb"] == pytest.approx(9.313, abs=1e-3)
    assert row["ttft_p50_ms"] == 100.0 and row["prompt_tokens_per_s"] == 10.0
    assert row["cached_tokens_per_s"] == 2.0 and row["gpu_w"] is None


def test_idle_is_recorded_as_zero_not_as_unknown_and_an_outage_as_no_row():
    idle = row_from_status(
        {
            "memory": {"active_gb": 1.0},
            "requests": {"active": 0, "queued": 0, "items": []},
            "throughput": {"live_decode_tps": None},
        },
        [],
    )
    assert idle["decode_tps"] == 0.0 and idle["prefill_tps"] == 0.0
    assert idle["requests_active"] == 0.0


def test_a_tick_that_raises_does_not_end_the_loop(tmp_path):
    engine, clock = FakeEngine(), Clock()
    poller, _ = make(tmp_path, engine, clock, interval_s=0.25)
    calls = {"n": 0}
    real = poller.tick

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        await real()

    poller.tick = flaky  # type: ignore[method-assign]

    async def go():
        task = asyncio.create_task(poller.run())
        await asyncio.sleep(0.7)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(go())
    assert calls["n"] >= 2 and poller.errors == 1 and poller.up is True
