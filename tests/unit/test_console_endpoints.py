"""Console enrichment endpoints: /v1/yunshu/{requests/recent,history,memory,config}.

CPU only: fake runner stats, fake engines, the real registry / settings / routers.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from yunshu_engine import settings
from yunshu_engine.vlm_batch_runner import RunStats
from yunshu_gateway import history, memory_ledger, x_yunshu
from yunshu_gateway.routers import yunshu as yunshu_router


@pytest.fixture(autouse=True)
def _clean():
    x_yunshu.registry.clear()
    memory_ledger.reset_host_cache()
    yield
    x_yunshu.registry.clear()
    settings.clear_overrides()
    memory_ledger._weight_cache.clear()


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(yunshu_router.router, prefix="/v1")
    return app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://test"
    )


def _finished(rid="r1", model="m", prompt="SECRET PROMPT TEXT", t_admit=0.2):
    info = x_yunshu.RequestInfo(
        request_id=rid, method="POST", path="/v1/chat/completions"
    )
    st = RunStats()
    base = info.arrived
    st.prompt_tokens, st.cached_tokens, st.generated = 1000, 400, 51
    st.t_submit, st.t_admit = base + 0.05, base + t_admit
    st.t_first, st.t_last = base + 0.9, base + 1.9
    st.finish_reason = "stop"
    st.cache_tier, st.cache_reload_ms = "ssd", 12.3
    info.gen = SimpleNamespace(stats=st, model=model, cancelled=False)
    info.status = 200
    info.t_done = base + 2.0
    info.usage = {"prompt_tokens": 1000, "completion_tokens": 51}
    stats = x_yunshu.build_stats(info)
    x_yunshu.record_done(info, stats)
    return info, stats


# ── B3 ──────────────────────────────────────────────────────────────────


def test_x_yunshu_carries_phase_offsets():
    _, stats = _finished()
    off = stats["offsets_ms"]
    assert off["arrive"] == 0.0
    assert off["admit"] == pytest.approx(200, abs=1)
    assert off["first_token"] == pytest.approx(900, abs=1)
    assert off["last_token"] == pytest.approx(1900, abs=1)
    assert off["done"] == pytest.approx(2000, abs=1)
    assert stats["t0_wall"] > 1e9


def test_offsets_none_for_unreached_phases():
    info = x_yunshu.RequestInfo(request_id="q", method="POST", path="/v1/x")
    off = x_yunshu.phase_offsets_ms(info)
    assert off == {
        "arrive": 0.0,
        "admit": None,
        "first_token": None,
        "last_token": None,
        "done": None,
    }


def test_progress_payload_has_offsets():
    info, _ = _finished()
    p = x_yunshu.progress_payload(info)
    assert p["offsets_ms"]["first_token"] == pytest.approx(900, abs=1)
    assert p["t0_wall"] == round(info.arrived_wall, 3)


@pytest.mark.asyncio
async def test_recent_endpoint_shape_whitelist_order_limit():
    _finished("a", "m1")
    time.sleep(0.01)
    _finished("b", "m2")
    time.sleep(0.01)
    _finished("c", "m1")
    async with _client() as c:
        r = (await c.get("/v1/yunshu/requests/recent")).json()
        assert [e["request_id"] for e in r["data"]] == ["c", "b", "a"]  # newest first
        e = r["data"][0]
        assert e["offsets_ms"]["first_token"] == pytest.approx(900, abs=1)
        assert e["finish_reason"] == "stop"
        assert e["cache"]["tier"] == "ssd" and e["cache"]["reload_ms"] == 12.3
        assert e["path"] == "/v1/chat/completions" and e["model"] == "m1"
        assert "SECRET" not in repr(r)  # numbers and enums only
        lim = (await c.get("/v1/yunshu/requests/recent?limit=2")).json()
        assert [e["request_id"] for e in lim["data"]] == ["c", "b"]
        mod = (await c.get("/v1/yunshu/requests/recent?model=m1")).json()
        assert [e["request_id"] for e in mod["data"]] == ["c", "a"]
        since = r["data"][1]["t"]
        news = (await c.get(f"/v1/yunshu/requests/recent?since={since}")).json()
        assert [e["request_id"] for e in news["data"]] == ["c"]
        assert (await c.get("/v1/yunshu/requests/recent?limit=0")).status_code == 400


# ── B2 ──────────────────────────────────────────────────────────────────


def test_ring_wraps_and_memory_is_fixed():
    ring = history.HistoryRing(10)
    nbytes = ring.nbytes
    assert nbytes == 10 * (8 + 4 * len(history.FIELDS))
    for i in range(37):
        ring.append(1000.0 + i, {"active_gb": float(i)})
    assert len(ring) == 10
    out = ring.read()
    assert out["t"] == [1000.0 + i for i in range(27, 37)]  # oldest first, last 10 kept
    assert out["active_gb"][-1] == 36.0
    assert out["decode_tps"] == [None] * 10  # NaN -> null
    assert ring.nbytes == nbytes
    assert len(ring._t) == 10 and all(len(v) == 10 for v in ring._cols.values())


def test_ring_since_and_step():
    ring = history.HistoryRing(100)
    for i in range(20):
        ring.append(1000.0 + i, {"decode_tps": 10.0 if i % 2 else None, "queued": 1.0})
    assert ring.read(since=1014.0)["t"] == [1015.0 + k for k in range(5)]
    b = ring.read(step=10)
    assert len(b["t"]) == 2
    assert b["queued"] == [1.0, 1.0]
    assert b["decode_tps"] == [10.0, 10.0]  # mean of the available samples only


def test_ring_capacity_from_hours_and_interval():
    s = history.Sampler(5.0, 12.0)
    assert s.ring.capacity == 8640
    assert s.ring.nbytes == 8640 * (8 + 4 * len(history.FIELDS)) < 1024 * 1024


def test_defaults_sample_once_a_second_into_an_hour_ring():
    from yunshu_engine import settings

    assert settings.get("YUNSHU_HISTORY_INTERVAL_S") == 1.0
    assert settings.get("YUNSHU_HISTORY_HOURS") == 1.0
    assert settings.get("YUNSHU_HISTORY_STORE") is True


def test_sampler_survives_a_collector_that_raises(monkeypatch):
    s = history.Sampler(5.0, 1.0)
    monkeypatch.setattr(
        history, "sample_row", lambda: (_ for _ in ()).throw(RuntimeError("x"))
    )
    s.sample_once(1000.0)
    assert s.errors == 1 and "RuntimeError" in s.last_error and len(s.ring) == 0
    monkeypatch.undo()
    s.sample_once(1005.0)
    assert len(s.ring) == 1


def test_sample_row_reads_registry_without_an_engine(monkeypatch):
    monkeypatch.setattr(
        memory_ledger,
        "mlx_counters",
        lambda: {"active": 4 * 2**30, "cache": 2**30, "peak": 5 * 2**30},
    )
    info = x_yunshu.RequestInfo(
        request_id="live", method="POST", path="/v1/chat/completions"
    )
    st = RunStats()
    now = time.perf_counter()
    st.t_admit, st.t_first, st.t_last, st.generated = now - 3, now - 2, now - 1, 101
    info.gen = SimpleNamespace(stats=st)
    x_yunshu.registry.add(info)
    queued = x_yunshu.RequestInfo(request_id="wait", method="POST", path="/v1/x")
    x_yunshu.registry.add(queued)
    _finished("done1")
    row = history.sample_row()
    assert row["active_gb"] == 4.0 and row["cache_gb"] == 1.0
    assert row["requests_active"] == 2 and row["queued"] == 1
    assert row["decode_tps"] == pytest.approx(100.0)
    assert row["ttft_p50_ms"] is not None
    x_yunshu.registry.clear()


@pytest.mark.asyncio
async def test_history_endpoint_disabled_and_enabled(monkeypatch):
    monkeypatch.setattr(history, "_SAMPLER", None)
    async with _client() as c:
        off = (await c.get("/v1/yunshu/history")).json()
        assert off["enabled"] is False and off["series"]["t"] == []
        s = history.Sampler(5.0, 1.0)
        for i in range(5):
            s.ring.append(1000.0 + 5 * i, {"queued": float(i)})
        monkeypatch.setattr(history, "_SAMPLER", s)
        on = (await c.get("/v1/yunshu/history")).json()
        assert on["enabled"] and on["ring"]["rows"] == 5
        assert on["ring"]["bytes"] == s.ring.nbytes
        assert on["series"]["queued"] == [0.0, 1.0, 2.0, 3.0, 4.0]
        since = (await c.get("/v1/yunshu/history?since=1010")).json()
        assert since["series"]["t"] == [1015.0, 1020.0]


def test_history_switched_off_by_setting(monkeypatch):
    monkeypatch.setattr(history, "_SAMPLER", None)
    settings.set_override("YUNSHU_HISTORY_INTERVAL_S", "0")
    assert history.start_from_settings() is None


@pytest.mark.asyncio
async def test_sampler_task_runs_and_stops(monkeypatch):
    monkeypatch.setattr(history, "sample_row", lambda: {"queued": 2.0})
    s = history.Sampler(0.01, 0.01)
    s.start()
    await __import__("asyncio").sleep(0.08)
    await s.stop()
    n = len(s.ring)
    assert n >= 2
    await __import__("asyncio").sleep(0.03)
    assert len(s.ring) == n  # stopped


# ── B1 ──────────────────────────────────────────────────────────────────

GB = 2**30


class _Eng:
    def __init__(self, apc=None, loaded=True):
        self._apc = apc
        self.is_loaded = loaded
        self._model = None

    def apc_snapshot(self):
        return self._apc


def _manager(*entries):
    return SimpleNamespace(list_entries=lambda: list(entries))


def _entry(mid, eng, est=0, loaded=True):
    return SimpleNamespace(
        model_id=mid, engine=eng, estimated_bytes=est, is_loaded=loaded
    )


def _fake_world(monkeypatch, active=20 * GB, cache=2 * GB):
    monkeypatch.setattr(
        memory_ledger,
        "mlx_counters",
        lambda: {"active": active, "cache": cache, "peak": 25 * GB},
    )
    monkeypatch.setattr(
        memory_ledger,
        "_host_uncached",
        lambda: {
            "total_gb": 128.0,
            "pressure_level": "normal",
            "swap_used_gb": 0.0,
            "swap_total_gb": 7.0,
            "wired_limit_gb": None,
            "available_gb": 60.0,
        },
    )
    monkeypatch.setattr(memory_ledger, "recommended_working_set", lambda: 100 * GB)
    monkeypatch.setattr(
        memory_ledger,
        "process_footprint",
        lambda: {"footprint": 22 * GB, "footprint_peak": None},
    )


def test_ledger_owners_sum_and_unknowns_are_null(monkeypatch):
    _fake_world(monkeypatch)
    monkeypatch.setattr(memory_ledger, "measured_weight_bytes", lambda e: None)
    apc = {
        "resident_bytes": 3 * GB,
        "memory_max_bytes": 8 * GB,
        "warm_bytes": 1 * GB,
        "warm_max_bytes": 4 * GB,
        "storage_tiers": [
            {
                "name": "ssd0",
                "used_bytes": 5 * GB,
                "cap_bytes": 50 * GB,
                "available": True,
            }
        ],
    }
    mgr = _manager(
        _entry("big", _Eng(apc), est=14 * GB),
        _entry("off", _Eng(), 9 * GB, loaded=False),
    )
    led = memory_ledger.collect(mgr, None)
    by = {(o["kind"], o["id"]): o for o in led["owners"]}
    assert (
        by[("weights", "big")]["bytes"] == 14 * GB
        and by[("weights", "big")]["estimated"] is True
    )
    assert ("weights", "off") not in by  # unloaded models hold nothing
    assert by[("apc_ram", "big")]["bytes"] == 3 * GB
    assert by[("apc_warm", "big")]["bytes"] == 1 * GB
    assert by[("live_kv", None)]["bytes"] is None  # no source: null, not 0
    assert by[("mlx_cache", None)]["bytes"] == 2 * GB
    other = by[("other", None)]
    assert other["estimated"] is True and other["bytes"] == 20 * GB - (14 + 3 + 1) * GB
    # owners (excluding mlx_cache, which is outside active) sum to active
    assert (
        sum(o["bytes"] or 0 for o in led["owners"] if o["kind"] != "mlx_cache")
        == 20 * GB
    )
    assert (
        led["limits"]["apc_max_gb"] == 8.0 and led["limits"]["apc_warm_max_gb"] == 4.0
    )
    assert (
        led["mlx"]["peak_gb"] == 25.0
        and led["mlx"]["recommended_working_set_gb"] == 100.0
    )
    assert (
        led["host"]["pressure_level"] == "normal"
        and led["host"]["wired_limit_gb"] is None
    )
    assert led["free_gb"] == 60.0 and led["total_gb"] == 128.0
    assert led["storage_tiers"][0]["tier"] == "ssd0"


def test_ledger_over_attribution_is_reported_not_hidden(monkeypatch):
    _fake_world(monkeypatch, active=5 * GB)
    monkeypatch.setattr(memory_ledger, "measured_weight_bytes", lambda e: 9 * GB)
    led = memory_ledger.collect(_manager(_entry("m", _Eng())), None)
    other = next(o for o in led["owners"] if o["kind"] == "other")
    assert other["bytes"] == 0
    assert led["attribution_overshoot_gb"] == 4.0
    w = next(o for o in led["owners"] if o["kind"] == "weights")
    assert w["estimated"] is False and w["source"] == "model parameters"


def test_ledger_no_model_and_no_mlx(monkeypatch):
    _fake_world(monkeypatch)
    monkeypatch.setattr(
        memory_ledger,
        "mlx_counters",
        lambda: {"active": None, "cache": None, "peak": None},
    )
    led = memory_ledger.collect(None, None)
    kinds = [o["kind"] for o in led["owners"]]
    assert "weights" not in kinds
    assert next(o for o in led["owners"] if o["kind"] == "other")["bytes"] is None
    assert led["mlx"]["active_gb"] is None and led["host"]["pressure_level"] == "normal"


def test_host_is_cached_for_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(
        memory_ledger, "_host_uncached", lambda: calls.append(1) or {"total_gb": 1.0}
    )
    memory_ledger.host(now=100.0)
    memory_ledger.host(now=110.0)
    assert len(calls) == 1
    memory_ledger.host(now=116.0)
    assert len(calls) == 2


def test_real_host_reads_do_not_guess():
    h = memory_ledger._host_uncached()
    assert set(h) >= {"total_gb", "pressure_level", "swap_used_gb", "wired_limit_gb"}
    for v in h.values():
        assert v is None or isinstance(v, (int, float, str))
    if h["total_gb"] is not None:
        assert h["total_gb"] > 1


@pytest.mark.asyncio
async def test_memory_endpoint(monkeypatch):
    _fake_world(monkeypatch)
    async with _client() as c:
        r = await c.get("/v1/yunshu/memory")
        assert r.status_code == 200
        j = r.json()
        assert j["object"] == "yunshu.memory" and j["mlx"]["active_gb"] == 20.0


# ── B5 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_endpoint_sources_and_masking():
    settings.set_override("YUNSHU_QUEUE_LIMIT", "7")
    settings.set_override("YUNSHU_BRAVE_API_KEY", "hunter2-super-secret")
    async with _client() as c:
        r = await c.get("/v1/yunshu/config")
        assert r.status_code == 200
        assert "hunter2" not in r.text
        j = r.json()
        rows = {s["name"]: s for s in j["settings"]}
        q = rows["YUNSHU_QUEUE_LIMIT"]
        assert q["value"] == 7 and q["source"] == "cli" and q["default"] == 64
        assert q["type"] == "int" and q["category"] == "server"
        assert rows["YUNSHU_BRAVE_API_KEY"]["value"] == "***"
        cli = {s["name"]: s for s in settings.effective(("stable",))}
        assert all(
            rows[n]["source"] == cli[n]["source"] for n in cli
        )  # same as the CLI
        assert all(len(s["description"]) <= 240 for s in j["settings"])
        assert all(s["stability"] == "stable" for s in j["settings"])
        allr = (await c.get("/v1/yunshu/config?include=all")).json()
        assert len(allr["settings"]) > len(j["settings"])
        assert {s["stability"] for s in allr["settings"]} >= {"stable", "experimental"}
        assert (
            allr["experimental_count"]
            <= allr["experimental_max"]
            == settings.MAX_EXPERIMENTAL
        )
        assert (await c.get("/v1/yunshu/config?include=bogus")).status_code == 422


def test_effective_rows_carry_default_type_choices():
    row = next(
        r for r in settings.effective(("stable",)) if r["name"] == "YUNSHU_SPEC_TREE"
    )
    assert (
        row["default"] == "auto" and row["type"] == "enum" and "tree" in row["choices"]
    )


# ── auth ────────────────────────────────────────────────────────────────

PATHS = [
    "/v1/yunshu/requests/recent",
    "/v1/yunshu/history",
    "/v1/yunshu/memory",
    "/v1/yunshu/config",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_routes_follow_the_token(path):
    settings.set_override("YUNSHU_AUTH_DISABLED", "false")
    settings.set_override("YUNSHU_AUTH_TOKEN", "tok")
    async with _client() as c:
        assert (await c.get(path)).status_code == 401
        assert (
            await c.get(path, headers={"Authorization": "Bearer nope"})
        ).status_code == 401
        ok = await c.get(path, headers={"Authorization": "Bearer tok"})
        assert ok.status_code == 200


# ── cost per poll ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_cost_is_small(capsys, monkeypatch):
    """Every handler, 500 requests worth of state, full ring; no GPU. Prints the mean."""
    for i in range(500):
        _finished(f"r{i}", "m")
    s = history.Sampler(5.0, 12.0)
    for i in range(s.ring.capacity):
        s.ring.append(1e9 + 5 * i, {f: float(i % 97) for f in history.FIELDS})
    monkeypatch.setattr(history, "_SAMPLER", s)
    eng = _Eng({"resident_bytes": 3 * GB, "memory_max_bytes": 8 * GB})
    fake_mgr = _manager(_entry("m", eng, 14 * GB))
    results = {}
    async with _client() as c:
        for path in PATHS + ["/v1/yunshu/history?step=60", "/v1/yunshu/status"]:
            monkeypatch.setattr(
                yunshu_router,
                "get_model_manager",
                (lambda: fake_mgr) if path.endswith("/memory") else (lambda: None),
            )
            await c.get(path)
            n = 30
            t0 = time.perf_counter()
            for _ in range(n):
                r = await c.get(path)
                assert r.status_code == 200
            results[path] = (time.perf_counter() - t0) / n * 1000
            results[path + " bytes"] = len(r.content)
    t0 = time.perf_counter()
    for _ in range(20):
        memory_ledger._host_uncached()
    results["host() uncached (sysctl+psutil)"] = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(50):
        history.sample_row()
    results["sampler tick (sample_row)"] = (time.perf_counter() - t0) / 50 * 1000
    with capsys.disabled():
        for k, v in results.items():
            print(f"POLLCOST {k}: {v:.2f}")
    for path in PATHS:
        assert results[path] < 50, (
            results
        )  # generous CI bound; real numbers are printed
    assert not math.isnan(results["/v1/yunshu/config"])


def test_measured_weight_bytes_walks_parameters():
    arr = lambda n: SimpleNamespace(nbytes=n)  # noqa: E731
    model = SimpleNamespace(
        parameters=lambda: {"a": arr(100), "layers": [{"w": arr(50)}, {"w": arr(25)}]}
    )
    eng = SimpleNamespace(_model=model)
    assert memory_ledger.measured_weight_bytes(eng) == 175
    assert memory_ledger.measured_weight_bytes(SimpleNamespace()) is None


def test_ledger_reports_binary_gb_with_exact_bytes(monkeypatch):
    """GB = 1024**3 and every *_gb has an exact *_bytes sibling (engine memunits contract)."""
    _fake_world(monkeypatch)
    led = memory_ledger.collect(_manager(_entry("m", _Eng())), None)
    assert led["mlx"]["active_gb"] == 20.0
    assert led["mlx"]["active_bytes"] == 20 * GB
    assert led["mlx"]["recommended_working_set_bytes"] == 100 * GB
    assert led["process"]["footprint_bytes"] == 22 * GB
    assert led["process"]["footprint_peak_bytes"] is None
    assert "apc_max_bytes" in led["limits"]
    assert memory_ledger.gb(128 * GB, 1) == 128.0
    out: dict = {}
    memory_ledger.put(out, "x", None)
    assert out == {"x_gb": None, "x_bytes": None}


def test_prefill_progress_waits_for_the_cache_hit_instead_of_jumping():
    """Until the cache hit is known the prompt is counted whole, so a percentage would start low and
    jump when the hit arrives; the payload then carries no progress rather than a wrong one."""
    from yunshu_engine.vlm_batch_runner import RunStats

    info = x_yunshu.RequestInfo(
        request_id="p1", method="POST", path="/v1/chat/completions"
    )
    st = RunStats(prompt_tokens=10_000)
    now = time.perf_counter()
    st.t_submit = st.t_admit = now - 1
    st.prefill_total, st.prefill_done = 10_000, 500
    st.prefill_known = False
    info.gen = SimpleNamespace(stats=st)
    p = x_yunshu.progress_payload(info)
    assert p["phase"] == "prefill"
    assert p["percent"] is None and p["processed_tokens"] is None and p["eta_s"] is None
    assert p["prompt_tokens"] == 10_000
    # The hit arrives: 8,000 cached, 2,000 to compute, 500 done.
    st.cached_tokens, st.prefill_total, st.prefill_known = 8_000, 2_000, True
    p = x_yunshu.progress_payload(info)
    assert p["cached_tokens"] == 8_000
    assert p["percent"] == 25.0
    assert p["processed_tokens"] == 8_500
