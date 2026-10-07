"""Cache browser: per-entry hit accounting, tier overview and clear on the real APC manager,
plus the /v1/yunshu/cache routes. Accounting only: lookups must return what they did before."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from yunshu_engine.apc_manager import (  # noqa: E402
    ENTRY_HITS_MAX,
    SpillDiskStore,
    YunshuAPCManager,
)
from yunshu_gateway.routers import admin_cache  # noqa: E402

IM_START, USER = 900, 901


def _cache(n: int):
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 4, 8)), mx.ones((1, 2, 3)) * n]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    mx.eval(rec.cache, kv.keys, kv.values)
    return [rec, kv]


def _mgr(tmp_path=None, **kw):
    disk = (
        SpillDiskStore(tmp_path, namespace="t", num_workers=1, max_bytes=1 << 30)
        if tmp_path
        else None
    )
    return YunshuAPCManager(
        num_blocks=8,
        block_size=16,
        disk=disk,
        overrides={"memory_max_gb": 1},
        head_marker=(IM_START, USER),
        **kw,
    )


def _ids(i):
    return list(range(1000 * i, 1000 * i + 96))


def _store(m, ids):
    m.begin_request()
    m.store_exact_cache(ids, _cache(96))


def test_overview_counts_hits_per_entry_and_never_exposes_tokens():
    m = _mgr()
    a, b = _ids(1), _ids(2)
    _store(m, a)
    _store(m, b)
    for _ in range(3):
        _, n = m.lookup_exact_cache(a + [5, 6])
        assert n == 96
    m.lookup_exact_cache([7] * 100)  # a miss
    ov = m.cache_overview()
    ram = next(t for t in ov["tiers"] if t["name"] == "ram")
    assert ram["entries"] == 2 and ram["used_bytes"] > 0 and ram["cap_bytes"] == 1 << 30
    assert ov["lookups"] == {"hit": 3, "miss": 1, "by_tier": {"ram": 3}}
    by_hits = sorted(ov["entries"], key=lambda e: -e["hits"])
    assert [e["hits"] for e in by_hits] == [3, 0]
    assert by_hits[0]["tokens"] == 96 and by_hits[0]["bytes"] > 0
    assert by_hits[0]["tier"] == "ram" and by_hits[0]["last_hit_age_s"] is not None
    assert by_hits[1]["last_hit_age_s"] is None
    assert sorted(e["lru_rank"] for e in ov["entries"]) == [0, 1]
    # the entry just hit is the most recently used
    assert by_hits[0]["lru_rank"] == 0
    blob = json.dumps(ov)
    assert "1000" not in blob.replace('"cap_bytes"', "")  # no token ids
    assert all(
        set(e) >= {"key", "tokens", "bytes", "tier", "lru_rank", "hits"}
        for e in ov["entries"]
    )


def test_accounting_does_not_change_what_a_lookup_returns():
    plain, watched = _mgr(), _mgr()
    for m in (plain, watched):
        for i in (1, 2, 3):
            _store(m, _ids(i))
    for i in (1, 2, 3, 4):
        q = _ids(i) + [1, 2]
        _, n1 = plain.lookup_exact_cache(q)
        watched.cache_overview()  # reading the side table in between
        _, n2 = watched.lookup_exact_cache(q)
        assert n1 == n2 == (96 if i < 4 else 0)
    assert dict(plain.tier_hits) == dict(watched.tier_hits)


def test_entries_are_capped_and_flagged_truncated():
    m = _mgr(max_entries=16)
    for i in range(1, 6):
        _store(m, _ids(i))
    ov = m.cache_overview(max_entries=2)
    assert len(ov["entries"]) == 2 and ov["entries_truncated"] is True
    assert ov["tiers"][0]["entries"] == 5 - (5 - len(m._exact_cache))
    assert m.cache_overview(max_entries=0)["entries"] == []


def test_hit_side_table_is_bounded():
    m = _mgr()
    for i in range(ENTRY_HITS_MAX + 50):
        m._note_entry_hit((i, i + 1), 0)
    assert len(m._entry_hits) == ENTRY_HITS_MAX


def test_clear_ram_frees_bytes_keeps_counters_and_misses_afterwards():
    m = _mgr()
    a = _ids(1)
    _store(m, a)
    m.lookup_exact_cache(a + [1])
    before = m.cache_overview()
    res = m.clear_tier("ram")
    assert (
        res["entries"] == 1
        and res["freed_bytes"] == before["tiers"][0]["used_bytes"] > 0
    )
    after = m.cache_overview()
    assert after["tiers"][0]["entries"] == 0 and after["entries"] == []
    assert after["lookups"]["hit"] == 1  # counters survive a clear
    _, n = m.lookup_exact_cache(a + [1])
    assert n == 0
    _store(m, a)  # and the cache works again
    _, n = m.lookup_exact_cache(a + [1])
    assert n == 96


def test_ssd_tier_is_listed_and_clear_drops_the_files(tmp_path):
    m = _mgr(tmp_path, max_entries=2)
    for i in (1, 2, 3):
        _store(m, _ids(i))  # the third one pushes the LRU entry to the SSD
    m.disk.flush()
    ssd = next(t for t in m.cache_overview()["tiers"] if t["name"] == "ssd")
    assert ssd["entries"] == 1 and ssd["used_bytes"] > 0
    res = m.clear_tier("ssd")
    assert res["entries"] == 1 and res["freed_bytes"] > 0 and res["busy"] == 0
    assert not list(tmp_path.rglob("*.safetensors"))
    ssd = next(t for t in m.cache_overview()["tiers"] if t["name"] == "ssd")
    assert ssd["entries"] == 0
    assert m.clear_tier("warm") == {"tier": "warm", "entries": 0, "freed_bytes": 0}
    with pytest.raises(ValueError):
        m.clear_tier("nope")


# ── routes ─────────────────────────────────────────────────────────────


class _Engine:
    def __init__(self, apc):
        self.apc = apc
        self.thread = None

    def apc_overview(self, n=200):
        return self.apc.cache_overview(n)

    def apc_clear(self, tier):
        import threading

        self.thread = threading.current_thread().name
        return self.apc.clear_tier(tier)


@pytest.fixture
def client(monkeypatch):
    apc = _mgr()
    eng = _Engine(apc)
    entry = SimpleNamespace(model_id="m1", is_loaded=True, engine=eng)
    monkeypatch.setattr(
        admin_cache,
        "get_model_manager",
        lambda: SimpleNamespace(list_entries=lambda: [entry]),
    )
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(admin_cache.router, prefix="/v1")
    return TestClient(app), apc, eng


def test_cache_route_reports_tiers_and_entries(client):
    c, apc, _ = client
    _store(apc, _ids(1))
    apc.lookup_exact_cache(_ids(1) + [1])
    j = c.get("/v1/yunshu/cache").json()
    assert j["enabled"] is True and j["caches"][0]["model"] == "m1"
    cache = j["caches"][0]
    assert cache["tiers"][0]["name"] == "ram" and cache["entries"][0]["hits"] == 1
    assert c.get("/v1/yunshu/cache?entries=0").json()["caches"][0]["entries"] == []
    assert c.get("/v1/yunshu/cache?entries=9999").status_code == 422


def test_clear_route_runs_on_the_mlx_thread_and_reports_freed_bytes(client):
    c, apc, eng = client
    _store(apc, _ids(1))
    r = c.post("/v1/yunshu/cache/clear", json={"tier": "ram"})
    assert r.status_code == 200
    j = r.json()
    assert j["freed_bytes"] > 0 and j["cleared"][0]["tier"] == "ram"
    assert eng.thread and eng.thread != "MainThread"
    assert apc.cache_overview()["tiers"][0]["entries"] == 0
    assert c.post("/v1/yunshu/cache/clear", json={"tier": "gpu"}).status_code == 422
    assert c.post("/v1/yunshu/cache/clear", json={"model": "nope"}).status_code == 404
    # no body clears every tier
    assert c.post("/v1/yunshu/cache/clear").status_code == 200


def test_cache_routes_without_a_prefix_cache(monkeypatch):
    monkeypatch.setattr(
        admin_cache,
        "get_model_manager",
        lambda: SimpleNamespace(
            list_entries=lambda: [
                SimpleNamespace(model_id="t", is_loaded=True, engine=object())
            ]
        ),
    )
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(admin_cache.router, prefix="/v1")
    j = TestClient(app).get("/v1/yunshu/cache").json()
    assert j == {"caches": [], "enabled": False}


def test_clear_needs_admin(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(admin_cache.router, prefix="/v1")
    c = TestClient(app)
    assert c.post("/v1/yunshu/cache/clear", json={}).status_code == 401
    assert c.get("/v1/yunshu/cache").status_code == 200
