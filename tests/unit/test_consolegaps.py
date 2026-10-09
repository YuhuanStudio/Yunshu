import json
from types import SimpleNamespace

from yunshu_gateway.serve_log import ServeLog


def test_history_metadata_projection_and_paging(tmp_path):
    log = ServeLog(tmp_path, 4096, 2)
    for i in range(3):
        log.append(
            {
                "t_end": 100 + i,
                "model": "m",
                "request_id": f"r{i}",
                "prompt": "SECRET",
                "latency": {"prompt": "SECRET"},
            }
        )
    page = log.history(limit=2, before=None, retention_days=0)
    assert [r["request_id"] for r in page["data"]] == ["r2", "r1"]
    assert "SECRET" not in json.dumps(page)
    assert (
        log.history(limit=2, before=page["next_cursor"], retention_days=0)["data"][0][
            "request_id"
        ]
        == "r0"
    )


def test_chain_and_tree_depth_counts():
    from yunshu_engine.spec_metrics import record_depth

    st = SimpleNamespace(spec_depth_drafted=[], spec_depth_accepted=[])
    record_depth(st, 2, 3)
    record_depth(st, 2, 4, parents=[-1, 0, 0, 1, 2])
    assert st.spec_depth_drafted == [3, 3, 1]
    assert st.spec_depth_accepted == [2, 2, 0]


def test_new_routes_admin_auth_and_openapi(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yunshu_gateway.routers.yunshu import router

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    with TestClient(app) as client:
        for path in (
            "cache",
            "bundle",
            "bundle/manifest",
            "spec-decode",
            "models/impact?model=x",
        ):
            assert client.get("/v1/yunshu/" + path).status_code == 401
        assert client.post("/v1/yunshu/cache/clear").status_code == 401
        schema = app.openapi()["paths"]
        assert all(
            "/v1/yunshu/" + path in schema
            for path in (
                "cache",
                "requests/history",
                "bundle",
                "bundle/manifest",
                "spec-decode",
                "models/impact",
                "cache/clear",
            )
        )


def test_history_route_uses_retention_and_cursor(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yunshu_gateway import serve_log
    from yunshu_gateway.routers.yunshu import router

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    monkeypatch.setenv("YUNSHU_SERVE_LOG_RETENTION_DAYS", "0")
    log = ServeLog(tmp_path, 4096, 1)
    log.append({"t_end": 1, "model": "m", "request_id": "r", "messages": ["SECRET"]})
    monkeypatch.setattr(serve_log, "get_log", lambda: log)
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    with TestClient(app) as client:
        assert (
            client.get("/v1/yunshu/requests/history").json()["data"][0]["request_id"]
            == "r"
        )
        assert (
            client.get("/v1/yunshu/requests/history?before=broken").status_code == 400
        )
        assert client.get("/v1/yunshu/requests/history?limit=0").status_code == 400
        monkeypatch.setattr(serve_log, "get_log", lambda: None)
        assert client.get("/v1/yunshu/requests/history").json()["enabled"] is False


def test_event_ring_and_entry_metadata():
    from yunshu_engine.cache_observation import Observation, ObservedCache

    obs = Observation()
    obs.request_id = "r"
    cache = ObservedCache(obs)
    entry = SimpleNamespace(token_ids=[1, 2])
    cache[42] = entry
    obs.reason = "ttl"
    cache.pop(42)
    assert obs.events[-1]["reason"] == "ttl"
    assert obs.events[-1]["request_id"] == "r"
    for i in range(600):
        obs.event("admission", i, 2, "ram", "stored")
    assert len(obs.events) == 512
    obs.hit(42)
    row = obs.row(42, "namespace", 2, 100, 50, "ram")
    assert row["hits"] == 1 and row["last_hit"] > 0
    assert "token_ids" not in row


def test_bound_spec_rounds_are_attributed(monkeypatch):
    from yunshu_engine import spec_metrics as sm

    calls = []
    monkeypatch.setattr(sm, "_original", lambda *a: calls.append(a))
    monkeypatch.setattr(sm, "_totals", {})
    st = SimpleNamespace(
        spec_mode="dflash", spec_depth_drafted=[], spec_depth_accepted=[]
    )
    sm.bind(st)
    try:
        sm.record(None, 2, 3)
    finally:
        sm.bind(None)
    sm.record(None, 1, 3)
    assert len(calls) == 2
    total = sm.snapshot()["data"][0]
    assert total["num_drafts"] == 1 and total["num_accepted_tokens"] == 2
    assert total["per_depth"][2]["accepted"] == 0


def test_model_impact_predicts_only_safe_victims():
    from yunshu_engine.model_manager import ModelManager

    mgr = ModelManager(max_models=1)
    mgr.register_model("old", "/old", estimated_bytes=100)
    mgr.register_model("new", "/new", estimated_bytes=100)
    old = mgr.get_entry("old")
    old.is_loaded = True
    old.engine = SimpleNamespace(has_active_requests=lambda: False)
    preview = mgr.console_impact("new")
    assert preview["load"]["would_evict"] == ["old"]
    assert preview["unload"] == {
        "in_flight_policy": "reject",
        "waits": False,
        "interrupts": False,
    }
    old.leases = 1
    assert mgr.console_impact("new")["load"]["blocked"] is True
    assert old.is_loaded  # preview does not mutate the manager


def test_bundle_route_shares_cli_builder(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yunshu_cli import bundle
    from yunshu_gateway.routers.yunshu import router

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    monkeypatch.setattr(
        bundle, "build", lambda **kw: {"version": "test", "excluded": "prompts"}
    )
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    with TestClient(app) as client:
        result = client.get("/v1/yunshu/bundle")
        assert result.json()["version"] == "test"
        assert "attachment" in result.headers["content-disposition"]
        assert (
            client.get("/v1/yunshu/bundle/manifest").json()["generator"]
            == "yunshu_cli.bundle.build"
        )


def test_route_probe_on_cpu_fake_server(monkeypatch, tmp_path):
    """Exercise the exact probe's success and failure rules before any GPU job."""
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from yunshu_cli import bundle
    from yunshu_engine import mlx_executor
    from yunshu_gateway import serve_log
    from yunshu_gateway.routers import monitoring, yunshu

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
    import pytest
    from route_checks import Fail, console_backend_gaps

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    log = ServeLog(tmp_path, 4096, 2)
    monkeypatch.setattr(serve_log, "get_log", lambda: log)
    obs = {
        "entries": [
            {
                "id": "1",
                "namespace": "n",
                "tokens": 600,
                "bytes_logical": 100,
                "bytes_physical": 100,
                "tier": "ram",
                "last_hit": None,
                "hits": 0,
            }
        ],
        "events": [
            {
                "id": 1,
                "t": time.time(),
                "action": "admission",
                "entry_id": "1",
                "tokens": 600,
                "tier": "ram",
                "reason": "stored",
                "request_id": "consolegaps-schema",
            }
        ],
    }

    def clear():
        obs["entries"].clear()

    engine = SimpleNamespace(
        _apc_backend=SimpleNamespace(console_snapshot=lambda: obs, clear=clear),
        has_active_requests=lambda: False,
    )
    monkeypatch.setattr(monitoring, "_collect_engines", lambda *a: [("m", engine)])
    monkeypatch.setattr(
        yunshu,
        "get_model_manager",
        lambda: SimpleNamespace(
            console_impact=lambda model: {
                "object": "yunshu.model.impact",
                "model": model,
                "unload": {
                    "in_flight_policy": "reject",
                    "waits": False,
                    "interrupts": False,
                },
                "load": {
                    "would_evict": [],
                    "blocked": False,
                    "advisory": True,
                    "post_load_pressure": "rechecked_after_load",
                },
            }
        ),
    )
    monkeypatch.setattr(
        bundle,
        "build",
        lambda **kw: {k: None for k in yunshu.bundle_manifest()["included"]},
    )
    app = FastAPI()
    app.include_router(yunshu.router, prefix="/v1")
    enforced = True

    @app.post("/v1/chat/completions")
    async def generate(request: Request):
        await request.json()
        log.append(
            {
                "t_end": time.time(),
                "model": "m",
                "request_id": request.headers["X-Request-Id"],
            }
        )
        return {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "x_yunshu": {
                "model": "m",
                "structured_output": {
                    "enforced": enforced,
                    "grammar_backend": "llguidance",
                },
                "speculative": None,
            },
        }

    with ThreadPoolExecutor(1) as pool, TestClient(app) as client:
        monkeypatch.setattr(mlx_executor, "get_mlx_executor", lambda: pool)
        ctx = SimpleNamespace(http=client, model="m", notes={})
        console_backend_gaps(ctx)
        assert ctx.notes["consolegaps"]["cache_entries"] == 1
        enforced = False
        with pytest.raises(Fail, match="schema enforcement"):
            console_backend_gaps(ctx)


def test_observed_cache_preserves_pop_and_lru_key_contract():
    import pytest

    from yunshu_engine.cache_observation import Observation, ObservedCache

    cache = ObservedCache(Observation())
    with pytest.raises(KeyError):
        cache.pop("missing")
    entry = SimpleNamespace(token_ids=[1])
    cache["opaque-prefix-key"] = entry
    cache.move_to_end("opaque-prefix-key")
    assert cache.last_access_key == "opaque-prefix-key"
    assert cache.pop("missing", None) is None


def test_console_identity_uses_long_contexts_and_short_bound():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify.suites import parse_suite

    cfg = parse_suite("console-identity")
    assert cfg["ctx"] == [1024, 32768]
    assert cfg["identity_timeout_min"] == 10
    assert cfg["spec_off"] and cfg["decode_tokens"] == 256


def test_disk_events_use_explicit_origin_and_unknown_not_previous_request():
    from yunshu_engine.cache_observation import Observation, disk_event

    obs = Observation()
    obs.request_id = "unrelated-current-request"
    store = SimpleNamespace(
        observation=obs, event_origin=lambda key: None, name="internal"
    )
    disk_event(store, "admission", 42, 300, "ssd_write")
    assert obs.events[-1]["request_id"] is None
    store.event_origin = lambda key: "captured-generation"
    disk_event(store, "admission", 42, 300, "ssd_write")
    assert obs.events[-1]["request_id"] == "captured-generation"


def test_history_keeps_full_length_http_request_ids(tmp_path):
    from yunshu_gateway.serve_log import event_from_stats

    rid = "r" * 128
    log = ServeLog(tmp_path, 4096, 1)
    log.append(event_from_stats({}, {"request_id": rid, "t_end": 100}))
    log.append(event_from_stats({}, {"request_id": rid, "t_end": 101}))
    page = log.history(limit=1, before=None, retention_days=0)
    assert page["data"][0]["request_id"] == rid
    assert (
        log.history(limit=1, before=page["next_cursor"], retention_days=0)["data"][0][
            "t"
        ]
        == 100
    )


def test_identity_verdict_rejects_missing_or_inconsistent_depth_evidence():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify.stages import _spec_depth_valid

    spec = {
        "mode": "mtp",
        "rounds": 1,
        "drafted": 2,
        "accepted": 1,
        "per_depth": [
            {"position": 0, "drafted": 1, "accepted": 1},
            {"position": 1, "drafted": 1, "accepted": 0},
        ],
    }
    rows = [{"xy": {"speculative": spec}}]
    assert _spec_depth_valid(rows, "mtp") == (True, "")
    assert not _spec_depth_valid([], "mtp")[0]
    assert not _spec_depth_valid(rows, "dflash")[0]
    spec["drafted"] = 3
    assert not _spec_depth_valid(rows, "mtp")[0]
