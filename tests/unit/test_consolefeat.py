"""Console backend contracts, independent of model weights or OS probe availability."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import yunshu


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    app = FastAPI()
    app.include_router(yunshu.router, prefix="/v1")
    return TestClient(app)


def test_host_parsers():
    from yunshu_gateway.host_state import parse_power, parse_thermal

    assert parse_thermal("CPU_Speed_Limit = 70")["state"] == "throttled"
    assert parse_thermal("no data")["state"] == "unknown"
    assert parse_power("Now drawing from 'AC Power'")["battery_percent"] is None
    assert (
        parse_power("Now drawing from 'Battery Power'\n 55%; discharging;")[
            "battery_percent"
        ]
        == 55
    )
    assert parse_power("bad")["state"] == "unknown"


def test_host_cache_and_fail_soft(client, monkeypatch):
    from yunshu_gateway import host_state

    calls = []

    def fail(args):
        calls.append(args)
        raise FileNotFoundError("not installed")

    monkeypatch.setattr(host_state, "_run", fail)
    monkeypatch.setattr(host_state, "_CACHED", None)
    a = client.get("/v1/yunshu/host")
    b = client.get("/v1/yunshu/host")
    assert a.status_code == 200
    assert a.json() == b.json()
    assert len(calls) == 3
    for key in ("thermal", "power", "memory_pressure"):
        assert a.json()[key]["state"] == "unknown"
        assert a.json()[key]["reason"]


def test_admin_auth(client, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    assert client.get("/v1/yunshu/host").status_code == 401
    assert client.get("/v1/yunshu/requests/recent").status_code == 401
    assert (
        client.post(
            "/v1/yunshu/models/register", json={"model": "test", "path": "/missing"}
        ).status_code
        == 401
    )
    assert (
        client.post("/v1/yunshu/models/cancel", json={"model": "test"}).status_code
        == 401
    )
    assert client.delete("/v1/yunshu/models/register/test").status_code == 401


def test_register_unload_only(client, monkeypatch, tmp_path):
    from yunshu_engine.model_manager import ModelManager

    manager = ModelManager()
    monkeypatch.setattr(
        "yunshu_gateway.ollama_models.get_model_manager", lambda: manager
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (tmp_path / "model.safetensors").write_bytes(b"weight-fixture")
    body = {"model": "local/model", "path": str(tmp_path)}
    r = client.post("/v1/yunshu/models/register", json=body)
    assert r.status_code == 200, r.text
    assert not manager.get_entry("local/model").is_loaded
    assert client.post("/v1/yunshu/models/register", json=body).status_code == 409
    manager.get_entry("local/model").is_loading = True
    assert client.delete("/v1/yunshu/models/register/local/model").status_code == 409
    manager.get_entry("local/model").is_loading = False
    assert client.delete("/v1/yunshu/models/register/local/model").status_code == 200
    assert (tmp_path / "model.safetensors").exists()
    assert client.delete("/v1/yunshu/models/register/local/model").status_code == 404


@pytest.mark.parametrize(
    "config,index",
    [
        ({}, None),
        ({"model_type": "qwen2"}, {"weight_map": {"x": "missing.safetensors"}}),
        ({"model_type": "qwen2"}, {"weight_map": {}}),
        ({"model_type": "qwen2"}, {"weight_map": {"x": "../outside.safetensors"}}),
    ],
)
def test_register_invalid(client, monkeypatch, tmp_path, config, index):
    from yunshu_engine.model_manager import ModelManager

    monkeypatch.setattr(
        "yunshu_gateway.ollama_models.get_model_manager", lambda: ModelManager()
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    if index is not None:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert (
        client.post(
            "/v1/yunshu/models/register", json={"model": "test", "path": str(tmp_path)}
        ).status_code
        == 400
    )


def test_cancel_route(client, monkeypatch):
    from yunshu_engine.model_manager import ModelManager

    manager = ModelManager()
    manager.register_model("test", "/missing")
    monkeypatch.setattr(
        "yunshu_gateway.ollama_models.get_model_manager", lambda: manager
    )
    assert (
        client.post("/v1/yunshu/models/cancel", json={"model": "test"}).status_code
        == 404
    )
    manager.get_entry("test").is_loading = True
    assert client.post("/v1/yunshu/models/cancel", json={"model": "test"}).json()[
        "load"
    ]
    assert "test" in manager._cancelled_loads


@pytest.mark.asyncio
async def test_cancel_load_drains_and_wakes_waiters(monkeypatch):
    import asyncio

    from yunshu_engine.model_manager import ModelManager

    manager = ModelManager()
    manager.register_model("test", "/missing", estimated_bytes=123)
    started, finish = asyncio.Event(), asyncio.Event()
    engine = SimpleNamespace(stop=AsyncMock())

    async def create(*args):
        started.set()
        await finish.wait()
        return engine

    monkeypatch.setattr(manager, "_create_and_load_engine", create)
    monkeypatch.setattr(manager, "_get_mlx_executor", lambda: None)
    loader = asyncio.create_task(manager.get_engine("test"))
    await started.wait()
    waiter = asyncio.create_task(manager.get_engine("test"))
    await asyncio.sleep(0)
    assert manager.cancel_load("test")
    finish.set()
    results = await asyncio.wait_for(
        asyncio.gather(loader, waiter, return_exceptions=True), 2
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    engine.stop.assert_awaited_once()
    entry = manager.get_entry("test")
    assert not entry.is_loading and not entry.is_loaded and entry.engine is None
    assert manager._current_memory_bytes == 0
    assert not manager._loading_events and not manager._cancelled_loads


def test_latency_fake_clock(client, monkeypatch):
    from yunshu_gateway import x_yunshu as x

    marks = {
        "model_lease_start": 10.001,
        "model_lease": 10.006,
        "gateway_admit": 10.002,
        "template_start": 10.010,
        "template_end": 10.030,
        "sse_first_flush": 10.110,
    }
    st = SimpleNamespace(
        t_submit=10.030,
        t_admit=10.040,
        t_prefill_end=10.090,
        t_first=10.100,
        latency_marks={"prefill_start": 10.050},
        cache_reload_ms=10,
    )
    info = x.RequestInfo(
        "fake",
        "POST",
        "/v1/chat/completions",
        arrived=10,
        latency_marks=marks,
        gen=SimpleNamespace(stats=st),
    )
    monkeypatch.setattr(x.time, "perf_counter", lambda: 10.2)
    latency = x.latency_breakdown(info)
    assert latency["durations_ms"] == {
        "model_lease": 5.0,
        "gateway_admit": 2.0,
        "engine_queue": 10.0,
        "template_tokenize": 20.0,
        "apc_lookup_restore": 10,
        "prefill": 40.0,
        "first_decode": 10.0,
        "sse_first_flush": 10.0,
    }
    missing = x.latency_breakdown(x.RequestInfo("missing", "POST", "/", arrived=10))
    assert all(v is None for v in missing["durations_ms"].values())
    x.registry.clear()
    x.registry.record_done(
        {"t": x.time.time(), "request_id": "fake", "latency": latency}
    )
    try:
        assert (
            client.get("/v1/yunshu/requests/recent").json()["data"][0]["latency"]
            == latency
        )
        assert client.get("/v1/yunshu/requests/recent?limit=513").status_code == 400
    finally:
        x.registry.clear()


class _BlockingHub:
    """A hub whose download waits for the job's cancel, like a transfer in progress."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.started = __import__("threading").Event()

    def list_files(self, repo, revision, patterns):
        return [("config.json", 10)]

    def cache_dir(self):
        return self.tmp

    def download(self, repo, revision, patterns, local_dir, on_file, on_bytes):
        import time

        on_file("config.json", 10, 0)
        self.started.set()
        for _ in range(200):
            on_bytes("config.json", 0)  # raises DownloadCancelledError once cancelled
            time.sleep(0.01)
        raise AssertionError("cancel did not stop download")


def test_cancel_download_hook(tmp_path):
    from yunshu_gateway import downloads as dl
    from yunshu_gateway import ollama_models as om

    hub = _BlockingHub(tmp_path)
    reg = dl.DownloadRegistry(hub)
    dl.set_registry(reg)
    try:
        job = reg.submit("org/model")
        assert hub.started.wait(2)
        assert om.cancel_download("org/model")
        assert job.wait(3)
        assert job.state == "cancelled"
        assert not om.cancel_download("org/model")
        assert not om.cancel_download("missing")
    finally:
        dl.set_registry(None)


def test_probe_cpu_schema():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "consolefeat_probe",
        Path(__file__).parents[2] / "scripts/research/consolefeat_routes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.parser()
        .parse_args(["--model", "/fixture", "--src", "python", "--out", "out"])
        .model
        == "/fixture"
    )
    with pytest.raises(AssertionError, match="missing latency"):
        module.validate_latency({"milestones_ms": {}})
    keys = (
        "gateway_receive",
        "gateway_admit",
        "model_lease",
        "template_start",
        "template_end",
        "engine_submit",
        "engine_admit",
        "first_decode",
        "sse_first_flush",
    )
    ds = dict.fromkeys(
        (
            "model_lease",
            "gateway_admit",
            "template_tokenize",
            "engine_queue",
            "sse_first_flush",
        ),
        1,
    )
    assert (
        module.validate_latency(
            {"milestones_ms": dict.fromkeys(keys, 0), "durations_ms": ds}
        )
        == ds
    )


def test_malformed_index_is_client_error(client, monkeypatch, tmp_path):
    from yunshu_engine.model_manager import ModelManager

    monkeypatch.setattr(
        "yunshu_gateway.ollama_models.get_model_manager", lambda: ModelManager()
    )
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    (tmp_path / "model.safetensors.index.json").write_text("[]")
    assert (
        client.post(
            "/v1/yunshu/models/register", json={"model": "test", "path": str(tmp_path)}
        ).status_code
        == 400
    )


def test_fast_path_stage_marks_preserved(monkeypatch):
    from yunshu_engine.fast_path_stats import FastPathStats
    from yunshu_gateway import x_yunshu as x

    clock = iter([10.0, 10.05, 10.09, 10.10])
    monkeypatch.setattr(
        "yunshu_engine.fast_path_stats.time.perf_counter", lambda: next(clock)
    )
    fp = FastPathStats(None, 100)
    fp.stats.latency_marks.update(
        engine_admit=10.01, apc_start=10.02, apc_end=10.03, prefill_start=10.04
    )
    fp.admit(0, 100)
    fp.progress(100, 100)
    fp.token(1)
    info = x.RequestInfo(
        "fast", "POST", "/", arrived=10.0, gen=SimpleNamespace(stats=fp.stats)
    )
    durations = x.latency_breakdown(info)["durations_ms"]
    assert durations["engine_queue"] == 10.0
    assert durations["apc_lookup_restore"] == 10.0
    assert durations["prefill"] == 50.0
    assert durations["first_decode"] == 10.0


def test_single_engine_acquisition_marks(monkeypatch):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway import engine as gateway_engine
    from yunshu_gateway.x_yunshu import RequestInfo

    info = RequestInfo("single", "POST", "/v1/chat/completions", arrived=10.0)
    engine = SimpleNamespace(is_loaded=True)
    monkeypatch.setattr(gateway_engine, "_engine", engine)
    token = current_request_info.set(info)
    try:
        assert gateway_engine.get_engine() is engine
        first = dict(info.latency_marks)
        assert "model_lease_start" in first and "model_lease" in first
        gateway_engine.get_engine()
        assert info.latency_marks == first
    finally:
        current_request_info.reset(token)


def test_probe_retries_busy_port(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "consolefeat_retry",
        Path(__file__).parents[2] / "scripts/research/consolefeat_routes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    calls = []

    def start():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("no free port in 18990-18996")
        return "server"

    assert module.retry_server(start) == "server" and len(calls) == 3
    with pytest.raises(RuntimeError, match="load failed"):
        module.retry_server(lambda: (_ for _ in ()).throw(RuntimeError("load failed")))


def test_register_hf_snapshot_shard_symlinks(client, monkeypatch, tmp_path):
    from yunshu_engine.model_manager import ModelManager

    manager = ModelManager()
    monkeypatch.setattr(
        "yunshu_gateway.ollama_models.get_model_manager", lambda: manager
    )
    blobs = tmp_path / "models--org--test" / "blobs"
    snapshot = tmp_path / "models--org--test" / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "configblob").write_text('{"model_type":"qwen2"}')
    (blobs / "weightblob").write_bytes(b"weights")
    (snapshot / "config.json").symlink_to(blobs / "configblob")
    (snapshot / "model.safetensors").symlink_to(blobs / "weightblob")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model.safetensors"}}'
    )
    r = client.post(
        "/v1/yunshu/models/register", json={"model": "hf-model", "path": str(snapshot)}
    )
    assert r.status_code == 200, r.text
    assert not manager.get_entry("hf-model").is_loaded
    assert manager.get_entry("hf-model").estimated_bytes == int(len(b"weights") * 1.8)
    assert client.delete("/v1/yunshu/models/register/hf-model").status_code == 200
    assert (blobs / "weightblob").exists() and (
        snapshot / "model.safetensors"
    ).is_symlink()


@pytest.mark.parametrize("absolute", [False, True])
def test_index_cannot_name_files_outside_model(client, monkeypatch, tmp_path, absolute):
    from yunshu_engine.model_manager import ModelManager

    monkeypatch.setattr(
        "yunshu_gateway.ollama_models.get_model_manager", lambda: ModelManager()
    )
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (root / "config.json").write_text('{"model_type":"qwen2"}')
    shard = str(outside) if absolute else "../outside.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )
    assert (
        client.post(
            "/v1/yunshu/models/register", json={"model": "test", "path": str(root)}
        ).status_code
        == 400
    )


def test_console_stage_accepts_gpuq_device_fields(tmp_path):
    # gpuq adds device / execution_device / remote_host to the checks record; the judge
    # failed a real all-PASS run (consolefeat-tiny-final3) because it required an exact key set.
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify import stages

    checks = {
        "console_registration_cancel": "PASS",
        "console_host_latency": "PASS",
        "stream_latency": "PASS",
        "single_stream_latency": "PASS",
        "console_backend_gaps": "PASS",
        "history_restart": "PASS",
        "device": "m5",
        "execution_device": "m5",
        "remote_host": None,
    }
    path = tmp_path / "console.jsonl"
    path.write_text(
        json.dumps({"complete": True, "failures": [], "checks": checks}) + "\n"
    )
    assert stages._console_validate(path) == (True, "")
    checks["stream_latency"] = "FAIL"
    path.write_text(
        json.dumps({"complete": True, "failures": [], "checks": checks}) + "\n"
    )
    ok, why = stages._console_validate(path)
    assert not ok and "stream_latency" in why
