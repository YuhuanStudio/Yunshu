"""Download manager, local inventory and fit-before-load. A fake hub: no network."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_engine import paths
from yunshu_gateway import downloads as dl
from yunshu_gateway.routers import admin_models as am
from yunshu_gateway.routers import ollama

CONFIG = json.dumps(
    {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
).encode()


class FakeHub:
    """Writes real files in chunks, calling the progress hooks like the hub's tqdm bars."""

    def __init__(self, files=None, chunk=100, cache=None):
        self.files = files or {"config.json": CONFIG, "model.safetensors": b"x" * 1000}
        self.chunk = chunk
        self.cache = cache
        self.written = 0
        self.first_chunk = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.listed = 0

    def list_files(self, repo, revision, patterns):
        self.listed += 1
        if repo.endswith("/missing"):
            raise RuntimeError("404 repository not found")
        return [(n, len(b)) for n, b in self.files.items() if dl.matches(n, patterns)]

    def cache_dir(self):
        return self.cache or paths.models_dir()

    def download(self, repo, revision, patterns, local_dir, on_file, on_bytes):
        out = (
            Path(local_dir) if local_dir else self.cache_dir() / repo.replace("/", "--")
        )
        out.mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            if not dl.matches(name, patterns):
                continue
            final = out / name
            # like the hub: bytes land in a .incomplete file, renamed when complete
            part = out / ".cache/huggingface/download" / (name + ".incomplete")
            part.parent.mkdir(parents=True, exist_ok=True)
            have = part.stat().st_size if part.exists() else 0
            on_file(name, len(data), have)
            while have < len(data):
                self.gate.wait(5)
                piece = data[have : have + self.chunk]
                with part.open("ab") as fh:
                    fh.write(piece)
                have += len(piece)
                self.written += len(piece)
                on_bytes(name, len(piece))
                self.first_chunk.set()
            part.rename(final)
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path / "models")
    (tmp_path / "models").mkdir()
    hub = FakeHub(cache=tmp_path / "hfcache")
    reg = dl.DownloadRegistry(hub)
    dl.set_registry(reg)
    monkeypatch.setattr(am, "get_model_manager", lambda: None)
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(am.router, prefix="/v1")
    yield TestClient(app), hub, reg, tmp_path
    dl.set_registry(None)


def wait_state(client, job_id, states, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/v1/yunshu/downloads/{job_id}").json()
        if j["state"] in states:
            return j
        time.sleep(0.01)
    raise AssertionError(f"job stuck: {j}")


def test_download_runs_with_real_progress_and_lands_in_models_dir(env):
    client, hub, reg, tmp = env
    r = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"})
    assert r.status_code == 202
    job = r.json()
    assert job["id"].startswith("dl_") and job["repo"] == "org/tiny"
    j = wait_state(client, job["id"], ("done", "failed"))
    assert j["state"] == "done", j
    assert j["bytes_total"] == 1000 + len(CONFIG) == j["bytes_done"]
    assert j["files_total"] == 2 and j["files_done"] == 2
    assert j["path"] == str(tmp / "models" / "org" / "tiny")
    assert (tmp / "models/org/tiny/model.safetensors").stat().st_size == 1000
    detail = client.get(f"/v1/yunshu/downloads/{job['id']}").json()
    assert {f["name"] for f in detail["files"]} == {"config.json", "model.safetensors"}
    listing = client.get("/v1/yunshu/downloads").json()
    assert [d["id"] for d in listing["downloads"]] == [job["id"]]
    assert listing["active"] == 0 and listing["free_bytes"] > 0
    # the list view stays small: no per-file array
    assert "files" not in listing["downloads"][0]


def test_progress_is_visible_midway_with_rate_and_eta(env):
    client, hub, *_ = env
    hub.chunk = 50
    hub.gate.clear()
    job = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"}).json()
    hub.gate.set()
    assert hub.first_chunk.wait(5)
    hub.gate.clear()
    time.sleep(0.05)
    mid = client.get(f"/v1/yunshu/downloads/{job['id']}").json()
    assert mid["state"] == "running"
    assert 0 < mid["bytes_done"] < mid["bytes_total"]
    assert mid["active_files"]
    hub.gate.set()
    wait_state(client, job["id"], ("done",))


def test_cancel_stops_the_transfer_and_resume_continues_it(env):
    client, hub, reg, tmp = env
    hub.chunk = 100
    hub.gate.set()
    job = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"}).json()
    assert hub.first_chunk.wait(5)
    hub.gate.clear()
    r = client.delete(f"/v1/yunshu/downloads/{job['id']}")
    assert r.status_code == 200
    hub.gate.set()
    j = wait_state(client, job["id"], ("cancelled", "done"))
    assert j["state"] == "cancelled"
    part = (
        (
            tmp
            / "models/org/tiny/.cache/huggingface/download/model.safetensors.incomplete"
        )
        .stat()
        .st_size
    )
    assert 0 < part < 1000
    written_before = hub.written
    # the same request again resumes: only the missing bytes are fetched
    again = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"}).json()
    assert again["id"] != job["id"]
    j2 = wait_state(client, again["id"], ("done", "failed"))
    assert j2["state"] == "done", j2
    assert (tmp / "models/org/tiny/model.safetensors").stat().st_size == 1000
    assert hub.written - written_before <= 1000 - part + len(CONFIG)


def test_same_active_request_returns_the_same_job(env):
    client, hub, *_ = env
    hub.gate.clear()
    a = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"}).json()
    b = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"}).json()
    assert a["id"] == b["id"]
    hub.gate.set()
    wait_state(client, a["id"], ("done",))


def test_refuses_when_the_disk_cannot_hold_the_model(env, monkeypatch):
    client, *_ = env
    monkeypatch.setattr(
        dl.shutil, "disk_usage", lambda p: SimpleNamespace(free=10_000, total=1, used=1)
    )
    monkeypatch.setattr(dl, "DISK_MARGIN_MIN", 1_000_000)
    r = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"})
    assert r.status_code == 507
    d = r.json()["detail"]
    assert "not enough disk space" in d["message"]
    assert d["needed_bytes"] > d["free_bytes"] == 10_000
    assert client.get("/v1/yunshu/downloads").json()["downloads"] == []


def test_validation_unknown_repo_and_missing_job(env):
    client, *_ = env
    assert (
        client.post("/v1/yunshu/downloads", json={"repo": "no-slash"}).status_code
        == 400
    )
    assert (
        client.post("/v1/yunshu/downloads", json={"repo": "../x/y"}).status_code == 400
    )
    assert (
        client.post(
            "/v1/yunshu/downloads", json={"repo": "org/ok", "revision": "a b"}
        ).status_code
        == 400
    )
    r = client.post("/v1/yunshu/downloads", json={"repo": "org/missing"})
    assert r.status_code == 502 and "cannot read org/missing" in r.json()["detail"]
    assert client.get("/v1/yunshu/downloads/dl_nope").status_code == 404
    assert client.delete("/v1/yunshu/downloads/dl_nope").status_code == 404


def test_allow_patterns_limit_the_files(env):
    client, hub, _, tmp = env
    job = client.post(
        "/v1/yunshu/downloads", json={"repo": "org/tiny", "allow_patterns": ["*.json"]}
    ).json()
    j = wait_state(client, job["id"], ("done", "failed"))
    assert j["bytes_total"] == len(CONFIG)  # the finished-model check does not apply
    assert not (tmp / "models/org/tiny/model.safetensors").exists()


def test_incomplete_result_is_a_failed_job(env):
    client, hub, *_ = env
    hub.files = {"config.json": CONFIG}  # no weights
    job = client.post("/v1/yunshu/downloads", json={"repo": "org/noweights"}).json()
    j = wait_state(client, job["id"], ("done", "failed"))
    assert j["state"] == "failed" and "incomplete" in j["error"]


def test_already_present_model_is_not_downloaded(env):
    client, hub, _, tmp = env
    d = tmp / "models/org/have"
    d.mkdir(parents=True)
    (d / "config.json").write_bytes(CONFIG)
    (d / "w.safetensors").write_bytes(b"z" * 10)
    j = client.post("/v1/yunshu/downloads", json={"repo": "org/have"}).json()
    assert j["state"] == "done" and j["already_present"] is True
    assert hub.listed == 0


def test_finished_model_is_registered_with_the_manager(env, monkeypatch):
    client, *_ = env
    registered = {}
    manager = SimpleNamespace(
        list_entries=lambda: [],
        register_model=lambda mid, path, **kw: registered.update({mid: path}),
    )
    monkeypatch.setattr(am, "get_model_manager", lambda: manager)
    job = client.post("/v1/yunshu/downloads", json={"repo": "org/tiny"}).json()
    j = wait_state(client, job["id"], ("done", "failed"))
    assert j["state"] == "done" and j["registered"] is True
    assert registered["org/tiny"].endswith("models/org/tiny")


def test_mutations_need_admin_reads_do_not(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path)
    dl.set_registry(dl.DownloadRegistry(FakeHub()))
    app = FastAPI()
    app.include_router(am.router, prefix="/v1")
    c = TestClient(app)
    assert c.post("/v1/yunshu/downloads", json={"repo": "org/x"}).status_code == 401
    assert c.delete("/v1/yunshu/downloads/dl_x").status_code == 401
    assert c.get("/v1/yunshu/downloads").status_code == 200
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    assert c.get("/v1/yunshu/downloads").status_code == 401
    ok = c.post(
        "/v1/yunshu/downloads",
        json={"repo": "org/x"},
        headers={"Authorization": "Bearer tok"},
    )
    assert ok.status_code == 202
    dl.set_registry(None)


# ── Ollama /api/pull on the same registry ──────────────────────────────


@pytest.fixture
def ollama_env(env, monkeypatch):
    client, hub, reg, tmp = env
    from yunshu_gateway import ollama_models as om

    entries: dict = {}
    manager = SimpleNamespace(
        list_entries=lambda: list(entries.values()),
        register_model=lambda name, path, **kw: entries.update(
            {name: SimpleNamespace(model_id=name, model_path=path)}
        ),
    )
    monkeypatch.setattr(om, "get_model_manager", lambda: manager)
    app = FastAPI()
    app.include_router(ollama.router)
    return TestClient(app), hub, entries, om


def test_ollama_pull_streams_real_progress_then_success(ollama_env):
    client, hub, entries, om = ollama_env
    hub.chunk = 100
    r = client.post("/api/pull", json={"model": "org/native"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert lines[0] == {"status": "pulling manifest"}
    assert lines[-1] == {"status": "success"}
    prog = [x for x in lines if "completed" in x]
    assert prog and prog[-1]["completed"] == prog[-1]["total"] == 1000 + len(CONFIG)
    assert all(x["digest"].startswith("sha256:") for x in prog)
    assert [p["completed"] for p in prog] == sorted(p["completed"] for p in prog)
    assert "org/native" in entries
    assert om.model_link("org/native").is_symlink()


def test_ollama_pull_without_stream_blocks_and_answers_once(ollama_env):
    client, hub, entries, _ = ollama_env
    r = client.post("/api/pull", json={"model": "org/native2", "stream": False})
    assert r.status_code == 200 and r.json() == {"status": "success"}
    assert "org/native2" in entries


def test_ollama_pull_failure_is_an_error_line(ollama_env):
    client, hub, entries, _ = ollama_env
    r = client.post("/api/pull", json={"model": "org/missing"})
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert "error" in lines[-1] and "org/missing" in lines[-1]["error"]
    assert "org/missing" not in entries


def test_ollama_pull_out_of_disk_is_reported(ollama_env, monkeypatch):
    client, *_ = ollama_env
    monkeypatch.setattr(
        dl.shutil, "disk_usage", lambda p: SimpleNamespace(free=10, total=1, used=1)
    )
    lines = [
        json.loads(x)
        for x in client.post("/api/pull", json={"model": "org/big"}).text.splitlines()
    ]
    assert "not enough disk space" in lines[-1]["error"]


# ── local inventory ────────────────────────────────────────────────────


def _model(root: Path, rel: str, config: dict, weights=b"w" * 64):
    d = root / rel
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    (d / "model.safetensors").write_bytes(weights)
    return d


def test_local_inventory_lists_unregistered_models_with_card_facts(env, monkeypatch):
    client, _, _, tmp = env
    models = tmp / "models"
    d = _model(
        models,
        "org/q4",
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 4096,
            "quantization": {"bits": 4, "group_size": 64},
        },
    )
    partial = _model(models, "org/half", {"model_type": "qwen3"})
    (partial / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "gone.safetensors"}})
    )
    monkeypatch.setattr("yunshu_cli.model.scan_hf_cache", lambda: [])
    monkeypatch.setattr(am, "_inv_cache", None)
    monkeypatch.setattr(
        am,
        "get_model_manager",
        lambda: SimpleNamespace(
            list_entries=lambda: [
                SimpleNamespace(model_id="q4-alias", model_path=str(d), is_loaded=True)
            ]
        ),
    )
    j = client.get("/v1/yunshu/models/local").json()
    by = {m["id"]: m for m in j["models"]}
    q4 = by["org/q4"]
    assert q4["size_bytes"] > 64 and q4["source"] == "models-dir"
    assert q4["architecture"] == "Qwen3ForCausalLM"
    assert q4["quantization"] == {"bits": 4, "group_size": 64}
    assert q4["context_length"] == 4096
    assert "chat" in q4["capabilities"]
    assert q4["complete"] is True
    assert q4["loaded"] is True and q4["registered_as"] == "q4-alias"
    half = by["org/half"]
    assert half["complete"] is False and "missing" in half["complete_reason"]
    assert half["loaded"] is False and half["registered_as"] is None
    assert j["total_bytes"] == sum(m["size_bytes"] for m in j["models"])


def test_local_inventory_scan_is_cached_between_polls(env, monkeypatch):
    client, _, _, tmp = env
    _model(tmp / "models", "org/a", {"model_type": "qwen3"})
    monkeypatch.setattr("yunshu_cli.model.scan_hf_cache", lambda: [])
    monkeypatch.setattr(am, "_inv_cache", None)
    calls = []
    real = am._scan_disk

    import yunshu_cli.model as cm

    orig = cm.scan_models_dir
    monkeypatch.setattr(cm, "scan_models_dir", lambda b: calls.append(1) or orig(b))
    client.get("/v1/yunshu/models/local")
    client.get("/v1/yunshu/models/local")
    assert len(calls) == 1
    client.get("/v1/yunshu/models/local?refresh=true")
    assert len(calls) == 2 and real


# ── fit-before-load ────────────────────────────────────────────────────


GB = 10**9


@pytest.fixture
def fit_client(monkeypatch):
    holder = {}
    monkeypatch.setattr(am, "get_model_manager", lambda: holder.get("m"))
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(am.router, prefix="/v1")
    return TestClient(app), holder


def _fit_manager(budget, entries):
    from yunshu_engine.model_manager import ModelManager, ModelType

    m = ModelManager(max_memory_bytes=budget, kv_reserve_ratio=0.25)
    for mid, size, loaded, last in entries:
        m.register_model(
            mid, "/x/" + mid, estimated_bytes=size, model_type=ModelType.LLM
        )
        e = m._entries[mid]
        e.is_loaded, e.last_access = loaded, last
        e.engine = SimpleNamespace(has_active_requests=lambda: False)
        if loaded:
            m._current_memory_bytes += size
    return m


def test_fit_fits_with_headroom(fit_client):
    client, h = fit_client
    h["m"] = _fit_manager(100 * GB, [("org/a", 10 * GB, False, 0)])
    j = client.get("/v1/yunshu/models/org/a/fit").json()
    assert j["verdict"] == "fits" and j["needed_bytes"] == int(12.5 * GB)
    assert j["weights_bytes"] == 10 * GB and j["kv_reserve_bytes"] == int(2.5 * GB)
    assert j["free_bytes"] == 100 * GB and j["would_evict"] == []
    assert j["basis"]["estimated"] is True


def test_fit_tight_when_it_only_fits_after_evicting_the_lru_model(fit_client):
    client, h = fit_client
    h["m"] = _fit_manager(
        34 * GB,
        [
            ("old", 12 * GB, True, 1),
            ("new", 12 * GB, True, 2),
            ("org/big", 16 * GB, False, 0),  # needs 20 GB, 10 GB free
        ],
    )
    j = client.get("/v1/yunshu/models/org/big/fit").json()
    assert j["verdict"] == "tight" and j["would_evict"] == ["old"]
    # the dry run mutated nothing
    assert h["m"]._entries["old"].is_loaded and h["m"]._current_memory_bytes == 24 * GB


def test_fit_wont_fit_when_the_budget_is_too_small(fit_client):
    client, h = fit_client
    h["m"] = _fit_manager(10 * GB, [("org/huge", 40 * GB, False, 0)])
    j = client.get("/v1/yunshu/models/org/huge/fit").json()
    assert j["verdict"] == "wont_fit" and "needs 50.0 GB" in j["reason"]


def test_fit_agrees_with_the_real_load_path(fit_client):
    """Verdict wont_fit <=> _ensure_memory_available raises; fits/tight <=> it succeeds."""
    import asyncio

    client, h = fit_client
    for budget, expect_ok in ((100 * GB, True), (10 * GB, False)):
        m = _fit_manager(budget, [("org/m", 20 * GB, False, 0)])
        h["m"] = m
        verdict = client.get("/v1/yunshu/models/org/m/fit").json()["verdict"]

        async def load(m=m):
            await m._ensure_memory_available(int(20 * GB * 1.25))

        if expect_ok:
            asyncio.run(load())
            assert verdict in ("fits", "tight")
        else:
            with pytest.raises(MemoryError):
                asyncio.run(load())
            assert verdict == "wont_fit"


def test_fit_loaded_unknown_and_single_engine_mode(fit_client):
    client, h = fit_client
    assert client.get("/v1/yunshu/models/org/a/fit").status_code == 400
    h["m"] = _fit_manager(10 * GB, [("org/a", 4 * GB, True, 0)])
    assert client.get("/v1/yunshu/models/org/a/fit").json()["loaded"] is True
    assert client.get("/v1/yunshu/models/nope/fit").status_code == 404


# ── reload ─────────────────────────────────────────────────────────────


class FakeManager:
    def __init__(self, loaded=True, active=False, leases=0, load_error=None):
        self.entry = SimpleNamespace(
            is_loaded=loaded,
            leases=leases,
            engine=SimpleNamespace(has_active_requests=lambda: active)
            if loaded
            else None,
        )
        self.calls: list = []
        self.load_error = load_error

    def resolve_model_id(self, mid):
        return mid if mid == "org/a" else None

    def get_entry(self, mid):
        return self.entry if mid == "org/a" else None

    async def unload_model(self, mid, force=False):
        self.calls.append(("unload", mid, force))
        self.entry.is_loaded = False
        return True

    async def get_engine(self, mid):
        self.calls.append(("load", mid))
        if self.load_error:
            raise self.load_error
        self.entry.is_loaded = True
        return SimpleNamespace(is_running=True)


@pytest.fixture
def reload_client(monkeypatch):
    holder = {}
    monkeypatch.setattr(am, "get_model_manager", lambda: holder.get("m"))
    monkeypatch.setattr(am, "_check_permission", lambda *a: None)
    app = FastAPI()
    app.include_router(am.router, prefix="/v1")
    return TestClient(app), holder


def test_reload_unloads_then_loads_the_same_model(reload_client):
    client, h = reload_client
    h["m"] = FakeManager()
    r = client.post("/v1/yunshu/models/org/a/reload")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "reloaded" and j["model"] == "org/a" and j["was_loaded"]
    assert h["m"].calls == [("unload", "org/a", False), ("load", "org/a")]


def test_reload_of_an_unloaded_model_just_loads(reload_client):
    client, h = reload_client
    h["m"] = FakeManager(loaded=False)
    r = client.post("/v1/yunshu/models/org/a/reload")
    assert r.status_code == 200 and r.json()["was_loaded"] is False
    assert h["m"].calls == [("load", "org/a")]


def test_reload_refuses_while_requests_run_unless_forced(reload_client):
    client, h = reload_client
    h["m"] = FakeManager(active=True)
    r = client.post("/v1/yunshu/models/org/a/reload")
    assert r.status_code == 409 and h["m"].calls == []
    r = client.post("/v1/yunshu/models/org/a/reload", json={"force": True})
    assert r.status_code == 200 and r.json()["forced"] is True
    assert h["m"].calls[0] == ("unload", "org/a", True)


def test_reload_refuses_a_leased_model(reload_client):
    client, h = reload_client
    h["m"] = FakeManager(leases=1)
    assert client.post("/v1/yunshu/models/org/a/reload").status_code == 409
    assert h["m"].calls == []


def test_reload_unknown_model_and_single_engine_mode(reload_client):
    client, h = reload_client
    assert client.post("/v1/yunshu/models/org/a/reload").status_code == 400
    h["m"] = FakeManager()
    assert client.post("/v1/yunshu/models/org/zzz/reload").status_code == 404


def test_reload_reports_a_failed_load_and_frees_the_guard(reload_client):
    from yunshu_gateway.routers import models as models_router

    client, h = reload_client
    h["m"] = FakeManager(load_error=RuntimeError("boom"))
    r = client.post("/v1/yunshu/models/org/a/reload")
    assert r.status_code == 500 and "boom" in r.json()["detail"]
    assert "org/a" not in models_router._model_ops_inflight


def test_reload_needs_admin(monkeypatch):
    app = FastAPI()
    app.include_router(am.router, prefix="/v1")
    monkeypatch.setattr(am, "get_model_manager", lambda: FakeManager())
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    c = TestClient(app)
    assert c.post("/v1/yunshu/models/org/a/reload").status_code == 401
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    assert c.post("/v1/yunshu/models/org/a/reload").status_code == 401
    ok = c.post(
        "/v1/yunshu/models/org/a/reload", headers={"Authorization": "Bearer tok"}
    )
    assert ok.status_code == 200


def test_local_inventory_lists_a_model_served_from_an_explicit_path(env, monkeypatch):
    """`serve -m /some/path` loads a model that is in neither the models dir nor the HF cache;
    the console must still list it, marked loaded."""
    client, _, _, tmp = env
    ext = _model(
        tmp / "elsewhere", "my-model", {"model_type": "qwen3"}, weights=b"w" * 128
    )
    monkeypatch.setattr("yunshu_cli.model.scan_hf_cache", lambda: [])
    monkeypatch.setattr(am, "_inv_cache", None)
    monkeypatch.setattr(
        am,
        "get_model_manager",
        lambda: SimpleNamespace(
            list_entries=lambda: [
                SimpleNamespace(
                    model_id="my-model", model_path=str(ext), is_loaded=True
                )
            ]
        ),
    )
    j = client.get("/v1/yunshu/models/local").json()
    row = next(m for m in j["models"] if m["id"] == "my-model")
    assert row["loaded"] is True and row["source"] == "path"
    assert row["size_bytes"] >= 128
