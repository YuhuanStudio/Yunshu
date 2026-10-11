"""Ollama native-model lifecycle, with no network or model execution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import ollama_models as om
from yunshu_gateway.routers import ollama


@pytest.fixture
def managed(monkeypatch, tmp_path):
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"fixture")
    entries = {
        "base": SimpleNamespace(
            model_id="base",
            model_path=str(source),
            estimated_bytes=7,
            model_type="llm",
            is_loaded=False,
        )
    }

    def register(name, path, **kw):
        entries[name] = SimpleNamespace(
            model_id=name,
            model_path=path,
            estimated_bytes=kw.get("estimated_bytes", 7),
            model_type=kw.get("model_type", "llm"),
            is_loaded=False,
        )

    manager = SimpleNamespace(
        get_entry=entries.get,
        register_model=register,
        list_entries=lambda: list(entries.values()),
        unregister_model=lambda name: entries.pop(name, None),
        unload_model=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(om, "get_model_manager", lambda: manager)
    monkeypatch.setattr(om.paths, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(ollama.router)
    return TestClient(app), entries, manager


def test_copy_create_pull_delete_shapes(managed):
    client, entries, _ = managed
    r = client.post(
        "/api/copy", json={"source": "base", "destination": "user/copy:latest"}
    )
    assert r.status_code == 200 and r.content == b""
    assert om.model_link("user/copy:latest").is_symlink()
    assert client.post(
        "/api/pull", json={"model": "user/copy:latest", "stream": False}
    ).json() == {"status": "success"}
    r = client.post("/api/create", json={"model": "second", "from": "base"})
    assert r.status_code == 200 and r.json() == {"status": "success"}
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert (
        client.request(
            "DELETE", "/api/delete", json={"model": "user/copy:latest"}
        ).status_code
        == 200
    )
    assert "user/copy:latest" not in entries
    assert (om.paths.models_dir() / "base/model.safetensors").exists()


def test_delete_checkpoint_refuses_dangling_copy(managed):
    client, _, _ = managed
    client.post("/api/copy", json={"source": "base", "destination": "copy"})
    assert (
        client.request("DELETE", "/api/delete", json={"model": "base"}).status_code
        == 409
    )
    client.request("DELETE", "/api/delete", json={"model": "copy"})
    assert (
        client.request("DELETE", "/api/delete", json={"model": "base"}).status_code
        == 200
    )
    assert not (om.paths.models_dir() / "base").exists()


def test_busy_copy_cannot_be_deleted(managed):
    client, entries, manager = managed
    client.post("/api/copy", json={"source": "base", "destination": "copy"})
    entries["copy"].is_loaded = True
    r = client.request("DELETE", "/api/delete", json={"model": "copy"})
    assert r.status_code == 409
    assert "copy" in entries and om.model_link("copy").exists()
    manager.unload_model.assert_awaited_once_with("copy")


def test_pull_native_repository(managed, monkeypatch):
    client, entries, _ = managed
    from yunshu_gateway import downloads

    class Hub:
        def list_files(self, repo, revision, patterns):
            return [("model.safetensors", 7)]

        def cache_dir(self):
            return om.paths.models_dir()

        def download(self, repo, revision, patterns, local_dir, on_file, on_bytes):
            on_file("model.safetensors", 7, 0)
            on_bytes("model.safetensors", 7)
            return om.paths.models_dir() / "base"

    monkeypatch.setattr(downloads, "_registry", downloads.DownloadRegistry(Hub()))
    r = client.post("/api/pull", json={"model": "org/native", "stream": False})
    assert r.status_code == 200 and r.json() == {"status": "success"}
    assert entries["org/native"].model_path == str(om.model_link("org/native"))


@pytest.mark.parametrize(
    "body",
    [
        {"model": "new", "from": "base", "system": "ignored"},
        {"model": "new", "files": {"x": "sha256:x"}},
    ],
)
def test_create_unsupported_not_silently_ignored(managed, body):
    client, entries, _ = managed
    assert client.post("/api/create", json=body).status_code == 400
    assert "new" not in entries


def test_errors_are_ollama_shape(managed):
    client, _, _ = managed
    for path, body in [
        ("/api/pull", {"model": "gguf:latest"}),
        ("/api/copy", {"source": "base", "destination": "../escape"}),
        ("/api/copy", {"source": "missing", "destination": "new"}),
    ]:
        r = client.post(path, json=body)
        assert r.status_code in (400, 404) and isinstance(r.json()["error"], str)


def test_persistent_namespace_names_are_discovered(managed):
    from yunshu_engine.model_manager import ModelManager

    client, entries, _ = managed
    client.post("/api/copy", json={"source": "base", "destination": "user/copy:latest"})
    found = {}
    manager = object.__new__(ModelManager)
    manager._entries = {}

    def register(model_id, **kw):
        found[model_id] = kw

    manager.register_model = register
    manager.discover_models(str(om.paths.models_dir()))
    assert "user/copy:latest" in found
    assert (
        found["user/copy:latest"]["model_path"]
        == entries["user/copy:latest"].model_path
    )


def test_alias_discovery_preserves_reranker_role(managed):
    from yunshu_engine.model_manager import ModelManager, ModelType

    client, entries, manager = managed
    root = om.paths.models_dir()
    source = root / "Qwen3-VL-Reranker-2B"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_vl","vision_config":{}}')
    entries["retrieval"] = SimpleNamespace(
        model_id="retrieval",
        model_path=str(source),
        estimated_bytes=0,
        model_type=ModelType.RERANKER,
        is_loaded=False,
    )
    assert (
        client.post(
            "/api/copy", json={"source": "retrieval", "destination": "generic-copy"}
        ).status_code
        == 200
    )
    scanner = object.__new__(ModelManager)
    scanner._entries = {}
    found = {}
    scanner.register_model = lambda model_id, **kw: found.update({model_id: kw})
    scanner.discover_models(str(root))
    assert found["generic-copy"]["model_type"] is ModelType.RERANKER


def test_wildcard_inference_alias_does_not_delete_or_block_copy(managed):
    client, entries, manager = managed
    manager.get_entry = lambda name: entries.get(name) or entries["base"]
    r = client.request(
        "DELETE", "/api/delete", json={"model": "not-a-registered-model"}
    )
    assert (
        r.status_code == 404
        and (om.paths.models_dir() / "base/model.safetensors").exists()
    )
    assert (
        client.post(
            "/api/copy", json={"source": "base", "destination": "new-copy"}
        ).status_code
        == 200
    )


@pytest.mark.asyncio
async def test_registered_role_controls_engine_under_generic_alias(monkeypatch):
    from yunshu_engine import model_manager as mm

    class Retrieval:
        is_reranker = False

        def __init__(self, path, config):
            pass

        async def start(self):
            self.started_as_reranker = self.is_reranker

    monkeypatch.setattr(mm, "_embedding_engine_class", lambda path: Retrieval)
    engine = await mm.instantiate_engine(mm.ModelType.RERANKER, "generic-copy")
    assert engine.is_reranker and engine.started_as_reranker
