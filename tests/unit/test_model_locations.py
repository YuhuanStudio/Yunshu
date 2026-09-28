"""Where models come from: the models directory, the Hugging Face cache, and a
persistent models-directory setting (`yunshu config set`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yunshu_engine import model_discovery as md
from yunshu_engine import settings


def _model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (path / "model.safetensors").write_bytes(b"\0" * 16)
    return path


@pytest.fixture
def no_hf(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)


def test_resolve_existing_path(tmp_path, no_hf):
    m = _model(tmp_path / "m")
    assert md.resolve_model_ref(str(m), tmp_path / "none") == str(m)


def test_resolve_pulled_layout_org_name(tmp_path, no_hf):
    base = tmp_path / "models"
    m = _model(base / "mlx-community" / "Qwen3.5-9B-MLX-4bit")
    assert md.resolve_model_ref("mlx-community/Qwen3.5-9B-MLX-4bit", base) == str(m)


def test_resolve_flat_layout_by_repo_name(tmp_path, no_hf):
    base = tmp_path / "models"
    m = _model(base / "Qwen3.5-9B-MLX-4bit")
    assert md.resolve_model_ref("mlx-community/Qwen3.5-9B-MLX-4bit", base) == str(m)


def test_resolve_hf_cache_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    snap = _model(tmp_path / "hub" / "snapshots" / "abc123")
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, **k: str(snap / name) if repo == "org/cached" else None,
    )
    assert md.resolve_model_ref("org/cached", tmp_path / "models") == str(snap)
    # Not cached: the id is returned unchanged (the loader downloads it).
    assert md.resolve_model_ref("org/other", tmp_path / "models") == "org/other"


def test_resolve_missing_path_unchanged(tmp_path, no_hf):
    assert md.resolve_model_ref("/nope/model", tmp_path) == "/nope/model"
    assert md.resolve_model_ref(None) is None


def test_hf_cache_models_join_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "detect_model_type", lambda p: "llm")
    base = tmp_path / "models"
    local = _model(base / "Qwen3.5-9B")
    cached_same_name = _model(tmp_path / "hub" / "a" / "snap")
    cached_new = _model(tmp_path / "hub" / "b" / "snap")
    monkeypatch.setattr(
        md,
        "hf_cache_snapshots",
        lambda: [
            ("mlx-community/Qwen3.5-9B", cached_same_name, 1),
            ("org/Gemma-4", cached_new, 1),
            ("org/local-dup", local, 1),  # same folder as a models-dir entry
        ],
    )
    models = md.discover_hf_cache_models(md.discover_models(base))
    assert models["Qwen3.5-9B"].model_path == str(local)  # models dir wins
    assert models["mlx-community/Qwen3.5-9B"].model_path == str(cached_same_name)
    assert models["Gemma-4"].model_path == str(cached_new)
    assert "local-dup" not in models


def test_gateway_discovery_includes_hf_cache(tmp_path, monkeypatch):
    from yunshu_gateway import engine as gw

    snap = _model(tmp_path / "hub" / "snap")
    monkeypatch.setattr(md, "detect_model_type", lambda p: "llm")
    monkeypatch.setattr(md, "hf_cache_snapshots", lambda: [("org/Cached", snap, 1)])
    manager = gw.init_model_manager(models_dir=str(tmp_path / "missing-dir"))
    assert manager.resolve_model_id("org/Cached") == "Cached"

    monkeypatch.setenv("YUNSHU_HF_CACHE_MODELS", "0")
    manager = gw.init_model_manager(models_dir=str(tmp_path / "missing-dir"))
    assert manager.resolve_model_id("Cached") is None


def test_user_config_file_sets_models_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_MODELS_DIR", raising=False)
    monkeypatch.delenv("YUNSHU_CONFIG", raising=False)
    target = settings.write_config_value("YUNSHU_MODELS_DIR", str(tmp_path / "m"))
    assert target == settings.user_config_path()
    assert settings.raw("YUNSHU_MODELS_DIR") == (str(tmp_path / "m"), "file")

    from yunshu_engine import paths

    assert paths.models_dir() == tmp_path / "m"
    settings.write_config_value("YUNSHU_MODELS_DIR", None)
    assert settings.raw("YUNSHU_MODELS_DIR")[1] == "default"


def test_write_config_value_validates(tmp_path):
    with pytest.raises(settings.SettingError):
        settings.write_config_value("YUNSHU_KV_PRECISION", "fp4")
    with pytest.raises(KeyError):
        settings.write_config_value("YUNSHU_NOT_A_SETTING", "1")


def test_cli_config_set_unset(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from yunshu_cli import app

    monkeypatch.delenv("YUNSHU_MODELS_DIR", raising=False)
    monkeypatch.delenv("YUNSHU_CONFIG", raising=False)
    runner = CliRunner()
    r = runner.invoke(app, ["config", "set", "models_dir", str(tmp_path / "x")])
    assert r.exit_code == 0, r.output
    assert "YUNSHU_MODELS_DIR" in settings.user_config_path().read_text()
    assert settings.get("YUNSHU_MODELS_DIR") == str(tmp_path / "x")

    r = runner.invoke(app, ["config", "set", "kv_precision", "fp4"])
    assert r.exit_code == 2
    r = runner.invoke(app, ["config", "set", "no_such_thing", "1"])
    assert r.exit_code == 2

    r = runner.invoke(app, ["config", "unset", "models_dir"])
    assert r.exit_code == 0, r.output
    assert settings.raw("YUNSHU_MODELS_DIR")[1] == "default"

    r = runner.invoke(app, ["config", "path"])
    assert r.output.strip() == str(settings.user_config_path())


def test_hf_repo_id_for_snapshot_paths(tmp_path):
    snap = (
        tmp_path
        / "hub"
        / "models--mlx-community--Qwen3.5-0.8B-4bit"
        / "snapshots"
        / "abc123"
    )
    assert md.hf_repo_id_for(snap) == "mlx-community/Qwen3.5-0.8B-4bit"
    assert md.hf_repo_id_for(str(snap)) == "mlx-community/Qwen3.5-0.8B-4bit"
    assert md.hf_repo_id_for(tmp_path / "models" / "Qwen3.5-0.8B") is None
    assert md.hf_repo_id_for(None) is None


def test_single_model_lists_hf_repo_id(mock_engine, monkeypatch):
    """A model served from a Hugging Face cache snapshot is listed under its
    repo id, not the revision hash; any requested name still reaches it."""
    from fastapi.testclient import TestClient

    from yunshu_gateway import engine as gw
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    monkeypatch.setattr(gw, "_model_manager", None)  # single-model mode
    mock_engine.is_loaded = True
    mock_engine.model_name = "abc123"
    set_engine(mock_engine, display_id="mlx-community/Qwen3.5-0.8B-4bit")
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        assert ids == ["mlx-community/Qwen3.5-0.8B-4bit"]
    finally:
        set_engine(None)
    set_engine(mock_engine)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        assert ids == ["abc123"]
    finally:
        set_engine(None)
