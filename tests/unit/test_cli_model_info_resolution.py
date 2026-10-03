"""Model info must identify a listed checkpoint without silently picking another."""

import json
import os
import subprocess
import sys
from pathlib import Path

from yunshu_cli import model as cli_model


def checkpoint(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"test"}')
    return path


def setup_inventory(monkeypatch, base, cached=()):
    monkeypatch.setattr(cli_model, "_get_models_dir", lambda: base)
    monkeypatch.setattr(cli_model, "scan_hf_cache", lambda: list(cached))
    monkeypatch.setattr(cli_model, "_detect_model_type", lambda _: "LLM")


def test_info_accepts_exact_hf_repo_id(monkeypatch, tmp_path):
    cached = checkpoint(tmp_path / "snapshot")
    setup_inventory(
        monkeypatch,
        tmp_path / "models",
        [{"name": "org/checkpoint", "path": str(cached)}],
    )
    emitted = []
    monkeypatch.setattr(cli_model, "is_json", lambda: True)
    monkeypatch.setattr(cli_model, "emit", emitted.append)
    cli_model.model_info("org/checkpoint")
    assert emitted[0]["path"] == str(cached.resolve())


def test_partial_name_searches_nested_org_layout(monkeypatch, tmp_path):
    path = checkpoint(tmp_path / "models" / "org" / "Qwen-test")
    setup_inventory(monkeypatch, tmp_path / "models")
    assert cli_model.resolve_info_model("qwen") == path.resolve()


def test_ambiguous_name_refuses_to_pick_first_model(monkeypatch, tmp_path):
    for name in ["Qwen-small", "Qwen-large"]:
        checkpoint(tmp_path / "models" / "org" / name)
    setup_inventory(monkeypatch, tmp_path / "models")
    result = invoke_info(tmp_path, "Qwen")
    assert result.returncode == 2, result.stderr
    error = json.loads(result.stdout)["error"]
    assert (
        "ambiguous" in error and "org/Qwen-large" in error and "org/Qwen-small" in error
    )


def test_exact_local_reference_wins_over_partial_matches(monkeypatch, tmp_path):
    exact = checkpoint(tmp_path / "models" / "org" / "Qwen")
    checkpoint(tmp_path / "models" / "org" / "Qwen-large")
    setup_inventory(monkeypatch, tmp_path / "models")
    assert cli_model.resolve_info_model("org/Qwen") == exact


def test_duplicate_inventory_paths_are_one_checkpoint(monkeypatch, tmp_path):
    exact = checkpoint(tmp_path / "models" / "org" / "Qwen")
    setup_inventory(
        monkeypatch, tmp_path / "models", [{"name": "alias/Qwen", "path": str(exact)}]
    )
    assert cli_model.resolve_info_model("qwen") == exact.resolve()


def test_missing_model_has_actionable_json_error(monkeypatch, tmp_path):
    setup_inventory(monkeypatch, tmp_path / "models")
    result = invoke_info(tmp_path, "absent")
    assert result.returncode == 1, result.stderr
    assert "yunshu model list" in json.loads(result.stdout)["error"]


def invoke_info(tmp_path, name):
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("YUNSHU_")
    }
    env.update(
        HOME=str(tmp_path),
        HF_HOME=str(tmp_path / "hf"),
        YUNSHU_MODELS_DIR=str(tmp_path / "models"),
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "python"),
    )
    return subprocess.run(
        [sys.executable, "-m", "yunshu_cli", "--json", "model", "info", name],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
