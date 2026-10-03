import json
from pathlib import Path

from scripts.research import spec_bench_snapshot as snapshot


def test_benchmark_snapshot_freezes_code_and_ignores_bytecode(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = root / "python"
    source.mkdir(parents=True)
    (source / "lane.py").write_text("version = 1\n")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "lane.pyc").write_bytes(b"unstable bytecode")
    monkeypatch.setattr(
        snapshot, "__file__", str(root / "scripts/research/snapshot.py")
    )
    monkeypatch.setattr(snapshot.subprocess, "check_output", lambda *a, **k: "abc123\n")
    frozen, record = snapshot.freeze(tmp_path / "arm.jsonl")
    assert record["head"] == "abc123"
    assert not (frozen / "__pycache__").exists()
    (source / "lane.py").write_text("version = 2\n")
    assert (frozen / "lane.py").read_text() == "version = 1\n"
    _, changed = snapshot.freeze(tmp_path / "changed.jsonl")
    assert changed["python_sha256"] != record["python_sha256"]
    assert Path(record["source"]) == frozen


def test_contended_admission_cannot_produce_a_successful_measurement(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(snapshot, "was_contended", lambda: True)
    output = tmp_path / "blocked.jsonl"
    assert snapshot.refuse_contended(output)
    result = json.loads(output.read_text())
    assert result["complete"] and result["contended"] and not result["success"]
