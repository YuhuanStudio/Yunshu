"""The first-run CLI: --version, pull, model list, doctor, serve pre-flight and
the launchd service. Nothing here downloads, starts a server or touches
launchd: downloads and launchctl are replaced, and HOME points at a temp dir."""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from yunshu_cli import app, model, service
from yunshu_engine import paths, settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_overrides():
    # `serve` records its flags as in-process setting overrides.
    settings.clear_overrides()
    yield
    settings.clear_overrides()


def _model(path: Path, shards: int = 1, index: bool = False) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
    for n in names:
        (path / n).write_bytes(b"\0" * 16)
    if index:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"w{i}": n for i, n in enumerate(names)}})
        )
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YUNSHU_MODELS_DIR", raising=False)
    monkeypatch.delenv("YUNSHU_MODEL", raising=False)
    return tmp_path


def _json(*args: str, home: Path, check: bool = True) -> tuple[int, dict]:
    env = {**os.environ, "HOME": str(home), "PYTHONPATH": os.pathsep.join(sys.path)}
    env.pop("YUNSHU_MODELS_DIR", None)
    env.pop("YUNSHU_MODEL", None)
    r = subprocess.run(
        [sys.executable, "-m", "yunshu_cli", "--json", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode, json.loads(r.stdout)


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert r.output.startswith("yunshu ")


def test_models_dir_default_and_setting(home, monkeypatch):
    assert paths.models_dir() == home / ".yunshu" / "models"
    monkeypatch.setenv("YUNSHU_MODELS_DIR", str(home / "m"))
    assert paths.models_dir() == home / "m"


# ── weights_complete ──────────────────────────────────────────────────────


def test_complete_single_and_sharded(tmp_path):
    assert model.weights_complete(_model(tmp_path / "a")) == (True, "complete")
    b = _model(tmp_path / "b", shards=3, index=True)
    assert model.weights_complete(b)[0]
    (b / "model-00001-of-00003.safetensors").unlink()
    assert model.weights_complete(b) == (False, "1 weight shard(s) missing")


def test_incomplete_download_marker(tmp_path):
    a = _model(tmp_path / "a")
    part = a / ".cache" / "huggingface" / "download"
    part.mkdir(parents=True)
    (part / "x.safetensors.incomplete").write_bytes(b"")
    assert model.weights_complete(a) == (False, "interrupted download")


def test_missing_and_configless(tmp_path):
    assert model.weights_complete(tmp_path / "nope") == (False, "missing")
    (tmp_path / "w").mkdir()
    assert model.weights_complete(tmp_path / "w") == (False, "no config.json")


# ── pull ──────────────────────────────────────────────────────────────────


def _no_download(**kw):
    raise AssertionError(f"must not download: {kw}")


def test_pull_rejects_bad_repo_id(home):
    r = runner.invoke(app, ["pull", "not-a-repo-id"])
    assert r.exit_code == 2


def test_pull_refuses_when_already_downloaded(home, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_download)
    _model(home / ".yunshu" / "models" / "org" / "m")
    r = runner.invoke(app, ["pull", "org/m"])
    assert r.exit_code == 0, r.output
    assert "Already downloaded" in r.output


def test_pull_refuses_flat_legacy_copy(home, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_download)
    _model(home / ".yunshu" / "models" / "m")
    assert "Already downloaded" in runner.invoke(app, ["pull", "org/m"]).output


def test_pull_refuses_when_in_hf_cache(home, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_download)
    snap = _model(home / "snap")
    monkeypatch.setattr(
        model, "scan_hf_cache", lambda: [{"name": "org/m", "path": str(snap)}]
    )
    r = runner.invoke(app, ["pull", "org/m"])
    assert r.exit_code == 0
    assert "Hugging Face cache" in r.output


def test_pull_downloads_into_org_name_and_resumes(home, monkeypatch):
    import huggingface_hub

    calls = []

    def fake(repo_id, local_dir, revision=None):
        calls.append((repo_id, local_dir, revision))
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"\0")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    monkeypatch.setattr(model, "scan_hf_cache", list)
    # A half-finished download (no weights yet) is resumed, not refused.
    partial = home / ".yunshu" / "models" / "org" / "m"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")
    r = runner.invoke(app, ["pull", "org/m", "--revision", "main"])
    assert r.exit_code == 0, r.output
    assert "Resuming" in r.output
    assert calls == [("org/m", str(partial), "main")]


def test_pull_force_redownloads(home, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw)
    )
    _model(home / ".yunshu" / "models" / "org" / "m")
    assert runner.invoke(app, ["pull", "org/m", "--force"]).exit_code == 0
    assert len(calls) == 1


def test_model_download_is_an_alias_of_pull():
    cmds = {c.name: c for c in model.model_app.registered_commands}
    assert cmds["download"].callback is model.pull
    assert cmds["download"].hidden


# ── model list ────────────────────────────────────────────────────────────


def test_model_list_json_reads_org_and_flat_layouts(tmp_path):
    base = tmp_path / "models"
    _model(base / "flat")
    _model(base / "org" / "nested")
    (base / "org" / "not-a-model").mkdir()
    code, out = _json("model", "list", "--dir", str(base), home=tmp_path)
    assert code == 0
    assert sorted(m["name"] for m in out["models"]) == ["flat", "org/nested"]
    assert {m["source"] for m in out["models"]} == {"models-dir"}


# ── doctor ────────────────────────────────────────────────────────────────


@pytest.fixture
def doctor_cli(home, monkeypatch):
    """CLI contracts must not depend on the shared venv or the host's GPU."""
    import importlib

    doctor = importlib.import_module("yunshu_cli.doctor")
    output = importlib.import_module("yunshu_cli._output")
    monkeypatch.setattr(output, "_json_mode", output._json_mode)
    monkeypatch.setattr(output, "_real_stdout", output._real_stdout)

    def json_mode(enabled):
        # CliRunner owns stdout/stderr capture; do not replace its streams.
        output._json_mode = enabled
        output._real_stdout = sys.stdout

    monkeypatch.setattr(output, "set_json_mode", json_mode)
    healthy = [
        doctor.Check(name, "ok", "fixture")
        for name in ("platform", "python", "mlx", "memory", "port")
    ]

    def checks(model, host, port):
        return healthy + (doctor.check_model(model, {}) if model else [])

    monkeypatch.setattr(doctor, "run_checks", checks)

    def invoke(*args):
        result = runner.invoke(app, ["--json", "doctor", *args])
        assert result.output, (result.exit_code, result.exception)
        return result.exit_code, json.loads(result.output)

    return invoke, doctor


def test_doctor_json_passes_here(doctor_cli):
    invoke, _ = doctor_cli
    code, out = invoke("--port", "18989")
    names = {c["name"] for c in out["checks"]}
    assert {"platform", "python", "mlx", "memory", "port"} <= names
    assert code == 0 and out["ok"], out


def test_doctor_fails_on_missing_model(tmp_path, doctor_cli):
    invoke, _ = doctor_cli
    code, out = invoke("-m", str(tmp_path / "missing"), "--port", "18989")
    assert code == 1 and not out["ok"]
    bad = [c for c in out["checks"] if c["status"] == "fail"]
    assert bad[0]["name"] == "model" and "yunshu pull" in bad[0]["fix"]


def test_doctor_cli_keeps_dependency_failures(doctor_cli, monkeypatch):
    invoke, doctor = doctor_cli
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda *args: [
            doctor.Check("version mlx-vlm", "fail", "below minimum", "upgrade")
        ],
    )
    code, out = invoke()
    assert code == 1 and not out["ok"]
    assert out["checks"][0]["name"] == "version mlx-vlm"


def test_doctor_model_checks(tmp_path):
    from yunshu_cli.doctor import check_model

    m = _model(tmp_path / "m")
    gib = 1024**3
    assert check_model(str(m), {"memory_size": 64 * gib})[1].status == "ok"
    # Weights larger than memory cannot load.
    assert check_model(str(m), {"memory_size": 8})[1].status == "fail"
    # Weights near the working set leave no room for the KV cache.
    info = {"memory_size": 64 * gib, "max_recommended_working_set_size": 17}
    assert check_model(str(m), info)[1].status == "warn"
    # A models directory passed as a model is named as such.
    c = check_model(str(tmp_path), {})[0]
    assert c.status == "warn" and "subfolders are models" in c.detail


def test_doctor_port_in_use():
    import socket

    from yunshu_cli.doctor import check_port

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        c = check_port("127.0.0.1", port)
    assert c.status == "warn" and str(port + 1) in c.fix


# ── serve pre-flight ──────────────────────────────────────────────────────


def test_serve_reports_a_missing_model_but_still_starts(tmp_path, monkeypatch):
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(1))
    r = runner.invoke(app, ["serve", "-m", str(tmp_path / "missing")])
    assert "no such directory" in r.output
    assert "not ready" in r.output
    assert started


def test_serve_defaults_to_localhost():
    import inspect

    from yunshu_cli.serve import serve

    assert inspect.signature(serve).parameters["host"].default.default == "127.0.0.1"


# ── service ───────────────────────────────────────────────────────────────


@pytest.fixture
def launchctl(monkeypatch):
    calls = []

    def fake(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[0] == "print" else 0, "", "")

    monkeypatch.setattr(service, "_launchctl", fake)
    return calls


def test_service_plist(home, tmp_path):
    m = _model(tmp_path / "m")
    plist = service.build_plist(
        service.serve_args(str(m), None, "127.0.0.1", 18980, None, ["MTP=1"])
    )
    assert plist["Label"] == paths.SERVICE_LABEL
    args = plist["ProgramArguments"]
    assert args[:4] == [sys.executable, "-m", "yunshu_cli", "serve"]
    assert args[args.index("--model") + 1] == str(m.resolve())
    assert args[args.index("--port") + 1] == "18980"
    assert args[-2:] == ["--set", "MTP=1"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["StandardOutPath"].startswith(str(home / "Library" / "Logs"))
    plistlib.dumps(plist)  # serializable


def test_service_install_dry_run_writes_nothing(home, tmp_path, launchctl):
    m = _model(tmp_path / "m")
    r = runner.invoke(app, ["service", "install", "-m", str(m), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "<plist" in r.output
    assert not paths.launch_agent_plist().exists()
    assert launchctl == []


def test_service_install_uninstall(home, tmp_path, launchctl):
    m = _model(tmp_path / "m")
    r = runner.invoke(app, ["service", "install", "-m", str(m), "-p", "18980"])
    assert r.exit_code == 0, r.output
    dest = paths.launch_agent_plist()
    assert dest.exists()
    assert ("bootstrap", f"gui/{os.getuid()}", str(dest)) in launchctl
    assert service._installed_address(dest) == ("127.0.0.1", 18980)
    # A second install does not silently replace the first.
    assert runner.invoke(app, ["service", "install", "-m", str(m)]).exit_code == 1
    assert runner.invoke(app, ["service", "uninstall"]).exit_code == 0
    assert not dest.exists()


def test_service_install_needs_a_model(home, launchctl):
    r = runner.invoke(app, ["service", "install"])
    assert r.exit_code == 2
    r = runner.invoke(app, ["service", "install", "-m", str(home / "missing")])
    assert r.exit_code == 2 and "no such directory" in r.output


def test_parse_launchctl_print():
    text = (
        "gui/501/com.yuhuanstudio.yunshu = {\n"
        "\tstate = running\n\tpid = 4242\n\tlast exit code = (never exited)\n}"
    )
    assert service.parse_print(text) == {
        "pid": 4242,
        "state": "running",
        "last_exit_code": "(never exited)",
    }


def test_serve_bounds_graceful_shutdown_by_the_drain_timeout(tmp_path, monkeypatch):
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: seen.update(k))
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", "7")
    runner.invoke(app, ["serve", "-m", str(tmp_path / "missing")])
    assert seen["timeout_graceful_shutdown"] == 7
