"""Watched engines always run with YUNSHU_OBSERVE=1 (tfbench, agentic, yv non-timing stages, conc probe)."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tfbench = _load("tfbench_observe", "scripts/research/tfbench.py")
servers = _load("agentic_servers_observe", "scripts/research/agentic/servers.py")
conc_probe = _load("conc_probe_observe", "scripts/research/conc_probe.py")


def _tfb_env(monkeypatch, tmp_path, extra=None, tfb_observe=None):
    monkeypatch.setattr(tfbench, "OUT", tmp_path)
    monkeypatch.setattr(tfbench, "free_port", lambda: 18999)
    monkeypatch.setattr(tfbench, "owns_listener", lambda *a: True)
    monkeypatch.delenv("TFB_OBSERVE", raising=False)
    if tfb_observe is not None:
        monkeypatch.setenv("TFB_OBSERVE", tfb_observe)
    launched = {}

    class Proc:
        pid = 123456789

        def poll(self):
            return None

    def launch(cmd, **kw):
        launched["env"] = kw["env"]
        kw["stdout"].write(b"VLM batch runner: apc=off draft=dflash block=8\n")
        kw["stdout"].close()
        return Proc()

    monkeypatch.setattr(tfbench.subprocess, "Popen", launch)
    monkeypatch.setattr(
        tfbench.urllib.request,
        "urlopen",
        lambda *_a, **_kw: io.BytesIO(b'{"data":[{"id":"t"}]}'),
    )
    monkeypatch.setattr(tfbench.Srv, "kill", lambda s: None)
    tfbench.Srv("yunshu", dict(extra or {}), "obs")
    return launched["env"]


def test_tfbench_server_runs_observe(monkeypatch, tmp_path):
    assert _tfb_env(monkeypatch, tmp_path)["YUNSHU_OBSERVE"] == "1"


def test_tfbench_explicit_and_opt_out(monkeypatch, tmp_path):
    env = _tfb_env(monkeypatch, tmp_path, {"YUNSHU_OBSERVE": "0"})
    assert env["YUNSHU_OBSERVE"] == "0"
    env = _tfb_env(monkeypatch, tmp_path, tfb_observe="0")
    assert "YUNSHU_OBSERVE" not in env


@pytest.mark.parametrize("opt_out,expected", [(None, "1"), ("0", None)])
def test_agentic_server_env(monkeypatch, tmp_path, opt_out, expected):
    monkeypatch.setattr(servers, "SERVER_HOME", tmp_path / "home")
    monkeypatch.delenv("YUNSHU_OBSERVE", raising=False)
    monkeypatch.delenv("AGENTIC_OBSERVE", raising=False)
    if opt_out:
        monkeypatch.setenv("AGENTIC_OBSERVE", opt_out)
    seen = {}

    def launch(*args, **kwargs):
        seen.update(kwargs["env"])
        raise OSError("stop")

    monkeypatch.setattr(servers.subprocess, "Popen", launch)
    server = servers.Server("yunshu", "m", 18999, tmp_path / "s.log")
    with pytest.raises(OSError):
        server.start(ready_timeout=0)
    assert seen.get("YUNSHU_OBSERVE") == expected


def _ctx(observe=False):
    from verify import stages

    return stages, SimpleNamespace(observe=observe)


@pytest.mark.parametrize(
    "stage,on",
    [
        ("smoke", True),
        ("identity", True),
        ("apc", True),
        ("speed", False),
        ("conc", False),
    ],
)
def test_yv_observe_per_stage(stage, on):
    stages, ctx = _ctx()
    env = stages.with_observe(ctx, stage, {"A": "1"})
    assert env["YUNSHU_OBSERVE"] == (
        "1" if on else "0"
    )  # speed stages say 0 explicitly


def test_yv_flag_and_explicit_env():
    stages, ctx = _ctx(observe=True)
    assert stages.with_observe(ctx, "speed", {})["YUNSHU_OBSERVE"] == "1"
    stages, ctx = _ctx()
    assert (
        stages.with_observe(ctx, "smoke", {"YUNSHU_OBSERVE": "0"})["YUNSHU_OBSERVE"]
        == "0"
    )


def test_conc_probe_refuses_an_engine_that_is_not_observing(monkeypatch, tmp_path):
    class R:
        def __init__(self, observe):
            self.observe = observe

        def raise_for_status(self):
            pass

        def json(self):
            return {"observe": self.observe}

    monkeypatch.setattr(conc_probe.httpx, "get", lambda *a, **k: R(False))
    assert conc_probe.engine_observes("http://x") is False
    monkeypatch.setattr(conc_probe.httpx, "get", lambda *a, **k: R(True))
    assert conc_probe.engine_observes("http://x") is True
    monkeypatch.setattr(conc_probe.httpx, "get", lambda *a, **k: R(False))
    monkeypatch.setattr(
        sys,
        "argv",
        ["p", "--url", "http://x", "--model", "m", "--out", str(tmp_path / "o")],
    )
    with pytest.raises(SystemExit, match="observe"):
        conc_probe.main()
