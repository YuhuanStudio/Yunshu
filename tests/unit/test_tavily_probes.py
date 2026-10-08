"""CPU preflight for probes and SDK harnesses before any GPU submission."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_probe_waits_for_busy_port_pool_and_fails_fast_otherwise():
    import pytest

    probe = module("tavily_probe")
    now = [0]

    def sleep(seconds):
        now[0] += seconds

    def busy():
        if now[0] < 4:
            raise RuntimeError("no free port in 18990-18996")
        return "ready"

    assert probe.wait_for_port(busy, clock=lambda: now[0], sleep=sleep) == "ready"
    assert now[0] == 4
    now[0] = 0
    with pytest.raises(RuntimeError, match="no free port"):
        probe.wait_for_port(busy, timeout=1, clock=lambda: now[0], sleep=sleep)
    assert now[0] == 1
    with pytest.raises(RuntimeError, match="model missing"):
        probe.wait_for_port(
            lambda: (_ for _ in ()).throw(RuntimeError("model missing"))
        )


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/research" / (name + ".py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_probe_dry_run_and_fail_closed(tmp_path):
    probe = module("tavily_probe")
    assert not probe.validate([{"complete": True}])[0]
    assert not probe.validate([{"check": "x", "pass": False}, {"complete": True}])[0]
    assert (
        probe.main(
            [
                "--model",
                str(tmp_path),
                "--src",
                str(ROOT / "python"),
                "--out",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )


def test_sdk_validation_and_args(tmp_path):
    sdk = module("tavily_sdk_parity")
    args = sdk.parser().parse_args(["--out", str(tmp_path / "out")])
    assert args.base_url is None
    assert not sdk.valid([{"complete": True}])
    sdk.check_response(
        {
            "response_time": 1,
            "request_id": "x",
            "usage": {},
            "results": [],
            "images": [],
        },
        "search",
    )


def test_probe_lifecycle_is_exercised_on_cpu(tmp_path, monkeypatch):
    from types import SimpleNamespace

    probe = module("tavily_probe")
    events = []

    class Server:
        def __init__(self, *args):
            events.append("start")

        def wait_ready(self):
            events.append("ready")

        def stop(self):
            events.append("stop")

    class Fake:
        url = "http://fixture"

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        probe,
        "load_dependencies",
        lambda: (
            Server,
            lambda: 18999,
            lambda *a, **kw: SimpleNamespace(),
            Fake,
            lambda c: {},
            lambda c: {"schema_valid": True},
        ),
    )
    assert (
        probe.main(
            [
                "--model",
                str(tmp_path),
                "--src",
                str(ROOT / "python"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert events == ["start", "ready", "stop", "close"]
