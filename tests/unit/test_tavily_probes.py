"""CPU preflight for probes and SDK harnesses before any GPU submission."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
