import asyncio
import json
from pathlib import Path

import pytest

from yunshu_gateway.resident_activity import ResidentActivityMiddleware, atomic_json


@pytest.mark.asyncio
async def test_fresh_ack_covers_full_stream(tmp_path, monkeypatch):
    path = tmp_path / "activity.json"
    monkeypatch.setattr(
        "yunshu_gateway.resident_activity.settings.get", lambda k: str(path)
    )
    monkeypatch.setattr(
        "yunshu_gateway.resident_activity.settings.get_float", lambda k: 0.2
    )
    entered = []

    async def app(scope, receive, send):
        entered.append(True)
        assert json.loads(path.read_text())["active"] == 1
        await asyncio.sleep(0.01)
        assert json.loads(path.read_text())["active"] == 1

    middleware = ResidentActivityMiddleware(app)
    atomic_json(Path(str(path) + ".ack"), {"ticket": "stale", "safe": True})

    async def scheduler():
        while not path.exists():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.02)
        assert not entered
        ticket = json.loads(path.read_text())["ticket"]
        atomic_json(Path(str(path) + ".ack"), {"ticket": ticket, "safe": True})

    await asyncio.gather(
        middleware({"type": "http", "path": "/v1/chat/completions"}, None, None),
        scheduler(),
    )
    assert entered and json.loads(path.read_text())["active"] == 0


@pytest.mark.asyncio
async def test_timeout_never_calls_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yunshu_gateway.resident_activity.settings.get", lambda k: str(tmp_path / "a")
    )
    monkeypatch.setattr(
        "yunshu_gateway.resident_activity.settings.get_float", lambda k: 0.01
    )

    async def app(*args):
        pytest.fail("engine admitted without scheduler acknowledgement")

    messages = []

    async def send(message):
        messages.append(message)

    await ResidentActivityMiddleware(app)(
        {"type": "http", "path": "/v1/chat/completions"}, None, send
    )
    assert messages[0]["status"] == 503
    assert json.loads((tmp_path / "a").read_text())["active"] == 0


def test_local_gate_requires_memory_and_fresh_ticket(
    gpuq_module, tmp_path, monkeypatch
):
    g = gpuq_module
    path = tmp_path / "activity"
    gate = g.ServingGate()
    gate.configure(
        {"activity_files": [str(path)], "idle_resume_s": 2, "reserve_gb": 16}
    )
    assert gate.poll(1) is True
    assert g.mem_blocked({"mem_gb": 24}, gate, lambda: None)
    atomic_json(path, {"schema": 1, "ticket": "new", "active": 1})
    assert gate.poll(2) is True
    gate.acknowledge(False)
    assert not Path(str(path) + ".ack").exists()
    gate.acknowledge(True)
    assert json.loads(Path(str(path) + ".ack").read_text())["ticket"] == "new"
    atomic_json(path, {"schema": 1, "ticket": "new", "active": 0})
    gate.poll(3)
    assert not gate.may_resume(4)
    assert gate.may_resume(5)


@pytest.fixture
def gpuq_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/dev/gpuq.py"
    spec = importlib.util.spec_from_file_location("resident_gpuq", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("priority", [-9, -2, -1, 0])
def test_every_research_priority_pauses(gpuq_module, tmp_path, monkeypatch, priority):
    g = gpuq_module
    monkeypatch.setattr(g, "LOGS", tmp_path)
    signals = []
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: signals.append(sig))
    gate = g.ServingGate(fetch=lambda u, k: True)
    gate.configure({"urls": ["fake"]})
    gate.poll(1)
    p = g.Pauser({"id": "fake", "pid": 123, "priority": priority}, tmp_path / "job")
    monkeypatch.setattr(p, "stopped", lambda: True)
    p.step(gate, False, 1)
    assert p.paused
    assert signals == [g.signal.SIGSTOP]


def test_isolated_launchd_never_targets_product(tmp_path, monkeypatch):
    from yunshu_cli import service

    monkeypatch.setattr(service, "_instance", "dev-servedev-test")
    monkeypatch.setattr(service, "_state_dir", tmp_path)
    assert service._target().endswith(".dev-servedev-test")
    assert service._plist().parent == tmp_path
    assert service.log_file().parent == tmp_path
    assert service.build_plist(["python"])["Label"].endswith(".dev-servedev-test")
