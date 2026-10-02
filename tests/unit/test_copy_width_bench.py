import importlib
import json
from pathlib import Path

import pytest


def fake_bench(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/research"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("bench_copy_width")
    killed = []
    log = tmp_path / "server.log"
    log.write_text("Speculative decoding: mtp (checkpoint MTP head)\n")

    class Server:
        def __init__(self, engine, env, tag):
            self.rows = int(env["YUNSHU_SPEC_COPY_ROWS"])
            self.model, self.url, self.log = "fake", "unused", log

        def kill(self):
            killed.append(self.rows)

    monkeypatch.setattr(module, "freeze", lambda output: (tmp_path, {"head": "fake"}))
    monkeypatch.setattr(module.bench, "Srv", Server)
    monkeypatch.setattr(module.bench, "send", lambda *a: {})
    monkeypatch.setattr(module.bench, "YUNSHU_SRC", module.bench.YUNSHU_SRC)
    monkeypatch.setattr(module.bench, "OUT", module.bench.OUT)
    monkeypatch.setattr(module.signal, "signal", lambda *a: None)
    output = tmp_path / "sweep.jsonl"
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["bench", "--output", str(output), "--rows", "8", "16", "--reps", "1"],
    )
    return module, killed, output


def test_failed_arm_is_recorded_and_later_arms_still_finish(tmp_path, monkeypatch):
    module, killed, output = fake_bench(tmp_path, monkeypatch)

    def decode(server, out, args):
        if server.rows == 8:
            raise RuntimeError("bad arm")

    monkeypatch.setattr(module.bench, "part_decode", decode)
    assert module.main() == 1
    assert killed == [8, 16]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["rc"] for row in records if "rc" in row] == [1, 0]
    assert records[-1]["complete"] and not records[-1]["success"]
    successful = tmp_path / "sweep-w16-r0.jsonl"
    assert json.loads(successful.read_text().splitlines()[-1])["complete"]


def test_interrupt_closes_only_the_server_started_by_this_arm(tmp_path, monkeypatch):
    module, killed, _ = fake_bench(tmp_path, monkeypatch)

    def interrupted(*args):
        raise SystemExit(143)

    monkeypatch.setattr(module.bench, "part_decode", interrupted)
    with pytest.raises(SystemExit):
        module.main()
    assert killed == [8]


def test_policy_comparison_records_separate_fixed_and_cost_arms(tmp_path, monkeypatch):
    module, killed, output = fake_bench(tmp_path, monkeypatch)
    curve = tmp_path / "curve.jsonl"
    curve.write_text("{}\n")
    log = tmp_path / "server.log"
    log.write_text("Speculative decoding: mtp\nCopy cost policy: test curve\n")
    monkeypatch.setattr(module.bench, "part_decode", lambda *args: None)
    monkeypatch.setattr(module.bench.subprocess, "Popen", module.bench.subprocess.Popen)
    module.sys.argv += ["--cost-curve", str(curve), "--compare-policy"]
    assert module.main() == 0
    assert killed == [8, 8, 16, 16]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    arms = [row for row in records if "rc" in row]
    assert [bool(row["cost_curve"]) for row in arms] == [False, True, False, True]
    assert all(row["rc"] == 0 for row in arms)
