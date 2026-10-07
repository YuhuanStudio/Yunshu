"""CPU preflight for the absolute reserve pilot, without a model/server/GPU."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
spec = importlib.util.spec_from_file_location(
    "bigmoe_probe", ROOT / "scripts/research/bigmoe_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_available_does_not_double_count_purgeable():
    assert (
        probe.available_bytes(
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 3.\nPages purgeable: 12.\n"
        )
        == 33 * 16384
    )


def test_semantic_and_engaged_checks():
    rows = [{"part": "decode", "ct": 2, "text": "4"}, {"complete": True}]
    log = "VLM batch runner: generic draft=off"
    assert probe.validate(rows, log) == ""
    assert probe.validate(rows[:-1], log) == "missing complete record"
    assert probe.validate(rows, "")
    rows[0]["ct"] = 0
    assert probe.validate(rows, log) == "empty generation"


def test_admission_refuses_before_subprocess(tmp_path, monkeypatch):
    args = probe.build_parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--src",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    monkeypatch.setattr(probe, "census", lambda _: {"base_bytes": 79 * 2**30})
    monkeypatch.setattr(probe, "sample_available", lambda: 100 * 2**30)
    monkeypatch.setattr(
        probe.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    result = probe.run(args)
    assert not result["complete"]
    assert result["failure"].startswith("admission:")


def test_fake_server_pipeline_and_guard(tmp_path, monkeypatch):
    import json

    args = probe.build_parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--src",
            str(tmp_path / "python"),
            "--out",
            str(tmp_path / "out.jsonl"),
        ]
    )
    monkeypatch.setattr(probe, "census", lambda _: {"base_bytes": 79 * 2**30})
    memory = iter([120 * 2**30, 40 * 2**30])
    monkeypatch.setattr(probe, "sample_available", lambda: next(memory))
    monkeypatch.setattr(
        probe,
        "process_tree_memory",
        lambda _: {"physical_footprint_sum_bytes": 80 * 2**30},
    )
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)

    class Child:
        pid = 123
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self, **kwargs):
            return 0

    def launch(cmd, env, **kwargs):
        assert env["TFB_YUNSHU_SRC"] == str(args.src)
        assert "--smoke" in cmd
        raw = Path(cmd[cmd.index("--out") + 1])
        raw.write_text(
            json.dumps({"part": "decode", "ct": 2, "text": "4"})
            + '\n{"complete":true}\n'
        )
        logdir = Path(env["TFB_OUT"])
        logdir.mkdir(exist_ok=True)
        (logdir / "server-fake.log").write_text("VLM batch runner: generic draft=off")
        return Child()

    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    result = probe.run(args)
    assert result["complete"] and result["min_available_bytes"] == 40 * 2**30
    assert result["physical_footprint_peak_bytes"] == 80 * 2**30
    stopped = []
    memory = iter([120 * 2**30, 31 * 2**30])
    monkeypatch.setattr(probe, "stop_owned", lambda proc: stopped.append(proc.pid))
    result = probe.run(args)
    assert not result["complete"] and result["failure"] == "reserve guard crossed"
    assert stopped == [123]


def test_invalid_reserve_cannot_disable_the_guard(tmp_path, monkeypatch):
    import pytest

    for value in ("nan", "inf", "0", "29"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "probe",
                "--model",
                str(tmp_path),
                "--src",
                str(tmp_path),
                "--out",
                str(tmp_path / "out.jsonl"),
                "--reserve-gib",
                value,
            ],
        )
        with pytest.raises(SystemExit) as exc:
            probe.main()
        assert exc.value.code == 2
