"""gpuq digest: finished jobs by family, with failures and empty outputs flagged."""

import importlib.util
import json
import sys
import time
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts/dev/gpuq_digest.py"
_spec = importlib.util.spec_from_file_location("gpuq_digest", _p)
gd = importlib.util.module_from_spec(_spec)
sys.modules["gpuq_digest"] = gd
_spec.loader.exec_module(gd)


def _job(root, jid, label, state, ended, cmd=None, rc=0, cwd="/"):
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    (root / "jobs" / f"{jid}.json").write_text(
        json.dumps(
            {
                "id": jid,
                "label": label,
                "cmd": cmd or ["true"],
                "cwd": cwd,
                "state": state,
                "rc": rc,
                "started": ended - 60,
                "ended": ended,
            }
        )
    )


def test_family_and_output_paths():
    assert gd.family("paired-gsm8k-ref-r3") == "paired-gsm8k-ref"
    assert gd.family("agentic-yunshu-default-claude-p2s11") == (
        "agentic-yunshu-default-claude"
    )
    assert gd.family("gate-main-quick") == "gate-main-quick"
    job = {"cwd": "/w", "cmd": ["py", "x", "--out", "r/a.jsonl", "--output=/abs/b"]}
    assert gd.output_paths(job) == [Path("/w/r/a.jsonl"), Path("/abs/b")]


def test_flags_failures_and_empty_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    now = time.time()
    good = tmp_path / "good.jsonl"
    good.write_text("x\n")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    _job(tmp_path, "a", "bench-x-r0", "done", now - 50, ["p", "--out", str(good)])
    _job(tmp_path, "b", "bench-x-r1", "done", now - 40, ["p", "--out", str(empty)])
    _job(tmp_path, "c", "bench-x-r2", "done", now - 30, ["p", "--output", "nope.json"])
    _job(tmp_path, "d", "other", "failed", now - 20, rc=1)
    _job(tmp_path, "e", "other2", "stalled", now - 10, rc=None)
    _job(tmp_path, "old", "bench-x-r9", "failed", now - 7200)
    res = gd.collect(tmp_path, now - 3600, now)
    bad = {e["id"]: e["issues"] for e in res["problems"]}
    assert set(bad) == {"b", "c", "d", "e"}
    assert "empty" in bad["b"][0] and "missing" in bad["c"][0]
    assert sorted(res["families"]) == ["bench-x", "other", "other2"]
    text = gd.render(res, now - 3600, now)
    assert "FLAGGED" in text and "bench-x  [done=3]" in text


def test_marker_moves_only_when_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    now = time.time()
    _job(tmp_path, "a", "t-r0", "done", now - 100)
    assert gd.main([]) == 0
    assert "1 finished jobs" in capsys.readouterr().out
    marker = gd.read_marker(tmp_path)
    assert marker is not None and marker >= now
    assert gd.main([]) == 0  # nothing new since the marker
    assert "0 finished jobs" in capsys.readouterr().out
    m2 = time.time() - 50
    gd.write_marker(tmp_path, m2)
    _job(tmp_path, "b", "t-r1", "lost", time.time() - 10, rc=None)
    assert gd.main(["--peek"]) == 1  # shown and flagged, marker untouched
    assert "state lost" in capsys.readouterr().out
    assert abs(gd.read_marker(tmp_path) - m2) < 0.01
    assert gd.main(["--since", "1d"]) == 1
    assert abs(gd.read_marker(tmp_path) - m2) < 0.01  # --since never moves it
