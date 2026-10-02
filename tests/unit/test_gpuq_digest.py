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


def test_declared_outputs_complete_and_label_filter(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    now = time.time()
    output = tmp_path / "declared.jsonl"
    output.write_text("partial\n")
    _job(tmp_path, "a", "mine-r0", "done", now - 10)
    p = tmp_path / "jobs/a.json"
    data = json.loads(p.read_text())
    data.update(outputs=[str(output)], expect_complete=True)
    p.write_text(json.dumps(data))
    _job(tmp_path, "b", "other-r0", "failed", now - 5, rc=7)
    assert gd.main(["--label-prefix", "mine", "--since", "1h"]) == 1
    text = capsys.readouterr().out
    assert "1 finished jobs" in text and "complete" in text and "other" not in text
    output.write_text("partial\ncomplete\n")
    assert gd.main(["--label-prefix", "mine", "--since", "1h"]) == 0
    assert not (tmp_path / ".digest_marker").exists()
    assert gd.main(["--label-prefix", "absent", "--since", "1h"]) == 0


def test_filtered_digest_keeps_global_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    marker = time.time() - 100
    gd.write_marker(tmp_path, marker)
    _job(tmp_path, "a", "mine", "done", time.time() - 1)
    assert gd.main(["--label-prefix", "mine"]) == 0
    assert gd.read_marker(tmp_path) == marker


def test_digest_rejects_done_with_nonzero_rc_and_cancelled(tmp_path):
    now = time.time()
    _job(tmp_path, "badrc", "mine", "done", now - 3, rc=7)
    _job(tmp_path, "cancel", "mine", "cancelled", now - 2, rc=None)
    assert {e["id"] for e in gd.collect(tmp_path, now - 10, now)["problems"]} == {
        "badrc",
        "cancel",
    }


def test_marker_round_trips_without_losing_timestamp_precision(tmp_path):
    marker = 1790942392.3600042
    gd.write_marker(tmp_path, marker)
    assert gd.read_marker(tmp_path) == marker
