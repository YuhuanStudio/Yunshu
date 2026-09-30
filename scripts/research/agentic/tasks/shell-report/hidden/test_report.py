import json
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(os.environ["AGENTIC_TASK_DIR"])
SRC = Path(os.environ["AGENTIC_TASK_SRC"])


def gen(tmp, seed):
    subprocess.run(
        [sys.executable, str(SRC / "gen.py"), str(tmp), str(seed)], check=True
    )
    return json.loads((tmp / "expected.json").read_text())


def run_report(data_dir, out):
    r = subprocess.run(
        [sys.executable, "report.py", str(data_dir), str(out)],
        cwd=TASK,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[-500:]
    return json.loads(Path(out).read_text())


def check(got, want):
    assert got["files"] == want["files"]
    assert got["rows"] == want["rows"]
    assert got["bad_rows"] == want["bad_rows"]
    assert list(got["revenue_by_region"]) == list(want["revenue_by_region"])
    for k, v in want["revenue_by_region"].items():
        assert abs(got["revenue_by_region"][k] - v) < 0.011, k


def test_fresh_data_other_seeds(tmp_path):
    for seed in (99, 1234):
        d = tmp_path / str(seed)
        d.mkdir()
        want = gen(d, seed)
        check(run_report(d / "data", d / "out.json"), want)


def test_original_data_report_json(tmp_path):
    want = gen(tmp_path, 7)
    got = json.loads((TASK / "report.json").read_text())
    check(got, want)
    # the shipped data is unchanged
    for f in (tmp_path / "data").iterdir():
        assert (TASK / "data" / f.name).read_bytes() == f.read_bytes()
