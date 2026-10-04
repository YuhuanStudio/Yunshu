import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

DEV = Path(__file__).resolve().parents[2] / "scripts/dev"
sys.path.insert(0, str(DEV))
spec = importlib.util.spec_from_file_location(
    "gpuq_preflight", DEV / "gpuq_preflight.py"
)
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

PY = sys.executable
# Not /bin/sleep: macOS hides the environment of platform binaries; leftovers are Python servers.
SLEEPER = [PY, "-c", "import time; time.sleep(30)"]


def test_preflight_catches_jobs_that_cannot_succeed(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("print(1)\n")
    bad = tmp_path / "bad.py"
    # The compiler (not the parser) rejects this one: it failed a real job.
    bad.write_text("X = 1\ndef f():\n    print(X)\n    global X\n")
    env = dict(os.environ)
    cwd = str(tmp_path)
    assert pf.preflight([PY, str(good)], cwd, env) == []
    assert pf.preflight(["env", "A=1", PY, "good.py", "--x"], cwd, env) == []
    assert "bad.py" in pf.preflight([PY, "bad.py"], cwd, env)[0]
    assert "does not exist" in pf.preflight([PY, "missing.py"], cwd, env)[0]
    assert "module not found" in pf.preflight([PY, "-m", "no_such_mod_x"], cwd, env)[0]
    assert pf.preflight([PY, "-m", "json.tool"], cwd, env) == []
    assert (
        "model path"
        in pf.preflight(
            [PY, "good.py", "--model", "/Volumes/P5Plus/models/none/x"], cwd, env
        )[0]
    )
    assert pf.preflight(["true"], cwd, env) == []
    assert "cwd" in pf.preflight(["true"], str(tmp_path / "gone"), env)[0]


def test_module_found_through_job_pythonpath(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mymod_x.py").write_text("")
    cmd = ["env", "PYTHONPATH=pkg", PY, "-m", "mymod_x"]
    assert pf.preflight(cmd, str(tmp_path), dict(os.environ)) == []


def test_kind_drops_worker_and_counters():
    assert pf.kind("wide8-quality200-1") == {"quality200"}
    assert pf.kind("prefill6-readme-http32k-r3-0800") == {"readme", "http32k"}
    assert pf.kind("memab-27b-r2") == frozenset()


def test_timeout_learned_from_same_kind():
    now = 1_000_000.0
    hist = [
        dict(
            id="a",
            label="wide6-production-quality200-20",
            state="done",
            started=now - 3000,
            ended=now - 1800,
            pauses=[],
        ),
        dict(
            id="b",
            label="wide6-api-small-r0",
            state="done",
            started=now - 600,
            ended=now - 500,
        ),
    ]
    t, note = pf.learned_timeout("wide8-quality200-1", 20 * 60, hist, now)
    assert t == 31 * 60 and "a ran 20.0 min" in note
    assert pf.learned_timeout("wide8-quality200-1", 60 * 60, hist, now) == (3600, None)
    assert pf.learned_timeout("wide8-new-thing", 1200, hist, now) == (1200, None)


def test_reap_kills_only_this_jobs_leftovers(tmp_path):
    rc = str(tmp_path / "j1.rc")
    other = str(tmp_path / "j2.rc")
    mine = subprocess.Popen(SLEEPER, env={**os.environ, "GPUQ_RC": rc})
    theirs = subprocess.Popen(SLEEPER, env={**os.environ, "GPUQ_RC": other})
    try:
        time.sleep(0.5)  # let both interpreters start
        assert pf.reap(rc, grace_s=5) == [mine.pid]
        mine.wait(5)
        time.sleep(0.1)
        assert theirs.poll() is None
    finally:
        for p in (mine, theirs):
            p.kill()
            p.wait()


def test_timeout_is_not_learned_from_unrelated_jobs():
    now = 1_000_000.0
    hist = [
        dict(
            id="g",
            label="gate-012-final",
            state="done",
            started=now - 9500,
            ended=now - 100,
        )
    ]
    assert pf.kind("memory1-final-identity-cand.default-a1-814b") == {
        "identity",
        "cand.default",
    }
    assert pf.learned_timeout(
        "memory1-final-identity-cand.default-a1-814b", 900, hist, now
    ) == (900, None)


def test_finished_runs_outrank_timed_out_ones():
    now = 1_000_000.0
    hist = [
        dict(
            id="slow",
            label="wide8-quality200-2",
            state="timeout",
            started=now - 2800,
            ended=now - 100,
        ),
        dict(
            id="ok",
            label="prefill7-quality200",
            state="done",
            started=now - 460,
            ended=now - 100,
        ),
    ]
    t, note = pf.learned_timeout("wide8-quality200-3", 300, hist, now)
    assert t == 10 * 60 and "ok ran 6.0 min" in note
    assert pf.learned_timeout("wide8-quality200-3", 300, hist[:1], now)[0] > 60 * 60


def test_ensure_out_dirs_creates_missing_parent(tmp_path):
    cmd = [
        "python",
        "x.py",
        "--out",
        "runs/a/b.jsonl",
        "--output=c/d.json",
        "--out-dir",
        "e",
    ]
    made = pf.ensure_out_dirs(cmd, str(tmp_path))
    assert (tmp_path / "runs/a").is_dir() and (tmp_path / "c").is_dir()
    assert (tmp_path / "e").is_dir() and len(made) == 3
    assert pf.ensure_out_dirs(cmd, str(tmp_path)) == []  # nothing left to make


def test_ensure_out_dirs_skips_shell_variables(tmp_path):
    assert pf.ensure_out_dirs(["x", "--out", "$D/f"], str(tmp_path)) == []
