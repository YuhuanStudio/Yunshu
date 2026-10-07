import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ttft_warm_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/ttft_warm_probe.py",
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_timeline_collapses_runs_and_drops_short_ones():
    s = [(1.000 + 0.002 * i, "mlx", ["a", "b"]) for i in range(10)]
    s += [(1.020 + 0.002 * i, "mlx", ["c"]) for i in range(2)]
    s += [(1.5, "mlx", ["late"])]
    out = probe.timeline(s, 1.0, 1.1)
    assert list(out) == ["mlx"]
    assert [r[2] for r in out["mlx"]] == ["a < b"]
    assert out["mlx"][0][0] == 0.0


def test_load_samples_reads_every_pid_file(tmp_path):
    for pid in (1, 2):
        (tmp_path / f"samples.{pid}").write_text(
            json.dumps([float(pid), "t", ["x"]]) + "\n"
        )
    assert len(probe.load_samples(str(tmp_path / "samples"))) == 2


def test_sampler_roundtrip_in_a_real_child_process(tmp_path):
    import os
    import subprocess
    import sys
    import time

    site = Path(probe.HERE) / "ttft_probe_site"
    env = dict(
        os.environ, PYTHONPATH=str(site), TTFT_PROBE_OUT=str(tmp_path / "samples")
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(0.01)"],
        env=env,
        start_new_session=True,
    )
    try:
        time.sleep(2)
        probe.signal_dump(child.pid)
        time.sleep(1)
        rows = probe.load_samples(str(tmp_path / "samples"))
        assert rows and rows[0][1] == "MainThread"
        t = rows[-1][0]
        assert probe.timeline(rows, t - 0.3, t)
    finally:
        child.kill()
