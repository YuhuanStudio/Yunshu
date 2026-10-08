import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "gpuq_report", Path(__file__).resolve().parents[2] / "scripts/dev/gpuq_report.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


T = 1000


def job(i, s, e, state="done", prio=0, sub=None, **kw):
    return {
        "id": i,
        "label": kw.pop("label", i),
        "started": s,
        "ended": e,
        "submitted": sub if sub is not None else s,
        "state": state,
        "priority": prio,
        **kw,
    }


def test_busy_union_pause_waste_and_wait():
    jobs = [
        job("a", 0, 600),  # 10 min ok
        job("b", 300, 900, state="failed"),  # overlaps a: union 0..900
        job("c", 1000, 1600, contended=True, sub=400, prio=-1, pauses=[[1100, 1200]]),
        job("d", 2000, 2100, state="cancelled"),
    ]
    jobs = [
        {
            **j,
            "started": j["started"] + T,
            "ended": j["ended"] + T,
            "submitted": j["submitted"] + T,
            "pauses": [[a + T, b + T] for a, b in j.get("pauses", [])],
        }
        for j in jobs
    ]
    r = R.build(jobs, T, T + 3600)
    assert abs(r["run_min"] - 1600 / 60) < 0.1
    assert abs(r["paused_min"] - 100 / 60) < 0.1
    assert r["wasted_min"] == {"cancelled": 1.7, "contended": 8.3, "failed": 10.0}
    assert r["queue_wait_by_priority"][-1]["median_min"] == 10.0


def test_time_to_verdict_groups_by_sha():
    jobs = [
        job("x", 100, 200, label="gate-abcdef12-smoke", sub=0),
        job("y", 300, 4000, label="gate-abcdef12-long", sub=250),
        job("z", 5000, 5100, label="nosha"),
    ]
    jobs = [
        {
            **j,
            "started": j["started"] + T,
            "ended": j["ended"] + T,
            "submitted": j["submitted"] + T,
        }
        for j in jobs
    ]
    r = R.build(jobs, T, T + 9000)
    assert r["time_to_verdict"]["per_candidate"]["abcdef12"]["hours"] == round(
        4000 / 3600, 2
    )
    assert r["time_to_verdict"]["candidates"] == 1


def test_cli_refuses_empty_dir(tmp_path, capsys):
    (tmp_path / "jobs").mkdir()
    assert R.main(["--dir", str(tmp_path)]) == 1
    (tmp_path / "jobs" / "a.json").write_text(json.dumps(job("a", 1, 2)))
    assert R.main(["--dir", str(tmp_path), "--since", "0"]) == 0
