"""CPU fixtures prove contaminated, failed and incomplete cells cannot open the gate."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/dev"))
import parityboard as pb


def fixture_cell(root, engine="yunshu-new", rep=0, value=1.0):
    run = root / "snapshot014-fixture"
    cells = run / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    name = f"{engine}-d1k-r{rep}"
    job_id = name + "-job"
    out = cells / (name + ".jsonl")
    meta = dict(
        engine=engine,
        git_sha="abc123",
        version="0.1.4",
        checkpoint="Qwen3.8-27B",
        snapshot_rep=rep,
        device="m5",
        spec_mode_expected="ar",
        engaged_spec_mode="ar",
    )
    rows = [dict(meta, part="session")]
    for phase in ("cold", "warm", "turn2"):
        rows.append(
            dict(
                meta,
                part="decode",
                ctx=1024,
                kind="prose",
                phase=phase,
                ct=256 if phase == "turn2" else 2048,
                finish="length",
                ttft_s=value,
                dec_tps=100 / value,
                pt=1024,
                xy={"prefill_tps": 1000 / value},
            )
        )
    rows.extend(
        [
            dict(meta, part="memory", peak_gib=30, idle_gib=20),
            dict(meta, part="part_done", complete=True),
        ]
    )
    out.write_text("\n".join(map(json.dumps, rows)))
    event = dict(
        ev="cell_submitted", cell=name, job=job_id, argv=["--out", str(out)], t=rep
    )
    with (run / "snapshot.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    job = jobs / (job_id + ".json")
    job.write_text(
        json.dumps(
            dict(
                state="done", rc=0, device="m5", contended=False, quiet=True, label=name
            )
        )
    )
    return run, jobs, out, job


def item(board, metric="ttft_cold_s"):
    return next(
        i
        for i in board["items"]
        if i["ctx"] == 1024 and i["kind"] == "prose" and i["metric"] == metric
    )


def test_reps_noise_and_direction(tmp_path):
    for engine, values in [
        ("yunshu-new", (1.01, 1.02, 1.03)),
        ("mlxlm", (0.98, 1, 1.02)),
    ]:
        for rep, value in enumerate(values):
            run, jobs, _, _ = fixture_cell(tmp_path, engine, rep, value)
    board = pb.build([run], jobs)
    assert item(board)["status"] == "parity"
    assert item(board)["gap"] == pytest.approx(0.02)
    assert item(board)["noise_band"] == pytest.approx(0.03)
    assert board["gate_open"] is False  # other unmeasured cells stay unknown
    assert item(board, "decode_cold_tps")["higher_is_better"]


@pytest.mark.parametrize(
    "change",
    ["terminal", "rc", "contended", "device", "phase", "mode", "nan", "marker"],
)
def test_fail_closed(tmp_path, change):
    run, jobs, out, jobfile = fixture_cell(tmp_path)
    rows = pb.read_jsonl(out)
    job = json.loads(jobfile.read_text())
    if change == "terminal":
        rows.pop()
    elif change == "rc":
        job.pop("rc")
    elif change == "contended":
        job["contended"] = True
    elif change == "device":
        job["device"] = "m3"
    elif change == "phase":
        rows.pop(2)
    elif change == "mode":
        rows[0]["engaged_spec_mode"] = "wrong"
    elif change == "nan":
        rows[1]["ttft_s"] = float("nan")
    else:
        (run / "CONTAMINATED.md").write_text("snapshot.yunshu-new-d1k-r0 contaminated")
    out.write_text("\n".join(map(json.dumps, rows)))
    jobfile.write_text(json.dumps(job))
    accepted, rejected, _ = pb.load_cells([run], jobs)
    assert not accepted and len(rejected) == 1


def test_no_self_parity_or_missing_reps(tmp_path):
    for rep in range(3):
        run, jobs, _, _ = fixture_cell(tmp_path, rep=rep)
    assert item(pb.build([run], jobs))["status"] == "unknown"
    for rep in range(2):
        fixture_cell(tmp_path, "mlxlm", rep)
    assert item(pb.build([run], jobs))["status"] == "unknown"


def test_latest_failed_attempt_does_not_fallback(tmp_path):
    run, jobs, out, _ = fixture_cell(tmp_path)
    event = dict(
        ev="cell_submitted",
        cell="yunshu-new-d1k-r0",
        job="missing",
        argv=["--out", str(out)],
        t=100,
    )
    with (run / "snapshot.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    accepted, rejected, _ = pb.load_cells([run], jobs)
    assert not accepted and rejected[0]["job"] == "missing"


def test_duplicate_reps_and_mixed_sha_unknown():
    sample = dict(value=1, git_sha="a", checkpoint="x", version="1", mode="ar")
    assert (
        pb.summarize({0: sample, 1: sample, 2: dict(sample, git_sha="b")})["status"]
        == "unknown"
    )
    assert pb.summarize({0: sample, 1: sample, 2: None})["status"] == "unknown"


def test_markdown_and_missing_models(tmp_path):
    b = pb.build([], tmp_path, models=("big-MoE",))
    assert not b["gate_open"] and b["missing"]
    assert "big-MoE" in pb.markdown(b)


def test_provisional_needs_external_engine_and_never_gates(tmp_path):
    run, jobs, _, _ = fixture_cell(tmp_path, "yunshu-new", 0, 1.0)
    board = pb.build([run], jobs)
    assert all(i["provisional"] is None for i in board["items"])
    fixture_cell(tmp_path, "splash", 0, 0.5)
    board = pb.build([run], jobs)
    got = item(board)
    assert got["status"] == "unknown" and got["provisional"]["best_engine"] == "splash"
    assert board["parity"] == 0 and board["gate_open"] is False
