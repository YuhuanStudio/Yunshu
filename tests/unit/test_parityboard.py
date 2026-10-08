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


def item(board, metric="ttft_cold_s", kind="prose"):
    return next(
        i
        for i in board["items"]
        if i["ctx"] == 1024 and i["kind"] == kind and i["metric"] == metric
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


def test_memory_cells_without_system_delta_method_are_flagged_not_compared(tmp_path):
    run, jobs, _, _ = fixture_cell(tmp_path, "yunshu-new", 0, 1.0)
    fixture_cell(tmp_path, "splash", 0, 0.5)
    board = pb.build([run], jobs)
    got = item(board, "memory_peak_gib", "session")
    assert (
        got["ours"] is None
        and got["provisional"] is None
        and got["status"] == "unknown"
    )
    assert {f["method"] for f in board["memory_method_flagged"]} == {"single-pid"}
    assert "needing rerun" in pb.markdown(board)


def test_head_to_head_and_summary(tmp_path):
    run, jobs, _, _ = fixture_cell(tmp_path, "yunshu-new", 0, 1.0)
    fixture_cell(tmp_path, "splash", 0, 0.5)  # splash faster: ours loses
    fixture_cell(tmp_path, "llamacpp", 0, 2.0)  # llamacpp slower: ours wins
    board = pb.build([run], jobs)
    h = board["head_to_head"]
    assert h["splash"]["decode"] == {
        "win": 0,
        "tie": 0,
        "loss": 3,
        "n": 3,
        "provisional": True,
    }
    assert h["llamacpp"]["decode"]["win"] == 3 and h["llamacpp"]["ttft"]["win"] == 2
    assert h["splash"]["followup_ttft"]["loss"] == 1
    assert "mlxlm" not in h
    assert "NOT behind" in board["summary"] and board["parity"] == 0
    assert "Head-to-head" in pb.markdown(board)


def test_head_to_head_tie_within_band():
    def m(v, mad):
        return {"status": "measured", "median": v, "mad": mad, "samples": [1, 2, 3]}

    got = pb.head_to_head(
        [("decode_cold_tps", True, {"yunshu-new": m(100, 3), "omlx": m(105, 3)})]
    )
    assert got["omlx"]["decode"] == {
        "win": 0,
        "tie": 1,
        "loss": 0,
        "n": 1,
        "provisional": False,
    }


def add_conc(out, agg=40.0, n=8, bad=False):
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    meta = {
        k: rows[0][k]
        for k in (
            "engine",
            "git_sha",
            "version",
            "checkpoint",
            "snapshot_rep",
            "device",
        )
    }
    conc = dict(
        meta,
        part="conc",
        n=n,
        trial=0,
        wall_s=10,
        total_tokens=n * 256 - (1 if bad else 0),
        agg_tps=agg,
        per_req_dec=[5.0] * n,
        ttfts=[1.0] * n,
        pts=[32768] * n,
        cts=[256] * n,
    )
    rows.insert(-1, conc)
    out.write_text("\n".join(map(json.dumps, rows)))


def test_conc8_item_and_incomplete_conc_rejected(tmp_path):
    for eng, agg in (("yunshu-new", 30.0), ("splash", 40.0)):
        for rep in range(3):
            run, jobs, out, _ = fixture_cell(tmp_path, eng, rep)
            add_conc(out, agg + rep)
    board = pb.build([run], jobs)
    it = [
        i
        for i in board["items"]
        if i["metric"] == "conc8_agg_tps" and i["ctx"] == 32768
    ][0]
    assert it["status"] == "gap" and it["best_engine"] == "splash"
    others = [
        i
        for i in board["items"]
        if i["metric"] == "conc8_agg_tps" and i["ctx"] != 32768
    ]
    assert others and all(
        i["status"] == "unknown" for i in others
    )  # missing contexts listed, not dropped
    tmp2 = tmp_path / "bad"
    tmp2.mkdir()
    run, jobs, out, _ = fixture_cell(tmp2, "yunshu-new", 0)
    add_conc(out, bad=True)
    board = pb.build([run], jobs)
    assert any("incomplete conc row" in r["reason"] for r in board["rejected"])


def test_extra_items_unknown_without_sources(tmp_path):
    board = pb.build([], tmp_path)
    assert {e["item"] for e in board["extra"]} == {
        "capability_matrix",
        "accuracy_mmlu_pro",
        "agentic_pass",
    }
    assert all(
        e["status"] == "unknown" and e["measurable_engines"] for e in board["extra"]
    )
    assert not board["gate_open"]


def write_verdict(d, engine, passed, fail=0, complete=True, na=0):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"verdict-{engine}.json").write_text(
        json.dumps(
            {
                "engine": engine,
                "complete": complete,
                "counts": {"pass": passed, "fail": fail, "error": 0, "na": na},
                "applicable": passed + fail,
            }
        )
    )


def test_capability_from_capmatrix_verdict(tmp_path):
    assert pb.capability_item(tmp_path)["status"] == "unknown"
    d = tmp_path / "27b"
    write_verdict(d, "yunshu", 30)
    assert pb.capability_item(tmp_path)["status"] == "unknown"  # fewer than 34 applicable
    write_verdict(d, "yunshu", 40, fail=1)
    assert pb.capability_item(tmp_path)["status"] == "gap"
    write_verdict(d, "yunshu", 40, complete=False)
    assert pb.capability_item(tmp_path)["status"] == "unknown"
    write_verdict(d, "yunshu", 40)
    write_verdict(d, "llamacpp", 20, fail=5)
    item = pb.capability_item(tmp_path)
    assert item["status"] == "parity" and item["rivals"] == {"llamacpp": "20/25"}


def test_capability_ignores_tiny_model_verdicts(tmp_path):
    write_verdict(tmp_path / "tiny", "yunshu", 40)
    assert pb.capability_item(tmp_path)["status"] == "unknown"


def write_arm(root, name, flags):
    d = root / "mmlu_pro"
    d.mkdir(exist_ok=True)
    rows = [{"kind": "meta"}] + [
        {"kind": "q", "id": f"q{i}", "correct": c} for i, c in enumerate(flags)
    ]
    (d / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)))


def test_accuracy_paired(tmp_path):
    base = [i % 2 == 0 for i in range(300)]
    write_arm(tmp_path, "ref", base)
    write_arm(tmp_path, "default", base[:-1] + [not base[-1]])
    assert pb.accuracy_item(tmp_path)["status"] == "parity"  # one discordant pair
    write_arm(tmp_path, "default", [False] * 300)
    assert pb.accuracy_item(tmp_path)["status"] == "gap"
    write_arm(tmp_path, "default", base[:50])
    assert pb.accuracy_item(tmp_path)["status"] == "unknown"  # <200 pairs


def test_agentic_ignores_incomplete_verdicts(tmp_path):
    bad = tmp_path / "a"
    bad.mkdir()
    (bad / "verdict.json").write_text(
        json.dumps({"verdict": "FAIL", "problems": ["missing run"], "current": {}})
    )
    assert pb.agentic_item(tmp_path)["status"] == "unknown"
    ok = tmp_path / "b"
    ok.mkdir()
    (ok / "verdict.json").write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "problems": [],
                "sha": "x",
                "current": {"claude": {"passed": 3, "runs": 4}},
            }
        )
    )
    e = pb.agentic_item(tmp_path)
    assert e["pass_runs"] == {"claude": [3, 4]} and e["status"] == "unknown"
