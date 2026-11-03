"""--profile-rep: layer-2 profile jobs (label ...-prof) that never feed table numbers.

CPU only: the job plan, the profile env, the engine-log parser and the profile row are checked with
fakes. No server, no GPU, no gpuq.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_attribution as fa  # noqa: E402
import fullcmp_profile as fp  # noqa: E402
import fullcmp_run as fr  # noqa: E402

PIN = {
    "version": "0.1.7.dev0",
    "sha": "0f2960047855546a9f13b2631b680352b1c57cfe",
    "python_executable": "/venv/bin/python",
    "launch": {
        "harness_env": {
            "TFB_YUNSHU_BIN": "/venv/bin/yunshu",
            "TFB_YUNSHU_SRC": "/src",
            "PYTHONPATH": "/src/python",
        }
    },
}


def _row(rid, engine, ctx=1024, kind="code", conc=1, status="unverified", group="27B"):
    return {
        "id": rid,
        "engine": engine,
        "model": group,
        "batch": "B1",
        "status": status,
        "thinking": "off",
        "turns": "single",
        "media": "none",
        "sampling": "greedy",
        "prompt_kind": kind,
        "ctx": f"{ctx // 1024}K",
        "ctx_tokens": ctx,
        "out_len": 512,
        "conc": conc,
        "reps": 3,
        "est_gpu_min": 3.0,
        "table": "perf",
    }


def test_profile_jobs_are_extra_jobs_with_prof_label():
    rows = [
        _row("a1", "yunshu"),
        _row("a2", "yunshu", ctx=8192),
        _row("b1", "tf-1.2.1"),
    ]
    plain = fr.plan_batch(rows, "B1", PIN)
    prof = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    extra = [j for j in prof.jobs if j.label.endswith("-prof")]
    assert len(prof.jobs) == len(plain.jobs) + len(extra)
    # one extra job per (engine, group, chunk), not per rep
    assert len(extra) == len({(j.engine, j.group, j.label) for j in extra})
    assert {j.engine for j in extra} == {"yunshu", "tf-1.2.1"}
    assert all(j.flags == ["profile"] for j in extra)
    # the plain jobs are unchanged
    assert [j.label for j in prof.jobs if not j.label.endswith("-prof")] == [
        j.label for j in plain.jobs
    ]


def test_profile_job_writes_outside_the_table_root_and_sets_profile_env():
    rows = [_row("a1", "yunshu")]
    prof = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    job = next(j for j in prof.jobs if j.label.endswith("-prof"))
    text = " ".join(job.cmd)
    assert "YUNSHU_DEBUG_TREE_PROFILE=1" in text
    assert "/profile/yunshu/" in text
    # never the table root's <engine>/<cell>-rN.jsonl layout
    assert f"{fr.RAW_ROOT}/yunshu/a1-r" not in text
    assert "-r0.jsonl" not in text


def test_competitor_profile_job_has_no_yunshu_env():
    rows = [_row("b1", "tf-1.2.1")]
    prof = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    job = next(j for j in prof.jobs if j.label.endswith("-prof"))
    assert "YUNSHU_DEBUG_TREE_PROFILE=1" not in " ".join(job.cmd)
    assert fp.profile_env("tf-1.2.1") == []


def test_profile_append_is_after_the_tfbench_cell_in_the_job_chain():
    rows = [_row("a1", "yunshu")]
    prof = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    job = next(j for j in prof.jobs if j.label.endswith("-prof"))
    script = job.cmd[-1]
    assert script.index("fullcmp_profile.py") > script.index("tfbench.py")
    assert " && " in script


def test_yunshu_round_profile_lines_map_to_stages_and_missing_stays_missing():
    log = "\n".join(
        [
            "INFO DFlash round profile: 10 rounds, 7.5 nodes/round, draft 1.20 ms, verify 40.00 ms, commit+host 3.00 ms",
            "INFO DFlash host between rounds: consumer (yield) 0.50 ms/round; commit parts {'m_a': 0.1}",
            "INFO Tree verify stage profile (10 rounds, ms/round): {'attn': 12.0, 'mlp': 20.0}",
        ]
    )
    stages = fp.parse_yunshu_stages(log)
    assert stages["decode_round"]["drafter"] == 1.2
    assert stages["decode_round"]["commit"] == 3.0
    assert stages["decode_round"]["verify_total"] == 40.0
    cov = fa.check_profile("yunshu", {"stages": stages, "competitor_stats": None})
    assert cov["stages.decode_round.drafter"] == "present"
    # plan / copy lookup / attention split are not in the engine log: MISSING, never invented
    assert cov["stages.decode_round.plan"] == "MISSING"
    assert "MISSING" in cov.values()


def test_competitor_profile_keeps_stats_lines_as_is():
    log = "random line\nTF stats: decode 80.1 tok/s prefill 1200 tok/s\nother\n"
    lines = fp.competitor_stat_lines(log)
    assert lines == ["TF stats: decode 80.1 tok/s prefill 1200 tok/s"]
    cov = fa.check_profile("tf-1.2.1", {"competitor_stats": lines})
    assert cov == {"competitor_stats": "present"}
    assert fa.check_profile("tf-1.2.1", {"competitor_stats": []}) == {
        "competitor_stats": "MISSING"
    }


def test_append_profile_row_reads_engine_log_path_from_the_cell(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "DFlash round profile: 2 rounds, 4.0 nodes/round, draft 1.00 ms, verify 9.00 ms, commit+host 1.00 ms\n"
    )
    cell = tmp_path / "a1-prof.jsonl"
    cell.write_text(
        json.dumps({"part": "header", "engine": "yunshu"})
        + "\n"
        + json.dumps({"part": "decode", "attr": {"engine_log_path": str(log)}})
        + "\n"
    )
    row = fp.append_profile_row(cell, "yunshu")
    assert row["part"] == "profile"
    assert row["stages"]["decode_round"]["drafter"] == 1.0
    back = [json.loads(x) for x in cell.read_text().splitlines()]
    assert back[-1]["part"] == "profile"


def test_profile_row_without_log_is_recorded_as_error_not_skipped(tmp_path):
    cell = tmp_path / "x-prof.jsonl"
    cell.write_text(json.dumps({"part": "header"}) + "\n")
    row = fp.append_profile_row(cell, "tf-1.2.1")
    assert row["error"] == "no engine_log_path in cell file"
    assert row.get("competitor_stats") is None
