"""--profile-rep: layer-2 profile jobs (label ...-prof) that never feed table numbers.

CPU only: the job plan, the profile env, the engine-log parser and the profile row are checked with
fakes. No server, no GPU, no gpuq.
"""

from __future__ import annotations

import json
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_attribution as fa  # noqa: E402
import fullcmp_profile as fp  # noqa: E402
import fullcmp_run as fr  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fullcmp_pin_io import stub_pin_io  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_io(monkeypatch):
    stub_pin_io(monkeypatch)

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
    text = job.script
    assert "YUNSHU_DEBUG_TREE_PROFILE=1" in text
    assert "/profile/yunshu/" in text
    # never the table root's <engine>/<cell>-rN.jsonl layout
    assert f"{fr.RAW_ROOT}/yunshu/a1-r" not in text
    assert "-r0.jsonl" not in text


def test_competitor_profile_job_has_no_yunshu_env():
    rows = [_row("b1", "tf-1.2.1")]
    prof = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    job = next(j for j in prof.jobs if j.label.endswith("-prof"))
    assert "YUNSHU_DEBUG_TREE_PROFILE=1" not in job.script
    assert fp.profile_env("tf-1.2.1") == []


def test_profile_append_is_after_the_tfbench_cell_in_the_job_chain():
    rows = [_row("a1", "yunshu")]
    prof = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    job = next(j for j in prof.jobs if j.label.endswith("-prof"))
    script = job.script
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


def test_audit_manifest_reps_follow_the_rows_not_a_constant():
    # Smoke rows run one rep; the manifest must expect one rep, not a hard-coded two.
    rows = [dict(_row("s1", "yunshu"), reps=1)]
    man = fr.audit_manifest(rows, ["B1"], PIN, ["yunshu"])
    assert man["reps"] == 1
    rows = [dict(_row("s1", "yunshu"), reps=3)]
    assert fr.audit_manifest(rows, ["B1"], PIN, ["yunshu"])["reps"] == 3


def test_own_jobs_are_not_foreign_when_gpuq_prints_the_job_id():
    # gpuq status prints the job id (<seq>-<label>), not the label: our own jobs must not refuse us.
    text = (
        "daemon: up\n"
        "wait-cpu  5/20s  device=m5  1103-113543-00-fullcmp-smoke2-dryrun\n"
        "running   device=m5  1103-113600-00-fullcmp-B1-yunshu-q27-r0\n"
        "pending   device=m5  1103-113700-00-someone-else-job\n"
    )
    assert fr.foreign_jobs(text) == [("pending", "1103-113700-00-someone-else-job")]


def test_status_job_id_is_found_when_the_line_has_trailing_text():
    # gpuq status appends "(quiet window satisfied)": the id is not the last column.
    text = (
        "running  device=m5  54s  p1  1103-114230-00-observe-ab-code-8192-p0  "
        "peak=32.5G/24G  held=21/300s (quiet window satisfied)\n"
    )
    assert fr.parse_status(text) == [
        ("running", "1103-114230-00-observe-ab-code-8192-p0")
    ]
    assert fr.foreign_jobs(text) == [
        ("running", "1103-114230-00-observe-ab-code-8192-p0")
    ]


def test_job_shell_script_does_not_start_with_a_path():
    # gpuq preflight treats any argument starting with "/" as an input path that must exist; the
    # job's sh -c script must not look like one (it begins with the quiet gate's interpreter path).
    rows = [_row("a1", "yunshu")]
    plan = fr.plan_batch(rows, "B1", PIN, profile_rep=True)
    for job in plan.jobs:
        script = job.script
        assert not script.startswith("/"), script[:60]


def test_real_wire_1107y_conc_profile_cell_finds_its_log_from_the_session_header():
    """Failing-before (wire-1107y): the conc cell's rows carry no engine_log_path (only the session header does);
    the profile stage must read it there, not fail with 'no engine_log_path'."""
    real = Path("/Volumes/Micron/yunshu-fullcmp/wire-1107y/profile/yunshu/"
                "SMOKE.smoke-conc-distinct.yunshu.1K.c2.code.o64.single.th-off.none.greedy.v0-prof.jsonl")
    if not real.exists():
        pytest.skip("no real wire-1107y conc profile cell on disk")
    assert fp._engine_log_path(real) is not None


def test_concurrent_cell_without_any_log_path_is_a_labelled_result_not_an_error(tmp_path):
    cell = tmp_path / "c-prof.jsonl"
    cell.write_text(json.dumps({"part": "conc", "n": 2, "requests": []}) + "\n")
    row = fp.append_profile_row(cell, "yunshu")
    assert "error" not in row
    assert row["stages_unavailable"].startswith("stages unavailable for concurrent path")


def test_yunshu_profile_row_names_every_absent_layer2_field_as_unavailable(tmp_path):
    """Failing-before (wire-1108 audit): the dev0/pb2 profile log carries drafter/verify_total/commit/host_yield
    only, so the row lacked stages.decode_round.plan/copy_lookup/..., prefill.*, ttft.*, host_sync_count and
    op_census and the guard reported them MISSING. The row now carries an explicit reason per absent field;
    the guard reports 'unavailable (reason)', and present fields stay present."""
    log = tmp_path / "server.log"
    log.write_text(
        "DFlash round profile: 114 rounds, 14.2 nodes/round, draft 2.27 ms, verify 45.49 ms, commit+host 0.12 ms\n"
        "DFlash host between rounds: consumer (yield) 1.14 ms/round\n"
    )
    cell = tmp_path / "a1-prof.jsonl"
    cell.write_text(json.dumps({"part": "session", "engine_log_path": str(log)}) + "\n")
    row = fp.append_profile_row(cell, "yunshu-pb2")
    un = row["layer2_unavailable"]
    for field in (
        "stages.decode_round.plan",
        "stages.decode_round.copy_lookup",
        "stages.decode_round.verify_weights",
        "stages.decode_round.verify_attention",
        "stages.prefill.chunk_compute",
        "stages.ttft.lookup",
        "host_sync_count",
        "op_census",
    ):
        assert un[field].startswith("unavailable:"), (field, un.get(field))
    assert "stages.decode_round.drafter" not in un and "stages.decode_round.commit" not in un
    cov = fa.check_profile("yunshu-pb2", row)
    assert fa.missing(cov) == [], fa.missing(cov)
    assert cov["stages.decode_round.drafter"] == "present"
    assert cov["host_sync_count"].startswith("unavailable (")
    # a field that is neither present nor reasoned stays MISSING (fail closed)
    bare = {k: v for k, v in row.items() if k != "layer2_unavailable"}
    assert "host_sync_count" in fa.missing(fa.check_profile("yunshu-pb2", bare))


def test_audit_gives_a_legacy_profile_row_the_same_static_reasons():
    """The wire-1108 profile rows were written before the reasons existed: the audit applies them."""
    import fullcmp_audit as fau

    row = {"part": "profile", "engine": "yunshu", "stages": {"decode_round": {"drafter": 2.27, "commit": 0.12}}}
    assert "host_sync_count" in fp.unavailable_layer2(row)
    assert hasattr(fau, "check_attribution_fields")


def test_phase_b_round_driver_lines_map_to_stages_and_drafter_commit_are_unavailable(tmp_path):
    """Failing-before (wire-1108 yunshu-pb2): the Phase B round driver prints no 'DFlash round profile' line,
    so the profile row had stages {} and the guard failed on drafter and commit. Its own lines (window ms/round,
    'round driver idle: N decode steps, X ms in a step, Y ms between steps') now map to stages, and the two
    per-stage fields it cannot emit carry explicit reasons."""
    log = tmp_path / "server.log"
    log.write_text(
        "x round_driver.dflash: Round driver DFlash window of 100 rounds (total 100): rows/round 1.00, "
        "48.7 ms/round, nodes/row 14.30, tokens/row-round 4.60\n"
        "x round_driver.driver: round driver idle: 100 decode steps, 48.0 ms in a step, 0.4 ms between steps\n"
        "x round_driver.driver: round driver idle: 100 decode steps, 50.0 ms in a step, 0.2 ms between steps\n"
    )
    cell = tmp_path / "p-prof.jsonl"
    cell.write_text(json.dumps({"part": "session", "engine_log_path": str(log)}) + "\n")
    row = fp.append_profile_row(cell, "yunshu-pb2")
    node = row["stages"]["decode_round"]
    assert node["rounds"] == 200 and node["step_ms"] == pytest.approx(49.0)
    assert node["host_between_steps_ms"] == pytest.approx(0.3) and node["window_ms_per_round"] == 48.7
    assert row["layer2_unavailable"]["stages.decode_round.drafter"].startswith("unavailable:")
    assert fa.missing(fa.check_profile("yunshu-pb2", row)) == []


def test_a_dev0_row_without_the_round_line_still_fails_on_drafter(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("nothing profiled here\n")
    cell = tmp_path / "d-prof.jsonl"
    cell.write_text(json.dumps({"part": "session", "engine_log_path": str(log)}) + "\n")
    row = fp.append_profile_row(cell, "yunshu")
    assert "stages.decode_round.drafter" in fa.missing(fa.check_profile("yunshu", row))


def test_legacy_pb2_profile_row_is_recomputed_from_its_engine_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("round driver idle: 10 decode steps, 40.0 ms in a step, 1.0 ms between steps\n")
    legacy = {"part": "profile", "engine": "yunshu-pb2", "engine_log_path": str(log), "stages": {}}
    row = fp.refresh_legacy_row(legacy)
    assert row["stages"]["decode_round"]["step_ms"] == 40.0
    assert fa.missing(fa.check_profile("yunshu-pb2", row)) == []
