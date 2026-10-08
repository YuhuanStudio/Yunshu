"""CPU tests for the cross-engine snapshot planner / validator (scripts/research/bench_snapshot.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import bench_engines as be  # noqa: E402
import bench_snapshot as bs  # noqa: E402

TREES = {"yunshu-new": "/t/new/python", "yunshu-base": "/t/base/python"}
OUT = Path("/nonexistent/bench014")


def cells(reps=3):
    return bs.plan_cells(list(bs.ENGINE_ORDER), reps, OUT, TREES)


def make_rows(job, engaged=None, **drop):
    """A trustworthy output for `job` (what tfbench.py would write)."""
    e = be.ENGINES[job.engine]
    mode = engaged or e.expected_mode
    meta = be.meta_row(
        job.engine,
        version="v",
        git_sha="abc" if e.kind == "yunshu" else None,
        engaged=mode,
        flags={},
    )
    rows = [{**meta, "part": "session", "engaged_spec_mode": mode}]
    combos = [
        (c, k, ph)
        for c in (job.ctxs or (1024,))
        for k in (job.kinds or ("prose",))
        for ph in bs.PHASES
    ]
    for part, n in bs.expected_rows(job).items():
        for i in range(n):
            c, k, ph = combos[i % len(combos)]
            rows.append(
                {
                    **meta,
                    "part": part,
                    "ttft_s": 1.0,
                    "dec_tps": 50.0,
                    "ct": bs.DECODE_TOKENS,
                    "phase": ph,
                    "ctx": c,
                    "content_tokens": c,
                    "kind": k,
                    "correct": True,
                    "n": 2,
                    "agg_tps": 90.0,
                }
            )
    rows.append({**meta, "part": "memory", "peak_gib": 30.0, "idle_gib": 20.0})
    rows.append({**meta, "part": "part_done", "complete": True})
    for key in drop.get("drop", ()):
        rows = [r for r in rows if r.get("part") != key]
    return rows


def test_plan_counts_and_labels():
    jobs = cells()
    # per engine and rep: 6 decode groups + 1 concurrency; needles (3 groups) only in rep 0
    assert len(jobs) == 7 * (3 * 8 + 3)
    labels = [j.label for j in jobs]
    assert len(set(labels)) == len(labels)
    assert all(label.startswith("snapshot014-") for label in labels)
    assert all(j.out != Path() for j in jobs)


def test_engines_interleave_inside_each_cell_and_reps_are_outermost():
    jobs = cells()
    first = jobs[: len(bs.ENGINE_ORDER)]
    assert [j.engine for j in first] == list(bs.ENGINE_ORDER)
    assert {j.group for j in first} == {"d1k"} and {j.rep for j in first} == {0}
    reps = [j.rep for j in jobs]
    assert reps == sorted(reps)


def test_cell_coverage_matches_the_spec():
    jobs = [j for j in cells() if j.engine == "yunshu-new" and j.rep == 0]
    decode = [
        (c, k) for j in jobs if j.part == "decode" for c in j.ctxs for k in j.kinds
    ]
    assert sorted(decode) == sorted((c, k) for c in bs.CTXS for k in bs.KINDS)
    assert {c for j in jobs if j.part == "needle" for c in j.ctxs} == {
        32768,
        65536,
        131072,
    }
    conc = [j for j in jobs if j.part == "conc"]
    assert len(conc) == 1 and "--conc-ns" in conc[0].argv and "2,4" in conc[0].argv


def test_decode_job_requests_2048_token_replies_and_never_sets_a_draft_override():
    j = next(j for j in cells() if j.engine == "yunshu-new" and j.group == "d32k")
    a = j.argv
    assert a[a.index("--decode-tokens") + 1] == "2048" and "--long-ask" in a
    assert "YUNSHU_VLM_DRAFT" not in j.env and "YUNSHU_VLM_DRAFT" not in " ".join(a)
    assert j.env["TFB_YUNSHU_SRC"] == "/t/new/python"
    base = bs.plan_cells(["yunshu-base"], 1, OUT, TREES)[0]
    assert base.env["TFB_YUNSHU_SRC"] == "/t/base/python"


def test_timeouts_cover_the_estimate_and_stall_covers_the_cold_prefill():
    for j in cells():
        assert j.timeout_min >= j.est_min * 1.5
        assert j.mem_gb >= 40
    long = next(
        j for j in cells() if j.engine == "llamacpp" and j.group == "d128k-prose"
    )
    short = next(j for j in cells() if j.engine == "yunshu-new" and j.group == "d1k")
    assert long.timeout_min > short.timeout_min
    assert long.stall_min >= bs.prefill_seconds("llamacpp", 131072) * 1.5 / 60


def test_submit_args_priority_label_quiet_and_declared_output():
    j = cells()[0]
    args = bs.submit_args(j)
    assert "--quiet" in args and "--priority=0" in args and "--expect-complete" in args
    assert args[args.index("--label") + 1] == j.label
    assert args[args.index("--out") + 1] == str(j.out)
    assert args[args.index("--mem-gb") + 1] == str(j.mem_gb)
    assert args[args.index("--timeout") + 1] == str(j.timeout_min)


def test_yunshu_launch_uses_real_discovery_not_an_override(tmp_path):
    env_in = {"YUNSHU_VLM_DRAFT": "mtp", "PATH": "/bin", "OPENAI_API_KEY": "x"}
    la = be.build_launch("yunshu-new", 18990, tmp_path, env_in)
    assert "YUNSHU_VLM_DRAFT" not in la.env and "OPENAI_API_KEY" not in la.env
    assert la.env["HOME"] == str(tmp_path)
    # the drafter sits next to the checkpoint in the user models dir the engine scans
    links = {Path(k).relative_to(tmp_path).as_posix(): v for k, v in la.links.items()}
    assert links[".yunshu/models/incoai/Qwen3.8-27B-DFlash2"] == be.DRAFTER
    assert links[".yunshu/models/Jundot/Qwen3.8-27B-oQ4e-mtp"] == be.CHECKPOINT
    assert la.cmd[1:3] == ["serve", "-m"] and la.cmd[3].startswith(str(tmp_path))


def test_hidden_drafter_is_caught_by_the_fail_closed_check():
    """The 2026-10-02 failure: an isolated HOME hid the drafter and the run silently used MTP."""
    log = "x\nVLM batch runner: apc=True draft=mtp block=6 verify_kernels=True\n"
    assert be.detect_mode("yunshu-new", log) == "mtp"
    with pytest.raises(
        RuntimeError, match="expected spec mode 'dflash', engaged 'mtp'"
    ):
        be.check_engaged("yunshu-new", be.detect_mode("yunshu-new", log))
    with pytest.raises(RuntimeError):
        be.check_engaged("yunshu-new", None)
    ok = "VLM batch runner: apc=True draft=dflash block=8 verify_kernels=True"
    be.check_engaged("yunshu-new", be.detect_mode("yunshu-new", ok))


@pytest.mark.parametrize(
    ("engine", "log", "probe", "mode"),
    [
        (
            "tf-new",
            "[tensorfold] drafter /m/Qwen3.8-27B-DFlash2 block=8 bits=4",
            None,
            "dflash",
        ),
        (
            "omlx",
            "INFO DFlash enabled for Qwen3.8-27B-oQ4e-mtp, draft=/m/d",
            None,
            "dflash",
        ),
        ("omlx", "INFO loaded model", None, None),
        ("mtplx", "│       Mode  stable MTP                            │", None, "mtp"),
        ("llamacpp", "common_speculative: types = draft-mtp", None, "mtp"),
        ("splash", "Ready", '{"draft": {"name": "dflash2"}}', "dflash"),
        ("splash", "Ready", "{}", None),
        ("mlxlm", "Starting httpd at 127.0.0.1 on port 18990", None, "ar"),
    ],
)
def test_detect_mode_per_engine(engine, log, probe, mode):
    assert be.detect_mode(engine, log, probe) == mode


def test_other_engine_launches(tmp_path):
    om = be.build_launch("omlx", 18991, tmp_path, {})
    settings = json.loads(next(iter(om.files.values())))
    assert settings["models"]["Qwen3.8-27B-oQ4e-mtp"]["dflash_enabled"] is True
    assert "--base-path" in om.cmd and str(tmp_path / "omlx-base") in om.cmd
    assert not any("/.omlx" in c for c in om.cmd)  # never the user's app config
    ll = be.build_launch("llamacpp", 18992, tmp_path, {}, ctx_tokens=135168, parallel=4)
    assert (
        ll.cmd[ll.cmd.index("-c") + 1] == "135168"
        and ll.cmd[ll.cmd.index("-np") + 1] == "4"
    )
    assert "draft-mtp" in ll.cmd and be.GGUF in ll.cmd
    mt = be.build_launch("mtplx", 18993, tmp_path, {})
    assert mt.cmd[0].endswith("mtplx/.venv/bin/mtplx") and "turbo" in mt.cmd
    sp = be.build_launch("splash", 18994, tmp_path, {})
    assert "--port" in sp.cmd and "18994" in sp.cmd and sp.probe == "/status"
    assert all(c.id != "" for c in be.ENGINES.values())
    assert be.ENGINES["llamacpp"].weights == be.ENGINES["splash"].weights == "different"


def test_all_ports_in_range_are_not_hardcoded():
    for e in be.ENGINES:
        if be.is_new_engine(e):
            la = be.build_launch(e, 18997, "/h", {})
            assert "8000" not in la.cmd


def test_validate_accepts_a_complete_job_for_every_engine():
    for j in bs.plan_cells(list(bs.ENGINE_ORDER), 1, OUT, TREES):
        assert bs.validate_rows(j, make_rows(j)) == [], j.name
    for p in bs.plan_pilots(list(bs.ENGINE_ORDER), OUT, TREES):
        assert bs.validate_rows(p, make_rows(p)) == [], p.name


def test_validate_fails_closed():
    j = next(j for j in cells() if j.engine == "yunshu-new" and j.group == "d32k")
    good = make_rows(j)
    assert bs.validate_rows(j, []) == ["no output"]
    # the drafter was hidden: MTP engaged although DFlash was expected
    bad = make_rows(j, engaged="mtp")
    assert any("engaged spec mode" in p for p in bs.validate_rows(j, bad))
    assert any("incomplete" in p for p in bs.validate_rows(j, good[:-1]))
    assert any(
        "no memory row" in p
        for p in bs.validate_rows(j, make_rows(j, drop=("memory",)))
    )
    assert any(
        "decode: 5 rows" in p
        for p in bs.validate_rows(j, [r for i, r in enumerate(good) if i != 3])
    )
    stripped = [{k: v for k, v in r.items() if k != "flags"} for r in good]
    assert any("lacks" in p for p in bs.validate_rows(j, stripped))
    short = [dict(r, ct=100) if r.get("part") == "decode" else r for r in good]
    assert any("ct=100" in p for p in bs.validate_rows(j, short))
    nosha = [dict(r, git_sha=None) for r in good]
    assert any("git sha" in p for p in bs.validate_rows(j, nosha))


def test_resumable_submit_skips_complete_and_queued_and_retries_failures(
    tmp_path, monkeypatch
):
    jobs = bs.plan_cells(["mlxlm"], 1, tmp_path, TREES)[:4]
    # job 0 complete on disk
    jobs[0].out.parent.mkdir(parents=True, exist_ok=True)
    jobs[0].out.write_text("\n".join(json.dumps(r) for r in make_rows(jobs[0])) + "\n")
    # job 1 is queued according to the state file
    bs.save_state(tmp_path, {jobs[1].name: {"id": "J1", "attempts": 1}})
    # job 2 failed once, left a partial file
    jobs[2].out.write_text('{"part": "session"}\n')
    bs.save_state(
        tmp_path, {**bs.load_state(tmp_path), jobs[2].name: {"id": "J2", "attempts": 1}}
    )
    submitted = []

    def fake_run(cmd, **kw):
        if cmd[1] == "wait":
            return subprocess.CompletedProcess(cmd, 2 if cmd[-1] == "J1" else 1, "", "")
        submitted.append(cmd[cmd.index("--label") + 1])
        return subprocess.CompletedProcess(cmd, 0, f"id-{len(submitted)}\n", "")

    monkeypatch.setattr(bs.subprocess, "run", fake_run)
    counts = bs.submit_jobs(jobs, tmp_path)
    assert counts == {"complete": 1, "queued": 1, "submitted": 2, "gave_up": 0}
    assert submitted == [jobs[2].label, jobs[3].label]
    assert (
        jobs[2].out.with_suffix(".failed1.jsonl").exists()
    )  # the failed attempt is kept apart
    # a second failure exhausts the attempts
    bs.save_state(
        tmp_path, {**bs.load_state(tmp_path), jobs[3].name: {"id": "J3", "attempts": 2}}
    )
    assert bs.submit_jobs(jobs[3:4], tmp_path)["gave_up"] == 1


def test_cells_are_refused_until_a_pilot_validated(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        bs,
        "resolve_trees",
        lambda a, dry: (TREES, {"yunshu-new": "a" * 40, "yunshu-base": "b" * 40}),
    )
    with pytest.raises(SystemExit, match="pilot not validated"):
        bs.main(
            ["submit", "--stage", "cells", "--engines", "mlxlm", "--out", str(tmp_path)]
        )


def test_aggregate_reports_median_and_range(tmp_path):
    for rep, ttft in enumerate((1.0, 3.0, 2.0)):
        j = next(
            j
            for j in bs.plan_cells(["mlxlm"], 3, tmp_path, TREES)
            if j.group == "d1k" and j.rep == rep
        )
        rows = make_rows(j)
        for r in rows:
            if r.get("part") == "decode":
                r["ttft_s"] = ttft
        j.out.parent.mkdir(parents=True, exist_ok=True)
        j.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    md = bs.render_markdown(bs.collect(tmp_path, ["mlxlm"]), ["mlxlm"])
    assert "2.00 (1.00-3.00) n=3" in md
    assert "Cold TTFT" in md and "memory" in md


def test_dry_run_plans_every_job_without_submitting(capsys, monkeypatch):
    real = subprocess.run

    def refuse(*a, **k):
        if a and a[0][:1] in ([bs.GPUQ], [bs.AGENTBENCH]):
            raise AssertionError("dry run must not call gpuq")
        return real(*a, **k)

    monkeypatch.setattr(bs.subprocess, "run", refuse)
    assert bs.main(["plan", "--dry-run"]) == 0
    out = capsys.readouterr().out
    print(out)
    assert (
        "snapshot014-yunshu-new-d128k-prose-r2" in out
        and "snapshot014-llamacpp-pilot" in out
    )
    assert "total" in out and "GPU h" in out


def test_tfbench_rows_carry_the_meta_fields(tmp_path, monkeypatch):
    import io

    import tfbench

    monkeypatch.setattr(tfbench, "was_contended", lambda: False)
    meta = be.meta_row(
        "yunshu-new",
        version="0.1.4",
        git_sha="deadbeef",
        engaged="dflash",
        flags={"a": 1},
    )
    monkeypatch.setattr(tfbench, "META", meta)
    buf = io.StringIO()
    tfbench.emit(buf, part="decode", ctx=1024)
    row = json.loads(buf.getvalue())
    for key in bs.REQUIRED_META:
        assert key in row
    assert (
        row["spec_mode"] == "dflash"
        and row["drafter"] == be.DRAFTER
        and row["version"] == "0.1.4"
    )


def test_exact_prompt_budget_and_instruction():
    from snapshot_prompts import assert_prompt, exact_prompt
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(
        WordLevel({"[UNK]": 0, "source": 1, "write": 2, "a": 3}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = Whitespace()
    result = exact_prompt("source " * 140000, 131072, " write", tok)
    assert "write" in result
    assert assert_prompt(result, 131072, tok) == 131072
    with pytest.raises(AssertionError, match="expected=131072"):
        assert_prompt("source", 131072, tok)


def test_snapshot_pilot_is_not_a_timing_job():
    job = bs.plan_pilots(["mlxlm"], OUT, TREES)[0]
    assert "--quiet" not in bs.submit_args(job)
    assert job.env["GPUQ_OWNER"] == "snapshot014"
    assert job.env["TFB_EXACT_PROMPTS"] == "1"


def test_aggregate_rejects_completed_but_wrong_token_budget(tmp_path):
    job = bs.plan_cells(["mlxlm"], 1, tmp_path, TREES)[0]
    job.out.parent.mkdir(parents=True)
    rows = make_rows(job)
    rows[1]["content_tokens"] = 1000
    job.out.write_text("\n".join(json.dumps(r) for r in rows))
    assert bs.collect(tmp_path, ["mlxlm"]) == {}


def test_aggregate_reads_only_promoted_yv_evidence(tmp_path):
    job = bs.plan_cells(["mlxlm"], 1, tmp_path, TREES)[0]
    directory = tmp_path / "cells"
    directory.mkdir()
    (directory / f"snapshot.{job.name}.a0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in make_rows(job))
    )
    assert bs.collect(tmp_path, ["mlxlm"], tmp_path) == {}
    (directory / f"snapshot.{job.name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in make_rows(job))
    )
    assert bs.collect(tmp_path, ["mlxlm"], tmp_path)


def test_generation_manifest_covers_exact_ladder_and_concurrent_cohorts(
    tmp_path, monkeypatch
):
    import gen_snapshot_prompts as gen

    source, destination = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    for ctx in bs.CTXS:
        for kind in bs.KINDS:
            (source / f"{kind}-{ctx}.txt").write_text("body\n\n---\nask")
    (tmp_path / "tokenizer.json").write_text("fixture")
    monkeypatch.setattr(gen, "MODEL", tmp_path)
    monkeypatch.setattr(
        gen,
        "exact_prompt",
        lambda source, target, suffix: f"{target}\n{source}{suffix}",
    )
    monkeypatch.setattr(
        gen, "assert_prompt", lambda text, target: int(text.splitlines()[0])
    )
    manifest = gen.generate(source, destination)
    assert manifest["complete"] is True
    assert len(manifest["prompts"]) == 10 + 2 * (2 + 4) * 2
    assert (destination / "prose-131072.txt").read_text().startswith("131072\n")
    assert (destination / "code-131072.txt").read_text().startswith("131072\n")
    assert (
        "cohort n=4 trial=1 request=3"
        in (destination / "conc-4-1-code-3.txt").read_text()
    )
