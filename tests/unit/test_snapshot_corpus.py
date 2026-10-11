import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import bench_snapshot as bs  # noqa: E402
import snapshot_corpus as sc  # noqa: E402


def test_summary_is_mean_median_and_paired_ratio_against_the_reference():
    per = {
        "yunshu-new": {"p0": 100.0, "p1": 50.0, "p2": 80.0},
        "tf-new": {"p0": 110.0, "p1": 50.0, "p2": 88.0},
    }
    out = sc.summarize(per)
    assert (
        out["yunshu-new"]["mean"] == (100 + 50 + 80) / 3
        and out["yunshu-new"]["median"] == 80.0
    )
    assert out["tf-new"]["paired_n"] == 3 and out["tf-new"]["wins_vs_ref"] == 3
    assert abs(out["tf-new"]["ratio_vs_ref"] - (1.1 + 1.0 + 1.1) / 3) < 1e-12


def test_variants_have_distinct_names_and_files_cover_both_kinds():
    files = sc.corpus_files(Path("/x"), 8192)
    assert len(files) == 2 * sc.CORPUS_N and len({f.name for f in files}) == len(files)
    assert (
        files[0].name == "prose-8192-v00.txt" and files[-1].name == "code-8192-v09.txt"
    )


def test_corpus_jobs_are_opt_in_and_run_n_prompts_per_kind():
    off = bs.plan_cells(["yunshu-new"], 1, Path("/o"), {})
    on = bs.plan_cells(["yunshu-new"], 1, Path("/o"), {}, corpus=True)
    assert not any(j.part == "corpus" for j in off)
    jobs = [j for j in on if j.part == "corpus"]
    assert [j.group for j in jobs] == ["cp1k", "cp8k", "cp32k"]
    argv = jobs[0].argv
    files = argv[argv.index("--prompt-files") + 1].split(",")
    assert len(files) == 2 * sc.CORPUS_N and argv[argv.index("--part") + 1] == "corpus"
    assert bs.expected_rows(jobs[0]) == {"corpus": 2 * sc.CORPUS_N}


def test_short_reply_fails_validation():
    job = next(
        j
        for j in bs.plan_cells(["tf-new"], 1, Path("/o"), {}, corpus=True)
        if j.part == "corpus"
    )
    names = [f.stem for f in sc.corpus_files(bs.CORPUS_DIR, job.ctxs[0], job.kinds)]
    base = {
        "engine": "tf-new",
        "part": "corpus",
        "dec_tps": 70.0,
        "ct": bs.CORPUS_TOKENS,
        "pt": 1024,
        "short": False,
    }
    rows = [{**base, "prompt": n} for n in names]
    problems = bs.validate_rows(job, rows)
    assert not any("short" in p or "corpus prompts" in p for p in problems)
    rows[3] = {**rows[3], "short": True, "ct": 900}
    assert any("short reply" in p for p in bs.validate_rows(job, rows))
