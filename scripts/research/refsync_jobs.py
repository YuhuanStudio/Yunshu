"""refsync Stage 2: the latest Mac engines against Yunshu main, decode corpus cells through gpuq.

Reuses the snapshot harness (tfbench.py corpus part, bench_engines registry, bench_snapshot job plumbing):
every cell is one server session that cold-decodes N=10 distinct exactly-tokenized prompts (greedy,
1024 tokens) and reports the paired per-prompt ratio against Yunshu main.

  refsync_jobs.py plan   [--set 27b|fn|all] [--engines a,b] [--reps N]
  refsync_jobs.py pilot  [--set ...]        one smoke session per engine (engaged-mode evidence); read logs first
  refsync_jobs.py cells  [--set ...]        submit the corpus cells (needs a validated pilot per engine)
  refsync_jobs.py status | aggregate [--md FILE]

Sets: 27b = Qwen3.8-27B cells 1K prose, 1K code, 8K prose; fn = Qwen3.8-Flash-Next cells 1K prose, 8K prose.
Labels start with ``refsync-``; priority 0 (the harness default).  CPU only until `pilot`/`cells`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_engines as be  # noqa: E402
import bench_snapshot as bs  # noqa: E402

REPO = HERE.parents[1]
BUILD = Path("/Volumes/P5Plus/yunshu-build/refsync")
OUT_DEFAULT = Path(
    "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/refsync/runs"
)
MAIN_VENV_YUNSHU = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"

SETS = {
    "27b": {
        "engines": [
            "yunshu-main",
            "yunshu-tf4",
            "tf-native",
            "omlx-latest",
            "mtplx-latest",
            "llamacpp-mtp",
            "llamacpp-dflash",
            "splash-latest",
        ],
        "reference": "yunshu-main",
        # (group, ctx, kinds)
        "groups": [
            ("c1p", 1024, ("prose",)),
            ("c1c", 1024, ("code",)),
            ("c8p", 8192, ("prose",)),
        ],
    },
    "long27": {
        "part": "decode",
        "engines": [
            "yunshu-main",
            "yunshu-tf4",
            "tf-native",
            "omlx-latest",
            "llamacpp-mtp",
            "llamacpp-dflash",
            "splash-latest",
        ],
        "reference": "yunshu-main",
        "groups": [
            ("L32", 32768, ("code",)),
            ("L64", 65536, ("code",)),
            ("L128", 131072, ("code",)),
        ],
    },
    "longfn": {
        "part": "decode",
        "engines": ["fn-yunshu-main", "fn-omlx-latest", "fn-llamacpp-mtp"],
        "reference": "fn-yunshu-main",
        "groups": [("F32", 32768, ("code",)), ("F64", 65536, ("code",))],
    },
    "fn": {
        "engines": ["fn-yunshu-main", "fn-omlx-latest", "fn-llamacpp-mtp"],
        "reference": "fn-yunshu-main",
        "groups": [("f1p", 1024, ("prose",)), ("f8p", 8192, ("prose",))],
    },
}

STARTUP_S = {
    "yunshu-main": 90,
    "yunshu-tf4": 90,
    "tf-native": 60,
    "omlx-latest": 90,
    "mtplx-latest": 90,
    "llamacpp-mtp": 60,
    "llamacpp-dflash": 60,
    "splash-latest": 60,
    "fn-yunshu-main": 300,
    "fn-omlx-latest": 420,
    "fn-llamacpp-mtp": 420,
}
FN_MEM_GB = 100


def configure() -> None:
    """Point the shared harness at the refsync build dir and label prefix (module-level state)."""
    bs.LABEL_PREFIX = "refsync"
    bs.BUILD = BUILD
    bs._STARTUP_S.update(STARTUP_S)  # noqa: SLF001


def mem_gb(engine: str, ctx: int) -> int:
    if engine.startswith("fn-"):
        return FN_MEM_GB
    base = bs.job_mem_gb("omlx" if engine == "omlx-latest" else "yunshu-new", (ctx,))
    if engine in ("mtplx-latest", "splash-latest"):
        base += 8
    return base


def job_env(engine: str) -> dict:
    env = bs.job_env(engine, {})
    env.update(
        TFB_OUT=str(BUILD / "work"),
        GPUQ_OWNER="refsync",
        TFB_YUNSHU_BIN=MAIN_VENV_YUNSHU,
        TFB_YUNSHU_SRC=str(REPO / "python"),
    )
    return env


def make_job(
    engine: str,
    group: str,
    ctx: int,
    kinds,
    rep: int,
    outdir: Path,
    part: str = "corpus",
) -> bs.Job:
    est = bs.estimate_minutes(engine, part, (ctx,), kinds)
    job = bs.Job(
        f"{engine}-{group}-r{rep}",
        engine,
        part,
        group,
        rep,
        "cell",
        (ctx,),
        tuple(kinds),
        est,
        mem_gb(engine, ctx),
        bs.timeout_minutes(est),
        bs.stall_minutes(engine, (ctx,)),
        outdir / engine / f"{group}-r{rep}.jsonl",
    )
    job.env = job_env(engine)
    job.argv = bs.tfbench_argv(job, job.out)
    return job


def pilots(engines, outdir: Path) -> list[bs.Job]:
    jobs = bs.plan_pilots(engines, outdir, {})
    for job in jobs:
        job.env = job_env(job.engine)
        job.mem_gb = mem_gb(job.engine, 1024)
    return jobs


def cells(set_name: str, engines, reps: int, outdir: Path) -> list[bs.Job]:
    spec = SETS[set_name]
    jobs = []
    for rep in range(reps):
        for group, ctx, kinds in spec["groups"]:
            for e in spec["engines"]:
                if e in engines:
                    jobs.append(
                        make_job(
                            e,
                            group,
                            ctx,
                            kinds,
                            rep,
                            outdir,
                            spec.get("part", "corpus"),
                        )
                    )
    return jobs


def collect_long(outdir: Path, set_name: str) -> dict:
    """(ctx, engine) -> {'status', 'cold_ttft', 'warm_ttft', 'turn2_ttft', 'dec_tps' (cold), 'cached'} per valid job;
    an invalid or missing job is reported with its reason, never dropped."""
    spec = SETS[set_name]
    out: dict = {}
    for group, ctx, kinds in spec["groups"]:
        for e in spec["engines"]:
            job = make_job(e, group, ctx, kinds, 0, outdir, "decode")
            rows = bs.read_rows(job.out)
            problems = bs.validate_rows(job, rows)
            if problems:
                out[(ctx, e)] = {"status": "; ".join(problems[:2]) or "no output"}
                continue
            d = {r["phase"]: r for r in rows if r.get("part") == "decode"}
            out[(ctx, e)] = {
                "status": "ok",
                "cold_ttft": d["cold"]["ttft_s"],
                "cold_tps": d["cold"]["dec_tps"],
                "warm_ttft": d["warm"]["ttft_s"],
                "turn2_ttft": d["turn2"]["ttft_s"],
                "turn2_cached": d["turn2"].get("cached"),
                "pt": d["cold"]["pt"],
            }
    return out


def render_long(outdir: Path, set_name: str) -> str:
    data = collect_long(outdir, set_name)
    lines = [
        "",
        f"## Long context, {set_name} (code prompt, greedy, cold decode 2048 tokens; warm = same prompt again; follow-up = turn 2 reusing the prefix)",
        "",
        "| ctx | engine | prompt tok | cold TTFT s | cold decode tok/s | warm TTFT s | follow-up TTFT s (cached tok) | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (ctx, e), v in sorted(data.items()):
        if v["status"] != "ok":
            lines.append(f"| {ctx} | {e} | - | - | - | - | - | {v['status']} |")
        else:
            lines.append(
                f"| {ctx} | {e} | {v['pt']} | {v['cold_ttft']} | {v['cold_tps']} | {v['warm_ttft']} | "
                f"{v['turn2_ttft']} ({v['turn2_cached']}) | ok |"
            )
    return "\n".join(lines)


def collect(outdir: Path, set_name: str) -> dict:
    """(ctx, kind) -> engine -> {prompt: dec_tps} over complete, valid jobs only."""
    spec = SETS[set_name]
    out: dict = {}
    for group, ctx, kinds in spec["groups"]:
        for e in spec["engines"]:
            for path in sorted((outdir / e).glob(f"{group}-r*.jsonl")):
                job = make_job(e, group, ctx, kinds, 0, outdir)
                job.out = path
                rows = bs.read_rows(path)
                if bs.validate_rows(job, rows):
                    continue
                for r in rows:
                    if r.get("part") == "corpus" and r.get("dec_tps"):
                        kind = r["prompt"].split("-")[0]
                        out.setdefault((ctx, kind), {}).setdefault(e, {})[
                            f"{r['prompt']}@{path.stem}"
                        ] = r["dec_tps"]
    return out


# Same-weights reference per engine (None: no same-weights Yunshu arm exists, only the cross-weights column).
SAME_WEIGHTS_REF = {
    "yunshu-tf4": "yunshu-main",  # different pack: shown as cross only
    "tf-native": "yunshu-tf4",
    "splash-latest": "yunshu-tf4",
    "omlx-latest": "yunshu-main",
    "mtplx-latest": "yunshu-main",
    "fn-omlx-latest": "fn-yunshu-main",
}
SAME_PACK_ENGINES = (
    "tf-native",
    "splash-latest",
    "omlx-latest",
    "mtplx-latest",
    "fn-omlx-latest",
)


def _paired(values, ref):
    common = [p for p in values if p in ref and ref[p] > 0]
    if not common:
        return None
    import statistics

    r = [values[p] / ref[p] for p in common]
    return statistics.fmean(r), (statistics.stdev(r) if len(r) > 1 else 0.0), len(r)


def render(outdir: Path, set_name: str) -> str:
    import statistics

    spec = SETS[set_name]
    groups = collect(outdir, set_name)
    engines = spec["engines"]
    main_ref = spec["reference"]
    lines = [
        "",
        f"## Decode corpus, {set_name}: 10 prompts per cell and rep, greedy, 1024 decode tokens, q4 weights as listed",
        "Same-weights ratio: per-prompt speed over the Yunshu arm on the same weights (TF/Splash over yunshu-tf4; oMLX/MTPLX over "
        f"{main_ref}). Cross-weights ratio: over {main_ref} (oQ4e). +-: stdev of the per-prompt ratios.",
        "",
        "| cell | engine | mean / median tok/s (n) | same-weights ratio | cross-weights ratio |",
        "|---|---|---|---|---|",
    ]
    for (ctx, kind), per in sorted(groups.items()):
        for e in engines:
            v = per.get(e)
            if not v:
                lines.append(f"| {kind}-{ctx} | {e} | n/a | n/a | n/a |")
                continue
            tps = list(v.values())
            same = (
                _paired(v, per.get(SAME_WEIGHTS_REF.get(e, ""), {}))
                if e in SAME_PACK_ENGINES
                else None
            )
            cross = _paired(v, per.get(main_ref, {})) if e != main_ref else None
            f = lambda t: "n/a" if t is None else f"x{t[0]:.3f} +-{t[1]:.3f} (n={t[2]})"  # noqa: E731
            lines.append(
                f"| {kind}-{ctx} | {e} | {statistics.fmean(tps):.1f} / {statistics.median(tps):.1f} ({len(tps)}) | "
                f"{f(same) if e in SAME_PACK_ENGINES else 'reference' if e == 'yunshu-tf4' else 'n/a'} | {f(cross)} |"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["plan", "pilot", "cells", "status", "aggregate"])
    ap.add_argument("--set", choices=[*SETS, "all"], default="all")
    ap.add_argument("--engines", default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    configure()
    names = [k for k in SETS if not k.startswith("long")] if a.set == "all" else [a.set]
    chosen = [e for e in a.engines.split(",") if e]
    bad = [e for e in chosen if e not in be.ENGINES]
    if bad:
        sys.exit(f"unknown engines {bad}")
    jobs: list[bs.Job] = []
    plist: list[bs.Job] = []
    for n in names:
        engines = chosen or SETS[n]["engines"]
        plist += pilots([e for e in SETS[n]["engines"] if e in engines], a.out)
        jobs += cells(n, engines, a.reps, a.out)
    if a.cmd == "plan":
        bs.print_plan(plist + jobs)
        for j in plist + jobs:
            gone = be.missing_paths(j.engine)
            if gone:
                print(f"MISSING for {j.engine}: {gone}")
        return 0
    if a.cmd == "status":
        for j in plist + jobs:
            problems = bs.validate_rows(j, bs.read_rows(j.out))
            print(
                f"{'ok  ' if not problems else 'todo'} {j.name}  {'; '.join(problems[:2])}"
            )
        return 0
    if a.cmd == "aggregate":
        text = "\n".join(
            render_long(a.out, n)
            if SETS[n].get("part") == "decode"
            else render(a.out, n)
            for n in names
        )
        if a.md:
            a.md.write_text(text)
        print(text)
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "pilot":
        for j in plist:
            gone = be.missing_paths(j.engine)
            if gone:
                sys.exit(f"{j.engine}: missing {gone}")
        print(json.dumps(bs.submit_jobs(plist, a.out)))
        return 0
    blocked = sorted(
        {
            j.engine
            for j in jobs
            if not bs.is_complete(plist[[p.engine for p in plist].index(j.engine)])
        }
    )
    if blocked:
        sys.exit(
            f"pilot not validated for {blocked}; run `pilot` and read the logs first"
        )
    print(json.dumps(bs.submit_jobs(jobs, a.out, max_active=40)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
