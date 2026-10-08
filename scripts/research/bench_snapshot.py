#!/usr/bin/env python3
"""Cross-engine snapshot driver: plan -> gpuq -> validate -> aggregate (resumable, fail closed).

    bench_snapshot.py plan   [--dry-run]            print every planned job, count and GPU-hour estimate
    bench_snapshot.py submit --stage pilot          one smoke job per engine (engaged-mode proof), priority 0
    bench_snapshot.py submit --stage cells          measurement cells; refuses engines whose pilot did not validate
    bench_snapshot.py submit --stage agent          agentbench: Yunshu new vs base (+ TensorFold)
    bench_snapshot.py status                        which jobs are complete / failed / missing
    bench_snapshot.py aggregate [--md FILE]         medians and ranges as markdown

Every cell is one gpuq job (priority 0, label snapshot014-<engine>-<group>-r<rep>, --quiet) that runs
scripts/research/tfbench.py once: one fresh server, one engine, one cell group. A job is complete only
when its JSONL validates (engaged spec mode == expected, part_done, memory row, every row carries the
engine / version / sha / flags / drafter / spec mode / checkpoint fields, the expected number of cells).
Re-running `submit` skips complete jobs and resubmits failed ones, never duplicating a queued one.
Submit order is rep-major, cell-group-major, engine-minor: engines interleave inside every cell.
Nothing here touches the GPU by itself; the planning and validation functions are pure (unit tests).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_engines as be  # noqa: E402

MAIN = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
GPUQ = str(MAIN / "scripts/dev/gpuq")
AGENTBENCH = str(MAIN / "scripts/dev/agentbench")
TFBENCH = str(HERE / "tfbench.py")
PY = be.MAIN_PY
BUILD = Path("/Volumes/P5Plus/yunshu-build/bench014")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "docs/research/snapshot014/runs"
PRIORITY = 0
LABEL_PREFIX = "snapshot014"

CTXS = (1024, 8192, 32768, 65536, 131072)
KINDS = ("prose", "code")
DECODE_TOKENS = 2048
TURN2_TOKENS = 256
NEEDLES = 10
PHASES = ("cold", "warm", "turn2")
ENGINE_ORDER = (
    "yunshu-new",
    "tf-new",
    "splash",
    "omlx",
    "mtplx",
    "mlxlm",
    "llamacpp",
)
REQUIRED_META = (
    "engine",
    "version",
    "git_sha",
    "flags",
    "drafter",
    "spec_mode",
    "checkpoint",
)

# (group, part, ctxs, kinds): the cell groups. Short contexts share a server, long ones get their own
# job so each stays inside one gpuq timeout and a failure costs one cell.
DECODE_GROUPS = (
    ("d1k", (1024,), KINDS),
    ("d8k", (8192,), KINDS),
    ("d32k", (32768,), KINDS),
    ("d64k-prose", (65536,), ("prose",)),
    ("d64k-code", (65536,), ("code",)),
    ("d128k-prose", (131072,), ("prose",)),
    ("d128k-code", (131072,), ("code",)),
)
NEEDLE_GROUPS = (("n32k", 32768), ("n64k", 65536), ("n128k", 131072))
CONC_NS = (2, 4)
CONC_TRIALS = 2


@dataclass
class Job:
    name: str
    engine: str
    part: str
    group: str
    rep: int
    stage: str  # pilot | cell
    ctxs: tuple = ()
    kinds: tuple = ()
    est_min: float = 0.0
    mem_gb: int = 50
    timeout_min: int = 20
    stall_min: int = 10
    out: Path = field(default_factory=Path)
    env: dict = field(default_factory=dict)
    argv: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{LABEL_PREFIX}-{self.name}"


# ---- GPU-time model (estimate; calibrated on the 2026-09-28 Splash log and the 2026-10-02 tfbench rows)

_PREFILL_TPS = (
    (1024, 900.0),
    (8192, 1040.0),
    (32768, 940.0),
    (65536, 810.0),
    (131072, 633.0),
)
_STARTUP_S = {
    "yunshu-new": 90,
    "yunshu-base": 90,
    "tf-new": 60,
    "splash": 60,
    "omlx": 90,
    "mtplx": 90,
    "mlxlm": 45,
    "llamacpp": 60,
}


def prefill_seconds(engine: str, ctx: int) -> float:
    pts = _PREFILL_TPS
    if ctx <= pts[0][0]:
        tps = pts[0][1]
    elif ctx >= pts[-1][0]:
        tps = pts[-1][1]
    else:
        tps = pts[-1][1]
        for (c0, t0), (c1, t1) in zip(pts, pts[1:], strict=False):
            if c0 <= ctx <= c1:
                tps = t0 + (t1 - t0) * (ctx - c0) / (c1 - c0)
                break
    return ctx / (tps * be.ENGINES[engine].prefill_factor)


def decode_seconds(engine: str, ctx: int, tokens: int) -> float:
    return tokens / (be.ENGINES[engine].decode_tps / (1 + ctx / 262144))


def decode_cell_seconds(engine: str, ctx: int) -> float:
    """cold (prefill + 2048 decode) + warm (cached prefix) + follow-up turn."""
    p = prefill_seconds(engine, ctx)
    return (
        p
        + decode_seconds(engine, ctx, DECODE_TOKENS)
        + 0.15 * p
        + 3
        + decode_seconds(engine, ctx, DECODE_TOKENS)
        + 5
        + decode_seconds(engine, ctx, TURN2_TOKENS)
    )


def estimate_minutes(engine: str, part: str, ctxs, kinds) -> float:
    s = _STARTUP_S[engine] + 30 + 20  # + idle footprint wait + warm-up requests
    if part == "decode":
        s += sum(decode_cell_seconds(engine, c) for c in ctxs for _ in kinds)
    elif part == "needle":
        for c in ctxs:
            # Generic hybrid-cache servers may prefill every changed question.
            # Reserve the cold worst case rather than assuming prefix reuse.
            s += NEEDLES * (prefill_seconds(engine, c) + 4)
    elif part == "conc":
        s += (
            len(CONC_NS)
            * CONC_TRIALS
            * (sum(CONC_NS) / len(CONC_NS))
            * (decode_seconds(engine, 32768, 2048) + prefill_seconds(engine, 32768))
        )
    elif part == "smoke":
        s += 60
    return s / 60.0


def job_mem_gb(engine: str, ctxs) -> int:
    top = max(ctxs) if ctxs else 1024
    base = 40 if top <= 8192 else 44 if top <= 32768 else 50 if top <= 65536 else 60
    if engine in ("splash", "omlx", "mtplx"):
        base += 8
    if engine == "llamacpp" and top >= 65536:
        base += 8
    return base


def timeout_minutes(est_min: float) -> int:
    """Honest timeout: 1.6x the estimate + 5 minutes, at least 10, rounded up (long cells exceed the
    20-minute default on purpose; the estimate is printed next to it)."""
    return max(10, int(est_min * 1.6 + 5 + 0.999))


def stall_minutes(engine: str, ctxs) -> int:
    """gpuq stops a job whose log is silent this long; a cold 128K prefill prints nothing meanwhile."""
    top = max(ctxs) if ctxs else 1024
    return max(10, int(prefill_seconds(engine, top) * 1.5 / 60) + 5)


# ---- planning --------------------------------------------------------------------------------------


def tfbench_argv(job: Job, out: Path) -> list[str]:
    argv = [
        PY,
        TFBENCH,
        "--engine",
        job.engine,
        "--part",
        job.part,
        "--rep",
        str(job.rep),
        "--out",
        str(out),
        "--tag",
        f"-{job.group}",
    ]
    if job.stage == "pilot":
        return argv + ["--smoke", "--only-kind", "prose", "--idle-s", "5"]
    if job.part == "decode":
        for c in job.ctxs:
            argv += ["--only-ctx", str(c)]
        for k in job.kinds:
            argv += ["--only-kind", k]
        argv += [
            "--decode-tokens",
            str(DECODE_TOKENS),
            "--long-ask",
            "--turn2-tokens",
            str(TURN2_TOKENS),
            "--ctx-tokens",
            str(max(job.ctxs) + 4096),
        ]
    elif job.part == "needle":
        for c in job.ctxs:
            argv += ["--only-ctx", str(c)]
        argv += ["--ctx-tokens", str(max(job.ctxs) + 4096)]
    elif job.part == "conc":
        argv += [
            "--conc-ns",
            ",".join(map(str, CONC_NS)),
            "--conc-trials",
            str(CONC_TRIALS),
            "--ctx-tokens",
            str((32768 + 4096) * max(CONC_NS)),
            "--conc-tokens",
            "2048",
            "--parallel",
            str(max(CONC_NS)),
        ]
    return argv


def plan_cells(
    engines, reps: int, outdir: Path, trees: dict, needle_reps: int = 1
) -> list[Job]:
    """All measurement jobs: rep-major, group-major, engine-minor (engines interleave inside a cell)."""
    jobs: list[Job] = []

    def add(engine, part, group, rep, ctxs, kinds):
        name = f"{engine}-{group}-r{rep}"
        est = estimate_minutes(engine, part, ctxs, kinds)
        job = Job(
            name,
            engine,
            part,
            group,
            rep,
            "cell",
            tuple(ctxs),
            tuple(kinds),
            est,
            72 if part == "conc" else job_mem_gb(engine, ctxs),
            timeout_minutes(est),
            stall_minutes(engine, ctxs),
            outdir / engine / f"{group}-r{rep}.jsonl",
        )
        job.env = job_env(engine, trees)
        job.argv = tfbench_argv(job, job.out)
        jobs.append(job)

    for rep in range(reps):
        for group, ctxs, kinds in DECODE_GROUPS:
            for e in engines:
                add(e, "decode", group, rep, ctxs, kinds)
        for e in engines:
            add(e, "conc", "c2c4", rep, (), ())
        if rep < needle_reps:
            for group, ctx in NEEDLE_GROUPS:
                for e in engines:
                    add(e, "needle", group, rep, (ctx,), ("prose",))
    return jobs


def plan_pilots(engines, outdir: Path, trees: dict) -> list[Job]:
    jobs = []
    for e in engines:
        est = estimate_minutes(e, "smoke", (), ())
        job = Job(
            f"{e}-smoke-pilot",
            e,
            "decode",
            "pilot",
            0,
            "pilot",
            (512,),
            (),
            est,
            job_mem_gb(e, ()),
            max(8, timeout_minutes(est)),
            10,
            outdir / e / "pilot.jsonl",
        )
        job.env = job_env(e, trees)
        job.argv = tfbench_argv(job, job.out)
        jobs.append(job)
    return jobs


def job_env(engine: str, trees: dict) -> dict:
    env = {
        "TFB_OUT": str(BUILD / "work"),
        "TFB_PORT_LAST": "18999",
        "TFB_WORK": "/Volumes/P5Plus/yunshu-build/snapshot014/corpus",
        "TFB_EXACT_PROMPTS": "1",
        "GPUQ_OWNER": "snapshot014",
        "GPUQ_DIR": "/Volumes/P5Plus/yunshu-gpuq",
        "TFB_YUNSHU_BIN": "/Volumes/P5Plus/yunshu-test-envs/snapshot014-yunshu/bin/yunshu",
        "TFB_MLXLM_PY": "/Volumes/P5Plus/yunshu-test-envs/snapshot014-mlxlm/bin/python",
    }
    if be.ENGINES[engine].kind == "yunshu":
        src = trees.get(engine)
        if src:
            env["TFB_YUNSHU_SRC"] = src
    return env


# ---- validation (pure) -----------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
    rows = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def expected_rows(job: Job) -> dict:
    if job.stage == "pilot":
        return {"decode": 3}
    if job.part == "decode":
        return {"decode": len(job.ctxs) * len(job.kinds) * len(PHASES)}
    if job.part == "needle":
        return {"needle": len(job.ctxs) * NEEDLES}
    return {"conc": len(CONC_NS) * CONC_TRIALS}


def validate_rows(job: Job, rows: list[dict]) -> list[str]:
    """Problems that make a job's output unusable (empty list: complete and trustworthy)."""
    if not rows:
        return ["no output"]
    problems = []
    done = [
        r for r in rows if r.get("part") == "part_done" and r.get("complete") is True
    ]
    if not done:
        problems.append("no part_done (incomplete job)")
    want = be.ENGINES[job.engine].expected_mode
    sessions = [r for r in rows if r.get("part") == "session"]
    if len(sessions) != 1:
        problems.append(f"{len(sessions)} session rows")
    for s in sessions:
        if s.get("engaged_spec_mode") != want:
            problems.append(
                f"engaged spec mode {s.get('engaged_spec_mode')!r}, want {want!r}"
            )
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_META if k not in r]
        if missing:
            problems.append(f"row {i} lacks {missing}")
            break
        if r.get("spec_mode") != want:
            problems.append(f"row {i} spec_mode {r.get('spec_mode')!r}, want {want!r}")
            break
    if be.ENGINES[job.engine].kind == "yunshu" and not any(
        r.get("git_sha") for r in rows
    ):
        problems.append("yunshu rows carry no git sha")
    if not [r for r in rows if r.get("part") == "memory"]:
        problems.append("no memory row")
    for part, n in expected_rows(job).items():
        got = len([r for r in rows if r.get("part") == part])
        if got != n:
            problems.append(f"{part}: {got} rows, want {n}")
    expected_sha = job.env.get("TFB_EXPECT_YUNSHU_SHA")
    if expected_sha and any(r.get("git_sha") != expected_sha for r in rows):
        problems.append("release SHA does not match pinned engine")
    if rows[-1].get("part") != "part_done" or rows[-1].get("complete") is not True:
        problems.append("terminal record is not complete")
    if any(r.get("engine") != job.engine for r in rows):
        problems.append("engine identity mismatch")
    if job.stage == "cell" and job.part == "decode":
        got = Counter(
            (r.get("ctx"), r.get("kind"), r.get("phase"))
            for r in rows
            if r.get("part") == "decode"
        )
        want_keys = Counter(
            (c, k, p) for c in job.ctxs for k in job.kinds for p in PHASES
        )
        if got != want_keys:
            problems.append(
                "decode coverage contains missing, duplicate or unexpected cells"
            )
        for r in rows:
            if r.get("part") != "decode":
                continue
            for metric in ("ttft_s", "dec_tps"):
                v = r.get(metric)
                if not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
                    problems.append(f"invalid {metric}")
            if (
                be.ENGINES[job.engine].weights == "same"
                and r.get("phase") in ("cold", "warm")
                and r.get("pt") != r.get("ctx")
            ):
                problems.append(
                    "actual server prompt tokens differ from reference budget"
                )
    if job.stage == "cell" and job.part == "decode":
        for r in rows:
            if (
                r.get("part") == "decode"
                and r.get("phase") in ("cold", "warm")
                and (
                    r.get("ct") != DECODE_TOKENS
                    or r.get("reference_prompt_tokens") != r.get("ctx")
                )
            ):
                problems.append(
                    f"decode {r.get('kind')}-{r.get('ctx')} {r.get('phase')}: ct={r.get('ct')}, reference_prompt_tokens={r.get('reference_prompt_tokens')}"
                )
                break
    return problems


def is_complete(job: Job) -> bool:
    return not validate_rows(job, read_rows(job.out))


# ---- gpuq ------------------------------------------------------------------------------------------


def submit_args(job: Job, attempt: int = 0) -> list[str]:
    return [
        GPUQ,
        "submit",
        "--label",
        job.label + (f"-a{attempt}" if attempt else ""),
        "--device",
        "m5",
        "--timeout",
        str(job.timeout_min),
        "--stall",
        str(job.stall_min),
        f"--priority={PRIORITY}",
        "--mem-gb",
        str(job.mem_gb),
        *(["--quiet"] if job.stage != "pilot" else []),
        "--out",
        str(job.out),
        "--expect-complete",
        "--",
        *job.argv,
    ]


def load_state(outdir: Path) -> dict:
    f = outdir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_state(outdir: Path, state: dict) -> None:
    (outdir / "state.json").write_text(json.dumps(state, indent=1))


def job_active(job_id: str) -> bool:
    """True while gpuq still has the job (wait exits 2 when it is unfinished)."""
    r = subprocess.run(
        [GPUQ, "wait", "--max-seconds", "1", job_id], capture_output=True, text=True
    )
    return r.returncode == 2


def submit_jobs(jobs, outdir: Path, max_attempts: int = 2) -> dict:
    state = load_state(outdir)
    counts = {"complete": 0, "queued": 0, "submitted": 0, "gave_up": 0}
    for job in jobs:
        if counts["queued"] + counts["submitted"] >= 6:
            break
        if is_complete(job):
            counts["complete"] += 1
            continue
        st = state.get(job.name, {})
        if st.get("id") and job_active(st["id"]):
            counts["queued"] += 1
            continue
        if st.get("attempts", 0) >= max_attempts:
            counts["gave_up"] += 1
            print(
                f"giving up on {job.name} after {st['attempts']} attempts",
                file=sys.stderr,
            )
            continue
        job.out.parent.mkdir(parents=True, exist_ok=True)
        if job.out.exists():  # a failed attempt's output must not mix with the retry
            job.out.rename(job.out.with_suffix(f".failed{st.get('attempts', 0)}.jsonl"))
        r = subprocess.run(
            submit_args(job, st.get("attempts", 0) + 1),
            capture_output=True,
            text=True,
            env={**os.environ, **job.env},
        )
        if r.returncode or not r.stdout.strip():
            print(f"submit {job.name} failed: {r.stderr.strip()}", file=sys.stderr)
            return {**counts, "error": job.name}
        state[job.name] = {
            "id": r.stdout.split()[-1],
            "attempts": st.get("attempts", 0) + 1,
            "submitted": time.strftime("%FT%T"),
        }
        save_state(outdir, state)
        counts["submitted"] += 1
    return counts


def pilot_ok(engine: str, outdir: Path) -> bool:
    j = plan_pilots([engine], outdir, {})[0]
    return is_complete(j)


# ---- trees -----------------------------------------------------------------------------------------


def pin_tree(ref: str) -> tuple[str, str]:
    """(sha, <tree>/python) of a pinned detached worktree of `ref` in the main repo."""
    sha = subprocess.run(
        ["git", "-C", str(MAIN), "rev-parse", ref + "^{commit}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not sha:
        sys.exit(f"unknown ref {ref}")
    tree = BUILD / "trees" / sha[:12]
    if not (tree / "python").exists():
        tree.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                "git",
                "-C",
                str(MAIN),
                "worktree",
                "add",
                "-q",
                "--detach",
                str(tree),
                sha,
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            sys.exit(r.stderr)
    actual = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != sha:
        sys.exit(f"pinned tree mismatch: expected {sha}, got {actual}")
    return sha, str(tree / "python")


def resolve_trees(args, dry: bool) -> tuple[dict, dict]:
    refs = {"yunshu-new": args.new_ref, "yunshu-base": args.base_ref}
    trees, shas = {}, {}
    for eng, ref in refs.items():
        if dry:
            sha = subprocess.run(
                ["git", "-C", str(MAIN), "rev-parse", ref + "^{commit}"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            trees[eng], shas[eng] = f"<tree@{(sha or ref)[:12]}>/python", sha
        else:
            shas[eng], trees[eng] = pin_tree(ref)
    return trees, shas


# ---- agent stage -----------------------------------------------------------------------------------


def agent_jobs(args) -> list[dict]:
    """agentbench invocations (Yunshu new/base) and TensorFold opencode cells; planning only."""
    spec = []
    for tag, ref in (("new", args.new_ref), ("base", args.base_ref)):
        spec.append(
            {
                "name": f"agent-{tag}",
                "ref": ref,
                "agents": args.agents,
                "kind": "agentbench",
                "label_prefix": f"{LABEL_PREFIX}-agent-{tag}",
            }
        )
    if args.agent_tf:
        spec.append(
            {
                "name": "agent-tf",
                "agents": "opencode",
                "kind": "tensorfold",
                "label_prefix": f"{LABEL_PREFIX}-agent-tf",
            }
        )
    return spec


def agent_argv(spec: dict, outdir: Path) -> tuple[list[str], dict]:
    env = {"AGENTIC_YUNSHU_DRAFTER": be.DRAFTER, "AGENTIC_EXPECT_SPEC": "dflash"}
    if spec["kind"] == "agentbench":
        return (
            [
                AGENTBENCH,
                "--ref",
                spec["ref"],
                "--agents",
                spec["agents"],
                "--tasks",
                "all",
                "--repeat",
                "1",
                "--priority",
                str(PRIORITY),
                "--label-prefix",
                spec["label_prefix"],
                "--baseline",
                "none",
                "--no-wait",
            ],
            env,
        )
    raise ValueError("tensorfold agent cells are submitted per task by submit_agent")


def agent_tasks() -> list[str]:
    r = subprocess.run(
        [AGENTBENCH, "--dry-run", "--agents", "opencode"],
        capture_output=True,
        text=True,
    )
    return [j["task"] for j in json.loads(r.stdout)]


def submit_agent_tf(outdir: Path) -> dict:
    """One gpuq job per task: run_agentic against TensorFold 0.6.1 + DFlash2 (the harness default is a
    different install, 0.6.5, so the binary is pinned through AGENTIC_TF_BIN)."""
    state = load_state(outdir)
    counts = {"submitted": 0, "queued": 0}
    for task in agent_tasks():
        name = f"agent-tf-{task}"
        out = outdir / "agent-tf" / f"{task}.jsonl"
        st = state.get(name, {})
        if st.get("id") and job_active(st["id"]):
            counts["queued"] += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            GPUQ,
            "submit",
            "--label",
            f"{LABEL_PREFIX}-{name}",
            "--device",
            "m5",
            "--timeout",
            "28",
            "--stall",
            "15",
            f"--priority={PRIORITY}",
            "--mem-gb",
            "50",
            "--out",
            str(out),
            "--",
            *tf_agent_cmd(task, out),
        ]
        env = {
            **os.environ,
            "AGENTIC_TF_BIN": be.TF_BIN,
            "AGENTIC_PORT_LO": "18994",
            "AGENTIC_PORT_HI": "18996",
        }
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode or not r.stdout.strip():
            return {**counts, "error": name, "stderr": r.stderr.strip()}
        state[name] = {"id": r.stdout.split()[-1], "submitted": time.strftime("%FT%T")}
        save_state(outdir, state)
        counts["submitted"] += 1
    return counts


def tf_agent_cmd(task: str, out: Path) -> list[str]:
    return [
        PY,
        str(MAIN / "scripts/research/agentic/run_agentic.py"),
        "run",
        "--serve",
        "tensorfold",
        "--checkpoint",
        be.CHECKPOINT,
        "--engine-label",
        "tensorfold-default",
        "--agent",
        "opencode",
        "--tasks",
        task,
        "--repeat",
        "1",
        "--timeout-min",
        "20",
        "--budget-min",
        "1",
        "--output",
        str(out),
    ]


AGENT_MIN_PER_JOB = 7.0


# ---- reporting -------------------------------------------------------------------------------------


def med_range(vals) -> str:
    v = [x for x in vals if isinstance(x, int | float)]
    if not v:
        return "unknown"
    m = statistics.median(v)
    return (
        f"{m:.2f} ({min(v):.2f}-{max(v):.2f}) n={len(v)}"
        if len(v) > 1
        else f"{m:.2f} n=1"
    )


def collect(outdir: Path, engines, yv_run: Path | None = None) -> dict:
    """{(engine, metric, ctx, kind): [values per rep]} from every validated output."""
    cells: dict = {}
    for e in engines:
        files = (
            sorted((yv_run / "cells").glob(f"snapshot.{e}-*.jsonl"))
            if yv_run
            else sorted((outdir / e).glob("*.jsonl"))
            if (outdir / e).is_dir()
            else []
        )
        for f in files:
            name = f.name.removeprefix(f"snapshot.{e}-") if yv_run else f.name
            if ".failed" in name or name == "pilot.jsonl":
                continue
            rows = read_rows(f)
            planned = {
                j.out.name: j for j in plan_cells([e], 9, outdir, {}, needle_reps=9)
            }
            job = planned.get(name)
            if job is None or validate_rows(job, rows):
                continue
            if any(r.get("contended") for r in rows):
                continue
            for r in rows:
                p = r.get("part")
                if p == "decode":
                    k = (e, r["phase"], r["ctx"], r["kind"])
                    cells.setdefault(k + ("ttft_s",), []).append(r.get("ttft_s"))
                    if (
                        r["phase"] == "warm"
                        and isinstance(r.get("cached"), int)
                        and isinstance(r.get("pt"), int)
                        and r["cached"] >= r["pt"] - 1
                    ):
                        cells.setdefault(
                            (e, "fullhit", r["ctx"], r["kind"], "ttft_s"), []
                        ).append(r.get("ttft_s"))
                    if r["phase"] == "cold":
                        cells.setdefault(k + ("dec_tps",), []).append(r.get("dec_tps"))
                elif p == "needle":
                    cells.setdefault(
                        (e, "needle", r["ctx"], "prose", "correct"), []
                    ).append(1.0 if r.get("correct") else 0.0)
                elif p == "conc":
                    cells.setdefault((e, "conc", r["n"], "-", "agg_tps"), []).append(
                        r.get("agg_tps")
                    )
                    for metric, field in (
                        ("mean_ttft_s", "ttfts"),
                        ("mean_dec_tps", "per_req_dec"),
                    ):
                        values = r.get(field, [])
                        if values and all(isinstance(v, int | float) for v in values):
                            cells.setdefault(
                                (e, "conc", r["n"], "-", metric), []
                            ).append(statistics.mean(values))
                elif p == "memory":
                    cells.setdefault(
                        (e, "mem", job.out.stem, "-", "peak_gib"), []
                    ).append(r.get("peak_gib"))
                    cells.setdefault(
                        (e, "mem", job.out.stem, "-", "idle_gib"), []
                    ).append(r.get("idle_gib"))
                    for metric in ("peak_rss_gib", "idle_rss_gib"):
                        if isinstance(r.get(metric), int | float):
                            cells.setdefault(
                                (e, "mem", job.out.stem, "-", metric), []
                            ).append(r[metric])
    return cells


def fmt_ctx(c) -> str:
    return f"{c // 1024}K" if isinstance(c, int) and c >= 1024 else str(c)


GAP_HYPOTHESES = {
    "yunshu-new": {
        "ttft": "Hybrid-state checkpoint restore, prefix lookup and first-token dispatch may dominate warm TTFT; cold TTFT includes prefill projections and GDN recurrence.",
        "decode": "DFlash acceptance, verify block cost and prompt-copy opportunities vary between prose/code and context lengths.",
        "memory": "Retained APC snapshots, allocator pools and speculative scratch may explain peak/idle differences.",
    },
    "tf-new": {
        "ttft": "Snapshot selection/restore and prefill kernel scheduling may account for differences.",
        "decode": "DFlash verify kernels, proposal acceptance and scheduler overhead may account for differences.",
        "memory": "Snapshot retention and allocator release policy may account for differences.",
    },
    "mlxlm": {
        "ttft": "Generic prefill kernels and hybrid-cache prefix reuse limits may account for differences.",
        "decode": "Autoregressive decoding evaluates the target each token; speculative engines amortize target verification over accepted proposals.",
        "memory": "No speculative drafter reduces resident weights; prompt cache and allocator retention still contribute.",
    },
    "omlx": {
        "ttft": "Paged prefix-cache lookup, hybrid state restore and SSD/hot-cache transitions may account for differences.",
        "decode": "DFlash acceptance and continuous-batch scheduling may account for differences.",
        "memory": "Hot/SSD cache policy, page pools and concurrent-request guard may account for differences.",
    },
    "splash": {
        "ttft": "Native prefill and GDN-state/cache scheduling may account for differences; BF16 KV is deliberately used here.",
        "decode": "Native Metal verification and DFlash proposal batching may account for differences; target weights differ.",
        "memory": "Native buffer arenas and prefix retention may account for differences; different target quantization confounds comparisons.",
    },
    "llamacpp": {
        "ttft": "GGUF kernels, graph scheduling and slot prefix reuse may account for differences; weights differ from oQ4e.",
        "decode": "Native MTP proposal acceptance, GGUF kernel layout and Metal graph dispatch may account for differences.",
        "memory": "Preallocated per-slot KV and graph buffers may account for differences; GGUF weight sizes differ.",
    },
    "mtplx": {
        "ttft": "Native-MTP hybrid prefill and SessionBank snapshot restore may account for differences.",
        "decode": "Native MTP acceptance, compiled verification routing and per-step state work may account for differences.",
        "memory": "Repaged KV and SessionBank retention may account for peak/idle differences.",
    },
}


def render_markdown(cells: dict, engines) -> str:
    out = [
        "# Snapshot tables",
        "",
        "Cells are median (min-max) n=<reps>; TTFT in s, decode in tok/s, memory in GiB.",
        "",
    ]

    def table(title, metric_key, phase, fmt=med_range, suffix="ttft_s"):
        out.append(f"## {title}")
        out.append("")
        out.append("| ctx / kind | " + " | ".join(engines) + " |")
        out.append("|---|" + "---|" * len(engines))
        for ctx in CTXS:
            for kind in KINDS:
                vals = [
                    fmt(cells.get((e, phase, ctx, kind, suffix), [])) for e in engines
                ]
                samples = [
                    cells.get((e, phase, ctx, kind, suffix), []) for e in engines
                ]
                medians = [
                    statistics.median([x for x in v if isinstance(x, int | float)])
                    if sum(isinstance(x, int | float) for x in v) >= 3
                    else None
                    for v in samples
                ]
                measured = [v for v in medians if v is not None]
                best = (
                    (max(measured) if suffix == "dec_tps" else min(measured))
                    if measured
                    else None
                )
                if best and best > 0:
                    vals = [
                        v
                        + (
                            f"; gap {100 * (1 - m / best if suffix == 'dec_tps' else m / best - 1):.1f}%"
                            if m is not None
                            else "; gap unknown (<3 reps)"
                        )
                        for v, m in zip(vals, medians, strict=True)
                    ]
                out.append(f"| {fmt_ctx(ctx)} {kind} | " + " | ".join(vals) + " |")
        out.append("")

    table("Cold TTFT", "ttft", "cold")
    table("Warm full-hit TTFT (cached_tokens >= prompt_tokens - 1)", "ttft", "fullhit")
    table("Warm repeated-request TTFT (cache coverage varies)", "ttft", "warm")
    table("Follow-up turn TTFT", "ttft", "turn2")
    table(
        "Decode tok/s (2048-token reply, cold request)", "dec", "cold", suffix="dec_tps"
    )
    out += [
        "## Long-context recall (needle, correct / asked)",
        "",
        "| ctx | " + " | ".join(engines) + " |",
        "|---|" + "---|" * len(engines),
    ]
    for ctx in (32768, 65536, 131072):
        row = []
        for e in engines:
            v = cells.get((e, "needle", ctx, "prose", "correct"), [])
            row.append(f"{int(sum(v))}/{len(v)}" if v else "unknown")
        out.append(f"| {fmt_ctx(ctx)} | " + " | ".join(row) + " |")

    def comparison(values, maximize=False):
        medians = [statistics.median(v) if len(v) >= 3 else None for v in values]
        eligible = [v for v in medians if v is not None]
        best = (max(eligible) if maximize else min(eligible)) if eligible else None
        return [
            med_range(v)
            + (
                f"; gap {100 * (1 - m / best if maximize else m / best - 1):.1f}%"
                if best and m is not None
                else "; gap unknown"
            )
            for v, m in zip(values, medians, strict=True)
        ]

    for metric, title, maximize in (
        ("agg_tps", "Concurrency effective throughput (prefill included), tok/s", True),
        ("mean_ttft_s", "Concurrency mean per-request TTFT, seconds", False),
        ("mean_dec_tps", "Concurrency mean per-request decode, tok/s", True),
    ):
        out += [
            "",
            f"## {title} (32K prompts, 2048-token replies)",
            "",
            "| n | " + " | ".join(engines) + " |",
            "|---|" + "---|" * len(engines),
        ]
        for n in CONC_NS:
            values = [cells.get((e, "conc", n, "-", metric), []) for e in engines]
            out.append(f"| {n} | " + " | ".join(comparison(values, maximize)) + " |")

    groups = [g for g, _, _ in DECODE_GROUPS] + ["c2c4"] + [g for g, _ in NEEDLE_GROUPS]
    for metric, title in (
        ("peak_gib", "Sampled peak memory"),
        ("idle_gib", "Idle memory after 30 seconds"),
        ("peak_rss_gib", "Sampled peak RSS"),
        ("idle_rss_gib", "Idle RSS after 30 seconds"),
    ):
        out += [
            "",
            f"## {title} (process-tree accounting sum, GiB)",
            "",
            "| group | " + " | ".join(engines) + " |",
            "|---|" + "---|" * len(engines),
        ]
        for group in groups:
            values = [
                [
                    v
                    for rep in range(9)
                    for v in cells.get((e, "mem", f"{group}-r{rep}", "-", metric), [])
                ]
                for e in engines
            ]
            out.append(f"| {group} | " + " | ".join(comparison(values)) + " |")
    out.append("")
    out += [
        "## Gap hypotheses (unproven; applies to each positive gap in the corresponding engine/metric)",
        "",
        "| engine | metric | hypothesis |",
        "|---|---|---|",
    ]
    for engine in engines:
        for metric, hypothesis in GAP_HYPOTHESES.get(engine, {}).items():
            out.append(f"| {engine} | {metric} | {hypothesis} |")
    out.append("")
    out.append(
        "Weights differ for splash (own Splash quantization) and llamacpp (UD-Q4_K_M GGUF), see SETUP.md."
    )
    return "\n".join(out) + "\n"


# ---- commands --------------------------------------------------------------------------------------


def print_plan(jobs, extra_lines=()) -> None:
    for j in jobs:
        print(
            f"{j.label:<46} p{PRIORITY} mem={j.mem_gb:>2}G timeout={j.timeout_min:>3}m stall={j.stall_min:>2}m "
            f"est={j.est_min:5.1f}m  {' '.join(j.argv[2:12])} ..."
        )
    for line in extra_lines:
        print(line)
    by_engine: dict = {}
    for j in jobs:
        n, m = by_engine.get(j.engine, (0, 0.0))
        by_engine[j.engine] = (n + 1, m + j.est_min)
    print()
    print(f"{'engine':<14}{'jobs':>6}{'GPU h (est)':>14}")
    for e, (n, m) in by_engine.items():
        print(f"{e:<14}{n:>6}{m / 60:>14.1f}")
    print(
        f"{'total':<14}{sum(n for n, _ in by_engine.values()):>6}{sum(m for _, m in by_engine.values()) / 60:>14.1f}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["plan", "submit", "status", "aggregate"])
    ap.add_argument("--stage", choices=["pilot", "cells", "agent"], default="cells")
    ap.add_argument("--engines", default=",".join(ENGINE_ORDER))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--needle-reps", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--new-ref", default="v0.1.4", help="released 0.1.4 tag (pinned tree)"
    )
    ap.add_argument("--base-ref", default="v0.1.3")
    ap.add_argument("--agents", default="claude,opencode")
    ap.add_argument("--agent-tf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--yv-run", type=Path)
    a = ap.parse_args(argv)
    engines = [e for e in a.engines.split(",") if e]
    bad = [e for e in engines if e not in be.ENGINES]
    if bad:
        sys.exit(f"unknown engines {bad}; have {sorted(be.ENGINES)}")
    dry = a.dry_run or a.cmd in ("plan", "status", "aggregate")
    trees, shas = resolve_trees(a, dry)
    pilots = plan_pilots(engines, a.out, trees)
    cells = plan_cells(engines, a.reps, a.out, trees, a.needle_reps)

    if a.cmd == "plan":
        extra = [
            f"# new={a.new_ref} ({shas.get('yunshu-new', '')[:12]}) base={a.base_ref} ({shas.get('yunshu-base', '')[:12]})"
        ]
        print_plan(pilots + cells, extra)
        n_agent = 20 * len(engines)
        print(
            f"\nagent stage: {n_agent} jobs (20 tasks x engines, opencode; executed through yv snapshot), ~{n_agent * AGENT_MIN_PER_JOB / 60:.1f} GPU h "
            f"(estimate, ~{AGENT_MIN_PER_JOB:.0f} min per job)"
        )
        for e in engines:
            gone = be.missing_paths(e)
            print(f"preflight {e}: {'ready' if not gone else 'MISSING ' + str(gone)}")
        return 0
    if a.cmd == "status":
        for j in pilots + cells:
            problems = validate_rows(j, read_rows(j.out))
            print(
                f"{'ok ' if not problems else 'todo'} {j.name}  {'; '.join(problems[:2])}"
            )
        return 0
    if a.cmd == "aggregate":
        md = render_markdown(collect(a.out, engines, a.yv_run), engines)
        if a.md:
            a.md.write_text(md)
        print(md)
        return 0
    # submit
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "pilot":
        for e in engines:
            gone = be.missing_paths(e)
            if gone:
                sys.exit(f"{e}: missing {gone}")
        print(submit_jobs(pilots, a.out))
        return 0
    if a.stage == "cells":
        blocked = [e for e in engines if not pilot_ok(e, a.out)]
        if blocked:
            sys.exit(
                f"pilot not validated for {blocked}; run `submit --stage pilot` and read the logs first"
            )
        print(submit_jobs(cells, a.out))
        return 0
    sys.exit(
        "agent cells run through scripts/dev/yv --suite snapshot, pinned release/harness SHAs"
    )


if __name__ == "__main__":
    sys.exit(main())
