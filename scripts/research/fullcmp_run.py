#!/usr/bin/env python3
"""Batch driver for the full fair comparison (FULLCMP_SPEC.md, user 2026-11-03).

Reads manifest rows (manifest.jsonl, written by fullcmp_manifest.py) for one batch and turns them
into gpuq jobs. Only status "unverified" rows are runnable (costed, census gives no negative
evidence); "unsupported" and "cut" rows are skipped and listed. A row also needs a tfbench part
(see harness_part) or it is skipped and listed with the reason.

Grouping: one server configuration = (engine, model, thinking). Each group becomes one job per rep
when its estimated time fits the job cap (20 min); a larger group is split into consecutive chunks
of <= 20 min (each chunk cold-starts its server again).

Rotation (rep r): the batch's engines, in manifest order, are rotated left by r, so rep r starts at
engine (r mod n). Within a rep, a group's jobs for one engine are consecutive; engines follow the
rotated order. Across a rep boundary the last engine of rep r is engine (r-1 mod n) and the first
engine of rep r+1 is engine (r+1 mod n), so they differ whenever n >= 3 (n = 2 is unavoidable; the
dry run prints the count). The rep-3 slots use rotation by REP3_INDEX (2), the same rule as
fullcmp_reps.rep3_order.

Submission: one job per (engine, group, chunk, rep):
    gpuq submit --label fullcmp-<batch>-<engine>-<group>[-pK]-r<rep> --priority 1 --quiet -- <cmd>
where <cmd> is `/bin/sh -c "mkdir -p <dir> && <tfbench cell 1> && ..."`, one tfbench invocation per
cell with env TFB_FULLCMP=1, TFB_TELEMETRY_DIR, the pinned Yunshu tree (TFB_YUNSHU_BIN /
TFB_YUNSHU_SRC / PYTHONPATH from yunshu_pin.json) and an isolated HOME per engine. Raw output:
<RAW_ROOT>/<engine>/<cell>-r<rep>.jsonl.

Quiet machine: before a batch the driver runs fullcmp_quiet.gate on the host (refuses, exit 2,
naming the foreign holders); each job runs the same gate first, inside the job, with 3 retries 30 s
apart (fullcmp_quiet.py gate, log under <RUN_DIR>/quiet/<label>.jsonl). Every cell command carries
TFB_JOB_LABEL (the job's label) and GPUQ_DIR, so tfbench can resolve the job id when gpuq does not
export GPUQ_JOB_ID (fullcmp_provenance.resolve_job_id).

Phases: the swap-stability preflight (fullcmp_preflight.py) runs before submitting, as a CPU job
through gpuq. Then ONE blocking `gpuq wait` over all job ids, then fullcmp_audit.py over the raw
root, then fullcmp_reps.py lists cells needing rep 3; those are queued as a second, flagged
submission (label suffix -rep3) and waited for the same way.

Exit status: 0 ok; 1 failed job, audit violation, or failed rep-3 job; 2 refused (foreign GPU job,
preflight failed, unmapped row, bad input).

Usage:
    fullcmp_run.py --batch B1 --manifest MANIFEST.jsonl [--engines a,b] [--rep N]
                   [--dry-run] [--phase preflight|run] [--allow-foreign]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_engines  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_quiet  # noqa: E402
from fullcmp_profile import profile_env  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
TFBENCH = SCRIPTS / "tfbench.py"
AUDIT = SCRIPTS / "fullcmp_audit.py"
REPS = SCRIPTS / "fullcmp_reps.py"
PREFLIGHT = SCRIPTS / "fullcmp_preflight.py"
QUIET = SCRIPTS / "fullcmp_quiet.py"
PROFILE_SCRIPT = SCRIPTS / "fullcmp_profile.py"

# Harness commands run with THIS checkout's venv (the worktree that holds the driver), never the
# main checkout's .venv. gpuq itself is invoked by its literal path (user rule).
REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
GPUQ = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq"
WRAP_ROOT = Path("/Volumes/Micron/yunshu-fullcmp/wrap")
TFOUT_ROOT = Path(
    "/Volumes/Micron/yunshu-fullcmp/tfout"
)  # tfbench server homes and logs
GPUQ_DIR_DEFAULT = "/Volumes/P5Plus/yunshu-gpuq"

RUN_DIR = Path("/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103")
PIN_PATH = RUN_DIR / "yunshu_pin.json"
THRESH_PATH = RUN_DIR / "rep_thresholds.json"
AUDIT_DIR = RUN_DIR / "audit"
PREFLIGHT_DIR = RUN_DIR / "preflight"
QUIET_DIR = RUN_DIR / "quiet"
RAW_ROOT = Path("/Volumes/Micron/yunshu-fullcmp/raw")
HOME_ROOT = Path("/Volumes/Micron/yunshu-fullcmp/home")
MODEL_PATH = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"

BATCH_ORDER = ("B1", "B2", "B4", "B5")
SMOKE_BATCH = "SMOKE"  # pre-flight matrix (fullcmp_manifest --smoke); audits only its own rows
BATCH_CHOICES = BATCH_ORDER + (SMOKE_BATCH,)
JOB_CAP_MIN = 20.0
REP3_INDEX = 2  # same as fullcmp_reps.REP3_INDEX
# seconds between the cold reply and the follow-up of an idle cell (manifest: 5 s idle)
IDLE_THINK_S = fm.IDLE_S
CONC_TRIALS = (
    2  # conc records per cell per rep (tfbench --conc-trials); each holds conc requests
)
PRIORITY = "1"
RUN_ID = "fullcmp-1103"
ACTIVE_STATES = ("pending", "running", "paused")
JOB_ID_RE = re.compile(r"\d{4}-\d{6}-\d{2}-\S+")  # gpuq job id: <run>-<seq>-<label>
PREFLIGHT_TIMEOUT_MIN = "10"


# ---------------------------------------------------------------------------
# Manifest rows and planning
# ---------------------------------------------------------------------------


def load_rows(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rotate(order, rep: int) -> list:
    """Rotated interleave: rep r starts at order[r mod n]."""
    if not order:
        return []
    k = rep % len(order)
    return list(order[k:]) + list(order[:k])


def prompt_variant(row: dict) -> int:
    """0 = the real corpus prompt (every real-matrix row); N > 0 = a smoke-only variant file."""
    return int(row.get("prompt_variant") or 0)


def prompt_id(row: dict) -> str:
    """'<kind>-<ctx>' for variant 0 (unchanged for the real matrix), '<kind>-<ctx>-v<N>' otherwise."""
    base = f"{row['prompt_kind']}-{row['ctx_tokens']}"
    n = prompt_variant(row)
    return f"{base}-v{n}" if n else base


def harness_part(row: dict):
    """(tfbench part, extra args, None) for a row tfbench can run, else (None, None, reason).
    A variant row adds --prompt-variant N; variant 0 adds nothing, so real command lines are unchanged."""
    part, extra, reason = _base_harness_part(row)
    n = prompt_variant(row)
    if part is not None and n:
        extra = [*extra, "--prompt-variant", str(n)]
    return part, extra, reason


def _base_harness_part(row: dict):
    if row.get("table") == "agent":
        return None, None, "agent task selection not in tfbench CLI"
    if row.get("table") == "behaviour":
        return None, None, "behaviour checks not in tfbench CLI"
    sampling = row.get("sampling") or "greedy"
    if sampling not in ("greedy", "genconfig"):
        return None, None, f"sampling {sampling} not in tfbench CLI"
    if sampling == "genconfig" and int(row["conc"]) != 1:
        return None, None, "genconfig sampling is a conc-1 decode cell only"
    if (
        row.get("turns") == "idle5"
        and "fu_idle" in (row.get("phases") or ())
        and row.get("prompt_kind") in ("code", "prose")
        and int(row["conc"]) == 1
    ):
        # idle follow-up: the cold reply, then the follow-up after the idle gap (tfbench --think-s)
        return (
            "decode",
            [
                "--only-ctx",
                str(row["ctx_tokens"]),
                "--only-kind",
                row["prompt_kind"],
                "--decode-tokens",
                str(row["out_len"]),
                "--phases",
                "cold,turn2",
                "--think-s",
                str(IDLE_THINK_S),
            ],
            None,
        )
    kind = row.get("prompt_kind")
    conc = int(row["conc"])
    if row.get("turns") == "restart":
        # cold prefill, then the same prompt on a relaunched server that reads the same HOME
        # (tfbench restart_server): the disk prefix tier is the only way that row can hit
        if conc != 1 or kind not in ("code", "prose"):
            return None, None, "restart cell is a conc-1 code/prose decode cell"
        return (
            "decode",
            [
                "--only-ctx",
                str(row["ctx_tokens"]),
                "--only-kind",
                kind,
                "--decode-tokens",
                str(row["out_len"]),
                "--phases",
                "cold,restart",
            ],
            None,
        )
    if row.get("turns") == "mixed":
        # one long request and short ones started together (tfbench part_mixed); the short
        # requests are the conc row's n - 1
        return (
            "mixed",
            [
                "--mixed-long-ctx",
                str(row["ctx_tokens"]),
                "--mixed-short-n",
                str(conc - 1),
                "--mixed-short-ctx",
                "1024",
                "--conc-tokens",
                str(row["out_len"]),
                "--conc-trials",
                str(CONC_TRIALS),
            ],
            None,
        )
    if row.get("turns") not in (None, "single"):
        return None, None, f"turns={row.get('turns')} not in tfbench CLI"
    if kind not in ("code", "prose", *fm.FAMILY_KINDS):
        return None, None, f"prompt kind {kind} not in tfbench CLI"
    if conc == 1:
        args = [
            "--only-ctx",
            str(row["ctx_tokens"]),
            "--only-kind",
            kind,
            "--decode-tokens",
            str(row["out_len"]),
        ]
        # thinking / image flags only where they differ from the defaults: plain decode command
        # lines stay exactly as before
        if row.get("thinking") not in (None, "off"):
            args += ["--thinking", str(row["thinking"])]
        if row.get("media") not in (None, "none"):
            args += ["--media", str(row["media"])]
        if sampling == "genconfig":
            args += ["--sampling", "genconfig"]
        # tfbench runs every decode phase unless told otherwise: pass exactly the manifest's
        # phases (lead 2026-11-03: the audit expects the manifest phases, so the run must match)
        planned = set(row.get("phases") or ())
        if planned and planned <= {"cold", "warm", "turn2"}:
            args += ["--phases", ",".join(p for p in ("cold", "warm", "turn2") if p in planned)]
        return ("decode", args, None)
    same = [] if row.get("sweep") != "conc-same" else ["--conc-same"]
    return (
        "conc",
        [
            "--conc-ns",
            str(conc),
            "--conc-ctx",
            str(row["ctx_tokens"]),
            "--conc-tokens",
            str(row["out_len"]),
            "--conc-trials",
            str(CONC_TRIALS),
            *same,
        ],
        None,
    )


def row_reps(row: dict) -> int:
    return int(row.get("reps") or 1)


def cell_est(row: dict) -> float:
    """Estimated GPU minutes for ONE rep of the row (the manifest est includes all reps)."""
    return float(row["est_gpu_min"]) / row_reps(row)


def group_key(row: dict) -> tuple:
    return (row["engine"], row["model"], row.get("thinking") or "off")


def group_slug(row: dict) -> str:
    return f"{row['model']}-th-{row.get('thinking') or 'off'}"


def pack_chunks(rows: list[dict], cap: float = JOB_CAP_MIN) -> list[list[dict]]:
    """Consecutive chunks (in id order) whose summed per-rep estimate is <= cap. A single row above
    the cap gets its own chunk (the caller reports it)."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    total = 0.0
    for row in sorted(rows, key=lambda r: r["id"]):
        est = cell_est(row)
        if cur and total + est > cap:
            chunks.append(cur)
            cur, total = [], 0.0
        cur.append(row)
        total += est
    if cur:
        chunks.append(cur)
    return chunks


@dataclass
class Job:
    label: str
    batch: str
    engine: str
    group: str
    rep: int
    cells: list[dict]
    est_min: float
    cmd: list[str]  # argv after `gpuq submit ... --`
    over_cap: bool = False
    flags: list[str] = field(default_factory=list)
    script: str = ""  # job body, written to cmd[1] by write_wrapper before submit

    @property
    def cell_ids(self) -> list[str]:
        return [c["id"] for c in self.cells]

    def submit_argv(self) -> list[str]:
        return [
            GPUQ,
            "submit",
            "--label",
            self.label,
            "--priority",
            PRIORITY,
            "--quiet",
            "--",
            *self.cmd,
        ]


def cell_env(
    engine: str,
    pin: dict,
    label: str = "",
    telemetry_root=None,
    extra=(),
    group: str = "",
) -> list[str]:
    """Env assignments every harness command carries (pinned Yunshu tree, isolated HOME).
    TFB_TELEMETRY_DIR is the raw root: tfbench writes <root>/<engine>/<cell>-rN.telemetry.jsonl,
    the same place the audit reads it beside each cell file. GPUQ_JOB_ID is not set here: gpuq
    exports it to the current daemon's jobs only. TFB_JOB_LABEL (the label this job was
    submitted under) plus GPUQ_DIR let tfbench resolve the job id when it is missing."""
    harness = pin["launch"]["harness_env"]
    env = [
        "TFB_FULLCMP=1",
        f"TFB_TELEMETRY_DIR={telemetry_root or RAW_ROOT}",
        f"TFB_YUNSHU_BIN={harness['TFB_YUNSHU_BIN']}",
        f"TFB_YUNSHU_SRC={harness['TFB_YUNSHU_SRC']}",
        f"PYTHONPATH={harness['PYTHONPATH']}",
        f"HOME={HOME_ROOT / engine}",
        f"GPUQ_DIR={GPUQ_DIR_DEFAULT}",
        f"TFB_MLXLM_PY={pin['python_executable']}",
        f"TFB_OUT={TFOUT_ROOT}",
    ]
    if label:
        env.append(f"TFB_JOB_LABEL={label}")
    # tfbench fails closed without these (fullcmp_provenance.REQUIRED_ENV); the group is the
    # manifest cell group, which the audit matches against each header.
    env.append(f"TFB_RUN_ID={RUN_ID}")
    if group:
        env.append(f"TFB_GROUP={group}")
    return env + list(extra)


def raw_path(engine: str, cell_id: str, rep: int) -> Path:
    return RAW_ROOT / engine / f"{cell_id}-r{rep}.jsonl"


def profile_path(engine: str, cell_id: str) -> Path:
    """Layer-2 profile rows: outside <engine>/<cell>-rN.jsonl, so the audit never reads them."""
    return RAW_ROOT / "profile" / engine / f"{cell_id}-prof.jsonl"


def cell_command(
    row: dict, rep: int, pin: dict, label: str = "", profile: bool = False
) -> list[str] | None:
    part, extra, reason = harness_part(row)
    if part is None:
        return None
    if profile:
        out = profile_path(row["engine"], row["id"])
        env = cell_env(
            row["engine"],
            pin,
            label,
            telemetry_root=RAW_ROOT / "profile",
            extra=profile_env(row["engine"]),
            group=comparison_group(row),
        )
        rep = 0
    else:
        out = raw_path(row["engine"], row["id"], rep)
        env = cell_env(row["engine"], pin, label, group=comparison_group(row))
    return [
        "env",
        *env,
        HARNESS_PY,
        str(TFBENCH),
        "--engine",
        bench_engines.tfbench_engine(row["engine"]),
        "--part",
        part,
        "--rep",
        str(rep),
        "--out",
        str(out),
        "--model",
        MODEL_PATH,
        *extra,
    ]


def quiet_gate_argv(label: str) -> list[str]:
    """Quiet-machine gate run first inside the job: up to 3 retries 30 s apart, then the job
    fails naming the holders (fullcmp_quiet.gate). Its attempts are logged per job label."""
    return [
        HARNESS_PY,
        str(QUIET),
        "gate",
        "--log",
        str(QUIET_DIR / f"{label}.jsonl"),
    ]


def job_script(
    cells: list[dict], rep: int, pin: dict, label: str = "", profile: bool = False
) -> str:
    """Shell body of one gpuq job: quiet gate, then mkdir, then one tfbench invocation per cell
    (&&-chained, so a busy machine or a failed cell stops the rest of the job). A profile job
    appends the layer-2 row (fullcmp_profile.py) after each cell. The body is written to a
    wrapper file (write_wrapper) and gpuq runs `/bin/sh <wrapper>`: gpuq preflight treats every
    absolute argv element as an input path that must exist, so the gate never goes on the argv."""
    if profile:
        dirs = sorted({str(profile_path(c["engine"], c["id"]).parent) for c in cells})
    else:
        dirs = sorted({str(raw_path(c["engine"], c["id"], rep).parent) for c in cells})
    steps = [shlex.join(quiet_gate_argv(label or "unlabelled"))]
    steps.append("mkdir -p " + " ".join(shlex.quote(d) for d in dirs))
    for c in cells:
        steps.append(shlex.join(cell_command(c, rep, pin, label, profile=profile)))
        if profile:
            out = profile_path(c["engine"], c["id"])
            steps.append(
                shlex.join(
                    [
                        HARNESS_PY,
                        str(PROFILE_SCRIPT),
                        "--cell",
                        str(out),
                        "--engine",
                        c["engine"],
                    ]
                )
            )
    return "set -e\n" + " && ".join(steps) + "\n"


def wrapper_path(label: str) -> Path:
    return WRAP_ROOT / f"{label or 'unlabelled'}.sh"


def job_command(
    cells: list[dict], rep: int, pin: dict, label: str = "", profile: bool = False
) -> list[str]:
    """The argv a job runs: /bin/sh on its wrapper file (the body is job_script)."""
    return ["/bin/sh", str(wrapper_path(label))]


def write_wrapper(job: Job) -> Path:
    path = Path(job.cmd[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + job.script)
    path.chmod(0o755)
    return path


def _output_roots() -> tuple[str, ...]:
    return tuple(
        str(p)
        for p in (
            RAW_ROOT,
            HOME_ROOT,
            WRAP_ROOT,
            QUIET_DIR,
            PREFLIGHT_DIR,
            AUDIT_DIR,
            TFOUT_ROOT,
        )
    )


def script_problems(script: str) -> list[str]:
    """What gpuq preflight cannot see inside a job body: absolute inputs (the tfbench and gate
    scripts, the interpreter, --model, the pinned Yunshu tree) must exist, and no input may
    point into the main checkout. Output and home paths are created by the job, not checked."""
    problems: list[str] = []
    outs = _output_roots()
    for tok in shlex.split(script):
        for part in (
            tok.split(":") if tok.startswith(("PYTHONPATH=", "TFB_")) else [tok]
        ):
            value = part.partition("=")[2] if re.match(r"^[A-Z_]+=", part) else part
            if not value.startswith("/") or "{" in value or value.startswith(outs):
                continue
            if "/Yunshu/.venv" in value or value.startswith(
                "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/"
            ):
                problems.append(f"job references the main checkout: {value}")
            elif not os.path.exists(value):
                problems.append(f"input path does not exist: {value}")
    if "/Yunshu/.venv" in script or "/Documents/YuhuanStudio/Yunshu/" in script:
        problems.append("job body references the main checkout")
    return sorted(set(problems))


@dataclass
class Plan:
    batch: str
    runnable: list[dict]
    skipped: list[tuple[dict, str]]
    engine_order: list[str]
    jobs: list[Job]


def plan_jobs(
    rows_by_rep: dict[int, list[dict]],
    engine_order: list[str],
    pin: dict,
    batch: str,
    label_suffix: str = "",
    profile: bool = False,
) -> list[Job]:
    """Jobs for each rep in rows_by_rep (dict insertion order is the rep order). A profile plan
    takes one rep key (0) and labels each job ...-<part>-prof."""
    jobs: list[Job] = []
    for rep, rows in rows_by_rep.items():
        order = rotate(engine_order, rep)
        for engine in order:
            eng_rows = [r for r in rows if r["engine"] == engine]
            groups: dict[str, list[dict]] = {}
            for r in eng_rows:
                groups.setdefault(group_slug(r), []).append(r)
            for slug in sorted(groups):
                chunks = pack_chunks(groups[slug])
                for k, chunk in enumerate(chunks):
                    part_tag = f"-p{k + 1}" if len(chunks) > 1 else ""
                    est = sum(cell_est(c) for c in chunk)
                    over = est > JOB_CAP_MIN + 1e-9
                    if profile:
                        label = f"fullcmp-{batch}-{engine}-{slug}{part_tag}-prof"
                        flags = ["profile"]
                    else:
                        label = f"fullcmp-{batch}-{engine}-{slug}{part_tag}-r{rep}{label_suffix}"
                        flags = ["rep3"] if label_suffix else []
                    jobs.append(
                        Job(
                            label=label,
                            batch=batch,
                            engine=engine,
                            group=slug,
                            rep=rep,
                            cells=chunk,
                            est_min=est,
                            cmd=job_command(chunk, rep, pin, label, profile=profile),
                            script=job_script(chunk, rep, pin, label, profile=profile),
                            over_cap=over,
                            flags=flags,
                        )
                    )
    return jobs


def audited_batches(batch: str) -> list[str]:
    """Batches the audit covers for a run: a SMOKE run audits only its own rows; a real batch
    audits itself and every earlier batch in BATCH_ORDER."""
    if batch == SMOKE_BATCH:
        return [SMOKE_BATCH]
    return list(BATCH_ORDER[: BATCH_ORDER.index(batch) + 1])


def unknown_engines(rows: list[dict], batch: str, engines) -> list[str]:
    """Requested engine ids that have no manifest row in the batch."""
    known = {r.get("engine") for r in rows if r.get("batch") == batch}
    return [e for e in engines if e not in known]


def plan_batch(
    rows: list[dict],
    batch: str,
    pin: dict,
    engines=None,
    rep=None,
    profile_rep: bool = False,
) -> Plan:
    batch_rows = [r for r in rows if r.get("batch") == batch]
    runnable, skipped = [], []
    for r in batch_rows:
        if r.get("status") != "unverified":
            skipped.append((r, f"status {r.get('status')}: {r.get('reason')}"))
            continue
        stub = bench_engines.stub_reason(r["engine"])
        if stub is not None:
            # a harness stub (lmstudio: no isolated-HOME start yet): listed as skipped, never planned
            skipped.append((r, stub))
            continue
        part, _extra, reason = harness_part(r)
        if part is None:
            skipped.append((r, reason))
            continue
        if engines and r["engine"] not in engines:
            continue
        runnable.append(r)
    engine_order: list[str] = []
    for r in runnable:
        if r["engine"] not in engine_order:
            engine_order.append(r["engine"])
    max_reps = max((row_reps(r) for r in runnable), default=0)
    reps = [rep] if rep is not None else list(range(max_reps))
    rows_by_rep = {k: [r for r in runnable if k < row_reps(r)] for k in reps}
    rows_by_rep = {k: v for k, v in rows_by_rep.items() if v}
    jobs = plan_jobs(rows_by_rep, engine_order, pin, batch)
    if profile_rep and runnable:
        # One extra profile job per (engine, group, chunk), after the timed jobs; its rows never
        # reach the audit or the table.
        jobs += plan_jobs({0: runnable}, engine_order, pin, batch, profile=True)
    return Plan(batch, runnable, skipped, engine_order, jobs)


def back_to_back(jobs: list[Job]) -> int:
    """Consecutive jobs on the same engine across a rep boundary (reps of one cell adjacent)."""
    n = 0
    for a, b in zip(jobs, jobs[1:], strict=False):
        if a.engine == b.engine and a.rep != b.rep:
            n += 1
    return n


def parse_status(text: str) -> list[tuple[str, str]]:
    """(state, job id) for every active job in `gpuq status` output."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        state = parts[0]
        if state in ACTIVE_STATES or state.startswith("wait-"):
            ids = [p for p in parts if JOB_ID_RE.fullmatch(p)]
            if ids:
                out.append((state, ids[0]))
    return out


def foreign_jobs(status_text: str) -> list[tuple[str, str]]:
    """gpuq prints job ids as <seq>-<label>; our labels start with fullcmp-, so match the label."""
    return [(s, j) for s, j in parse_status(status_text) if "-fullcmp-" not in f"-{j}"]


# ---------------------------------------------------------------------------
# Audit manifest and reps
# ---------------------------------------------------------------------------


def comparison_group(row: dict) -> str:
    """Rows that share a group must share the prompt. A smoke row's prompt variant is part of the
    prompt, so each variant is its own group (regression, e2e 2026-11-03: variants 0 and 1 of one
    kind shared a group and the provenance guard reported their different prompts)."""
    base = f"{row['batch']}.{row['ctx']}.{row['prompt_kind']}.c{row['conc']}"
    # a thinking or image cell sends a different request (template switch, image parts): own group
    if row.get("thinking") not in (None, "off"):
        base += f".th-{row['thinking']}"
    if row.get("media") not in (None, "none"):
        base += f".{row['media']}"
    if "prompt_variant" in row:
        return f"{base}.v{row['prompt_variant']}"
    return base


def census_pins(pin: dict) -> dict:
    """Every engine's pin (fullcmp_audit_manifest.build_pins over census.json), so the audit
    finds a pin for each engine in the batch. Reads the census; fails closed if it is missing."""
    import fullcmp_audit_manifest as fam

    census = json.loads((RUN_DIR / "census.json").read_text())
    return fam.build_pins(census, pin)


def fam_phase_names() -> dict:
    """The audit's phase names for manifest phases (fullcmp_audit_manifest.PHASE_NAMES)."""
    import fullcmp_audit_manifest as fam

    return fam.PHASE_NAMES


def expected_records(row: dict) -> int | None:
    """Expected request-record rows per cell (all reps) for a smoke row, from the smoke manifest.
    Decode: one record per planned phase per rep (warm_reps is 1). Conc: CONC_TRIALS records per rep,
    each holding conc requests. Real-matrix rows return None: their counts are not filled here."""
    if "prompt_variant" not in row:
        return None
    reps = row_reps(row)
    if int(row["conc"]) > 1:
        return reps * int(row["conc"]) * CONC_TRIALS
    return reps * len(row.get("phases") or ())


def _expected_spec_mode(engine: str) -> str | None:
    try:
        hid = bench_engines.tfbench_engine(engine)
    except KeyError:
        return None
    return bench_engines.ENGINES[hid].expected_mode


def _weight_group(engine: str) -> str | None:
    import fullcmp_audit_manifest as fam

    return fam.WEIGHT_GROUP.get(engine)


def audit_manifest(
    rows: list[dict], batches: list[str], pin: dict, engine_order: list[str]
) -> dict:
    """Expected-cell manifest for fullcmp_audit (its schema, FULLCMP_SCHEMA["manifest"]).
    Fields the driver cannot know (per-engine pins other than yunshu, controls, expected records,
    spec mode, weight groups) are left out or null, so the audit fails closed on them."""
    cells, groups = [], {}
    for r in rows:
        if r.get("batch") not in batches or r.get("status") != "unverified":
            continue
        if harness_part(r)[0] is None:
            continue
        grp = comparison_group(r)
        cells.append(
            {
                "engine": r["engine"],
                "model": r["model"],
                "column": f"{r['ctx']}/{r['prompt_kind']}/c{r['conc']}/o{r['out_len']}",
                "cell": r["id"],
                "group": grp,
                "ctx": r["ctx_tokens"],
                "kind": r["prompt_kind"],
                "thinking": r.get("thinking") or "off",
                "media": r.get("media") or "none",
                "prompt_id": prompt_id(r),
                "max_tokens": r["out_len"],
                # manifest phase names -> the audit's (fu -> turn2): the records carry the audit's
                # a concurrent cell records one record per trial: its planned phase is the record
                # part itself (conc, or mixed for the mixed-load cell), not the manifest's phases
                "phases": (
                    ["mixed" if r.get("turns") == "mixed" else "conc"]
                    if int(r["conc"]) > 1
                    else [fam_phase_names().get(p, p) for p in r.get("phases") or ()]
                ),
                "expected_records": expected_records(r),
                # Same fields fullcmp_audit_manifest.build_manifest writes: the engine's expected
                # spec mode (bench_engines) and its weight group. Null only when the engine
                # declares none, and the audit then reports it (fail closed).
                "spec_mode": _expected_spec_mode(r["engine"]),
                "weight_group": _weight_group(r["engine"]),
            }
        )
        groups.setdefault(grp, [])
        if r["engine"] not in groups[grp]:
            groups[grp].append(r["engine"])
    ordered = {
        g: {"order": [e for e in engine_order if e in es]} for g, es in groups.items()
    }
    return {
        "run_id": RUN_ID,
        # A smoke manifest is a harness check, not a table: the audit then skips the A/A and
        # injected-delay controls (fullcmp_audit.check_fairness_controls). Real batches never set it.
        "smoke": any(b == "SMOKE" for b in batches),
        "reps": max(
            (row_reps(r) for r in rows if r.get("batch") in batches), default=1
        ),
        "pins": census_pins(pin),
        "yunshu": {"git_sha": pin["sha"], "python": pin["python_executable"]},
        "controls": {},
        "groups": ordered,
        "cells": cells,
        "behaviour": behaviour_expectations(rows, batches),
    }


def behaviour_expectations(rows: list[dict], batches: list[str]) -> dict:
    """B4 expectations for fullcmp_audit.check_behaviour: {engine: {check: status}}. A row the
    census marks unsupported expects "unsupported" (the route must answer 404); an unverified one
    expects "pass". Rows of other batches are left out."""
    out: dict = {}
    for r in rows:
        if r.get("table") != "behaviour" or r.get("batch") not in batches:
            continue
        want = "unsupported" if r.get("status") == "unsupported" else "pass"
        out.setdefault(r["engine"], {})[r["check"]] = want
    return out


def parse_need_rep3(stdout: str) -> list[tuple[str, str]]:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        if rec.get("status") == "need_rep3":
            out.append((rec["engine"], rec["cell"]))
    return out


# ---------------------------------------------------------------------------
# Side effects (only in run / preflight phases)
# ---------------------------------------------------------------------------


def gpuq_env() -> dict:
    env = dict(os.environ)
    env.setdefault("GPUQ_DIR", GPUQ_DIR_DEFAULT)
    return env


def run_capture(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, env=gpuq_env())


def gpuq_status_text() -> str:
    return run_capture([GPUQ, "status"]).stdout


def swap_gate(batch: str) -> int:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    out = PREFLIGHT_DIR / f"swap_{batch}.json"
    argv = [
        GPUQ,
        "run",
        "--short",
        "--serving-ok",
        "--timeout",
        PREFLIGHT_TIMEOUT_MIN,
        "--label",
        f"fullcmp-preflight-{batch}",
        "--",
        "env",
        f"PYTHONPATH={SCRIPTS.parent.parent / 'python'}",
        HARNESS_PY,
        str(PREFLIGHT),
        "--out",
        str(out),
    ]
    print(f"preflight swap_stable: {shlex.join(argv)}", flush=True)
    return subprocess.run(argv, env=gpuq_env()).returncode


def self_check(jobs: list[Job]) -> dict[str, list[str]]:
    """CPU only, no queue: writes each job's wrapper and runs gpuq's own preflight on the argv
    that gpuq submit receives (scripts/dev/gpuq_preflight.preflight), plus the job body check.
    Returns {label: problems}; an empty dict means every job would pass gpuq's preflight."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
    import gpuq_preflight

    bad: dict[str, list[str]] = {}
    for job in jobs:
        write_wrapper(job)
        problems = gpuq_preflight.preflight(job.cmd, str(REPO_ROOT), dict(os.environ))
        problems += script_problems(job.script)
        if problems:
            bad[job.label] = problems
    return bad


def submit_all(jobs: list[Job]) -> tuple[list[str], int]:
    ids, bad = [], 0
    for job in jobs:
        write_wrapper(job)
        res = subprocess.run(
            job.submit_argv(),
            capture_output=True,
            text=True,
            env=gpuq_env(),
            cwd=str(REPO_ROOT),
        )
        if res.returncode != 0 or not res.stdout.strip():
            print(f"submit failed: {job.label}: {res.stderr.strip()}", file=sys.stderr)
            bad += 1
            continue
        jid = res.stdout.strip().splitlines()[-1]
        ids.append(jid)
        print(
            f"submitted {jid} ({job.est_min:.1f} min, {len(job.cells)} cells)",
            flush=True,
        )
    return ids, bad


def wait_all(ids: list[str]) -> int:
    if not ids:
        return 0
    res = subprocess.run([GPUQ, "wait", *ids], env=gpuq_env())
    return res.returncode


def run_audit(raw_root: Path, manifest_path: Path) -> int:
    return subprocess.run(
        [
            HARNESS_PY,
            str(AUDIT),
            "--root",
            str(raw_root),
            "--manifest",
            str(manifest_path),
        ],
        env=gpuq_env(),
    ).returncode


def run_reps(raw_root: Path, manifest_path: Path) -> list[tuple[str, str]]:
    res = subprocess.run(
        [
            HARNESS_PY,
            str(REPS),
            "--results-dir",
            str(raw_root),
            "--thresholds",
            str(THRESH_PATH),
            "--manifest",
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        env=gpuq_env(),
    )
    print(res.stderr, end="", file=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"fullcmp_reps failed (rc {res.returncode})")
    return parse_need_rep3(res.stdout)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_plan(plan: Plan) -> None:
    print(
        f"batch {plan.batch}: {len(plan.runnable)} runnable rows, {len(plan.skipped)} skipped"
    )
    print(f"engine rotation order (rep 0): {', '.join(plan.engine_order)}")
    for row, reason in plan.skipped:
        print(f"SKIP {row['id']}: {reason}")
    for job in plan.jobs:
        flags = " [over-cap: single cell above 20 min]" if job.over_cap else ""
        print(f"JOB {job.label}{flags}")
        print(f"  group: {job.engine}/{job.group} rep {job.rep}")
        print(f"  cells: {' '.join(job.cell_ids)}")
        print(f"  est_min: {job.est_min:.2f}")
        print(f"  command: {shlex.join(job.submit_argv())}")
    total = sum(j.est_min for j in plan.jobs)
    print(
        f"TOTAL {plan.batch}: {len(plan.jobs)} jobs, {total:.1f} min = {total / 60:.2f} h "
        f"(back-to-back same-engine across rep boundaries: {back_to_back(plan.jobs)})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def configure(raw_root: Path | None = None, work_dir: Path | None = None) -> None:
    """Point the driver at a separate raw root and work dir (smoke runs). Defaults are the
    production paths; a smoke run must never touch RAW_ROOT or the batch audit manifests."""
    global \
        RAW_ROOT, \
        HOME_ROOT, \
        AUDIT_DIR, \
        QUIET_DIR, \
        PREFLIGHT_DIR, \
        WRAP_ROOT, \
        TFOUT_ROOT
    if raw_root is not None:
        RAW_ROOT = Path(raw_root)
    if work_dir is not None:
        work = Path(work_dir)
        HOME_ROOT = work / "home"
        AUDIT_DIR = work / "audit"
        QUIET_DIR = work / "quiet"
        PREFLIGHT_DIR = work / "preflight"
        WRAP_ROOT = work / "wrap"
        TFOUT_ROOT = work / "tfout"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", required=True, choices=BATCH_CHOICES)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--engines", default="", help="comma list of manifest engine ids")
    ap.add_argument(
        "--rep", type=int, default=None, help="run only this rep index (0-based)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print every job; touch no GPU, no queue"
    )
    ap.add_argument("--phase", choices=("preflight", "run"), default="run")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="no GPU, no queue: run gpuq's preflight on every job's submit argv and job body",
    )
    ap.add_argument(
        "--allow-foreign",
        action="store_true",
        help="submit even with a non-fullcmp GPU job",
    )
    ap.add_argument(
        "--profile-rep",
        action="store_true",
        help="also queue one layer-2 profile job per cell group (label ...-prof; excluded from tables)",
    )
    ap.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="raw results root (default: the production root; smoke runs use their own)",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="work dir for homes, audit manifests, quiet and preflight logs (default: production)",
    )
    a = ap.parse_args(argv)
    configure(a.raw_root, a.work_dir)

    pin = json.loads(PIN_PATH.read_text())
    rows = load_rows(a.manifest)
    engines = [e for e in a.engines.split(",") if e] or None
    if engines:
        # An unknown id would filter every row out and run nothing: fail closed instead.
        missing = unknown_engines(rows, a.batch, engines)
        if missing:
            print(
                f"refusing: no {a.batch} manifest row for engine id(s) {missing} "
                f"(manifest ids: {sorted({r.get('engine') for r in rows if r.get('batch') == a.batch})})",
                file=sys.stderr,
            )
            return 2
    plan = plan_batch(
        rows, a.batch, pin, engines=engines, rep=a.rep, profile_rep=a.profile_rep
    )

    if a.dry_run or a.self_check:
        if a.self_check:
            bad = self_check(plan.jobs)
            for label, problems in bad.items():
                for p in problems:
                    print(f"SELF-CHECK {label}: {p}", file=sys.stderr)
            print(
                f"self-check: {len(plan.jobs) - len(bad)}/{len(plan.jobs)} jobs pass gpuq preflight",
                flush=True,
            )
            return 1 if bad else 0
        print("dry-run: no preflight, no queue read, no submission")
        print_plan(plan)
        return 0

    if a.phase == "preflight":
        return swap_gate(a.batch)

    if not a.allow_foreign:
        foreign = foreign_jobs(gpuq_status_text())
        if foreign:
            for state, jid in foreign:
                print(f"foreign GPU job {state}: {jid}", file=sys.stderr)
            print(
                "refusing: exclusive GPU rule (use --allow-foreign to override)",
                file=sys.stderr,
            )
            return 2

    if any(j.over_cap for j in plan.jobs):
        print("refusing: a single cell exceeds the 20 min job cap", file=sys.stderr)
        return 2
    # Quiet machine before the batch (each job gates again before its cells).
    QUIET_DIR.mkdir(parents=True, exist_ok=True)
    quiet = fullcmp_quiet.gate(log=QUIET_DIR / f"{a.batch}_batch.jsonl")
    if not quiet["passed"]:
        print(
            f"refusing: host busy before batch {a.batch}: foreign CPU "
            f"{quiet['total_foreign_pct']:.1f}%; holders: {fullcmp_quiet.describe_holders(quiet)}",
            file=sys.stderr,
        )
        return 2
    if swap_gate(a.batch) != 0:
        print("refusing: swap_stable preflight failed", file=sys.stderr)
        return 2

    ids, bad = submit_all(plan.jobs)
    failed = bad + (1 if wait_all(ids) != 0 else 0)

    batches = audited_batches(a.batch)
    manifest_path = AUDIT_DIR / f"{a.batch}_audit_manifest.json"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(audit_manifest(rows, batches, pin, plan.engine_order), indent=2)
        + "\n"
    )
    audit_rc = run_audit(RAW_ROOT, manifest_path)

    need = run_reps(RAW_ROOT, manifest_path)
    print(f"rep 3 needed for {len(need)} cells", flush=True)
    rep3_bad = 0
    if need:
        want = set(need)
        subset = [r for r in plan.runnable if (r["engine"], r["id"]) in want]
        rep3_jobs = plan_jobs(
            {REP3_INDEX: subset}, plan.engine_order, pin, a.batch, label_suffix="-rep3"
        )
        ids3, rep3_bad = submit_all(rep3_jobs)
        rep3_bad += 1 if wait_all(ids3) != 0 else 0
        audit_rc = run_audit(RAW_ROOT, manifest_path) or audit_rc

    if failed or audit_rc or rep3_bad:
        print(
            f"{a.batch}: failed jobs {failed}, audit rc {audit_rc}, rep-3 failures {rep3_bad}",
            file=sys.stderr,
        )
        return 1
    print(f"{a.batch}: {len(ids)} jobs done, audit clean", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
