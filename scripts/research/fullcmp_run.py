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
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_engines  # noqa: E402
import fullcmp_quiet  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
TFBENCH = SCRIPTS / "tfbench.py"
AUDIT = SCRIPTS / "fullcmp_audit.py"
REPS = SCRIPTS / "fullcmp_reps.py"
PREFLIGHT = SCRIPTS / "fullcmp_preflight.py"
QUIET = SCRIPTS / "fullcmp_quiet.py"

MAIN_PY = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
GPUQ = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq"
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
JOB_CAP_MIN = 20.0
REP3_INDEX = 2  # same as fullcmp_reps.REP3_INDEX
CONC_TRIALS = (
    2  # conc records per cell per rep (tfbench --conc-trials); each holds conc requests
)
PRIORITY = "1"
RUN_ID = "fullcmp-1103"
ACTIVE_STATES = ("pending", "running", "paused")
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


def harness_part(row: dict):
    """(tfbench part, extra args, None) for a row tfbench can run, else (None, None, reason)."""
    if row.get("table") == "agent":
        return None, None, "agent task selection not in tfbench CLI"
    if row.get("table") == "behaviour":
        return None, None, "behaviour checks not in tfbench CLI"
    if row.get("media") not in (None, "none"):
        return None, None, "image input not in tfbench CLI"
    if row.get("thinking") not in (None, "off"):
        return None, None, "thinking switch not in tfbench CLI"
    if row.get("sampling") not in (None, "greedy"):
        return None, None, "generation_config sampling not in tfbench CLI"
    if row.get("turns") not in (None, "single"):
        return None, None, f"turns={row.get('turns')} not in tfbench CLI"
    kind = row.get("prompt_kind")
    if kind not in ("code", "prose"):
        return None, None, f"prompt kind {kind} not in tfbench CLI"
    conc = int(row["conc"])
    if conc == 1:
        return (
            "decode",
            [
                "--only-ctx",
                str(row["ctx_tokens"]),
                "--only-kind",
                kind,
                "--decode-tokens",
                str(row["out_len"]),
            ],
            None,
        )
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


def cell_env(engine: str, pin: dict, label: str = "") -> list[str]:
    """Env assignments every harness command carries (pinned Yunshu tree, isolated HOME).
    TFB_TELEMETRY_DIR is the raw root: tfbench writes <root>/<engine>/<cell>-rN.telemetry.jsonl,
    the same place the audit reads it beside each cell file. GPUQ_JOB_ID is not set here: gpuq
    exports it to the current daemon's jobs only. TFB_JOB_LABEL (the label this job was
    submitted under) plus GPUQ_DIR let tfbench resolve the job id when it is missing."""
    harness = pin["launch"]["harness_env"]
    env = [
        "TFB_FULLCMP=1",
        f"TFB_TELEMETRY_DIR={RAW_ROOT}",
        f"TFB_YUNSHU_BIN={harness['TFB_YUNSHU_BIN']}",
        f"TFB_YUNSHU_SRC={harness['TFB_YUNSHU_SRC']}",
        f"PYTHONPATH={harness['PYTHONPATH']}",
        f"HOME={HOME_ROOT / engine}",
        f"GPUQ_DIR={GPUQ_DIR_DEFAULT}",
    ]
    if label:
        env.append(f"TFB_JOB_LABEL={label}")
    return env


def raw_path(engine: str, cell_id: str, rep: int) -> Path:
    return RAW_ROOT / engine / f"{cell_id}-r{rep}.jsonl"


def cell_command(row: dict, rep: int, pin: dict, label: str = "") -> list[str] | None:
    part, extra, reason = harness_part(row)
    if part is None:
        return None
    out = raw_path(row["engine"], row["id"], rep)
    return [
        "env",
        *cell_env(row["engine"], pin, label),
        MAIN_PY,
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
        MAIN_PY,
        str(QUIET),
        "gate",
        "--log",
        str(QUIET_DIR / f"{label}.jsonl"),
    ]


def job_command(cells: list[dict], rep: int, pin: dict, label: str = "") -> list[str]:
    """One gpuq job: quiet gate, then mkdir, then one tfbench invocation per cell (&&-chained,
    so a busy machine or a failed cell stops the rest of the job)."""
    dirs = sorted({str(raw_path(c["engine"], c["id"], rep).parent) for c in cells})
    steps = [shlex.join(quiet_gate_argv(label or "unlabelled"))]
    steps.append("mkdir -p " + " ".join(shlex.quote(d) for d in dirs))
    for c in cells:
        steps.append(shlex.join(cell_command(c, rep, pin, label)))
    return ["/bin/sh", "-c", " && ".join(steps)]


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
) -> list[Job]:
    """Jobs for each rep in rows_by_rep (dict insertion order is the rep order)."""
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
                    label = f"fullcmp-{batch}-{engine}-{slug}{part_tag}-r{rep}{label_suffix}"
                    jobs.append(
                        Job(
                            label=label,
                            batch=batch,
                            engine=engine,
                            group=slug,
                            rep=rep,
                            cells=chunk,
                            est_min=est,
                            cmd=job_command(chunk, rep, pin, label),
                            over_cap=over,
                            flags=["rep3"] if label_suffix else [],
                        )
                    )
    return jobs


def plan_batch(rows: list[dict], batch: str, pin: dict, engines=None, rep=None) -> Plan:
    batch_rows = [r for r in rows if r.get("batch") == batch]
    runnable, skipped = [], []
    for r in batch_rows:
        if r.get("status") != "unverified":
            skipped.append((r, f"status {r.get('status')}: {r.get('reason')}"))
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
            out.append((state, parts[-1]))
    return out


def foreign_jobs(status_text: str) -> list[tuple[str, str]]:
    return [
        (s, j) for s, j in parse_status(status_text) if not j.startswith("fullcmp-")
    ]


# ---------------------------------------------------------------------------
# Audit manifest and reps
# ---------------------------------------------------------------------------


def comparison_group(row: dict) -> str:
    return f"{row['batch']}.{row['ctx']}.{row['prompt_kind']}.c{row['conc']}"


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
                "prompt_id": f"{r['prompt_kind']}-{r['ctx_tokens']}",
                "max_tokens": r["out_len"],
                "phases": r.get("phases"),
                "expected_records": None,
                "spec_mode": None,
                "weight_group": None,
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
        "reps": 2,
        "pins": {"yunshu": {"version": pin["version"], "git_sha": pin["sha"]}},
        "yunshu": {"git_sha": pin["sha"], "python": pin["python_executable"]},
        "controls": {},
        "groups": ordered,
        "cells": cells,
    }


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
        MAIN_PY,
        str(PREFLIGHT),
        "--out",
        str(out),
    ]
    print(f"preflight swap_stable: {shlex.join(argv)}", flush=True)
    return subprocess.run(argv, env=gpuq_env()).returncode


def submit_all(jobs: list[Job]) -> tuple[list[str], int]:
    ids, bad = [], 0
    for job in jobs:
        res = subprocess.run(
            job.submit_argv(), capture_output=True, text=True, env=gpuq_env()
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
            MAIN_PY,
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
            MAIN_PY,
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", required=True, choices=BATCH_ORDER)
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
        "--allow-foreign",
        action="store_true",
        help="submit even with a non-fullcmp GPU job",
    )
    a = ap.parse_args(argv)

    pin = json.loads(PIN_PATH.read_text())
    rows = load_rows(a.manifest)
    engines = [e for e in a.engines.split(",") if e] or None
    plan = plan_batch(rows, a.batch, pin, engines=engines, rep=a.rep)

    if a.dry_run:
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

    batches = list(BATCH_ORDER[: BATCH_ORDER.index(a.batch) + 1])
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
