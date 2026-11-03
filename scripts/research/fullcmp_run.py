#!/usr/bin/env python3
"""Batch driver for the full fair comparison (FULLCMP_SPEC.md, user 2026-11-03).

Reads manifest rows (manifest.jsonl, written by fullcmp_manifest.py) for one batch and turns them
into gpuq jobs. Only status "unverified" rows are runnable (costed, census gives no negative
evidence); "unsupported" and "cut" rows are skipped and listed. A row also needs a tfbench part
(see harness_part) or it is skipped and listed with the reason.

Grouping: one server configuration = (engine, model, thinking). Each group becomes one job per rep
when its estimate fits the planning limit (70% of the 20 min cap = 14 min, JOB_LIMIT_MIN, including
job_budget per job: gate, startup, cells, post, finalize, shutdown); a larger group is split into consecutive chunks that each fit. A cell is one
tfbench invocation with its own server start, so cells of a chunk run back to back in one job, each
cold-starting its server. A single cell above the 14 min limit gets a job of its own, flagged
over_margin (it still runs; above 20 min it is refused). Each job's gpuq --timeout is est x 1.3 (floor
1.3 x the itemized total, floored by the longest same-kind run, capped at 20 min); see job_budget.

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
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_engines  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
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
# One prompt corpus for every batch: tfbench reads TFB_WORK/prompts/<kind>-<ctx>.txt in every cell.
TFB_WORK_DIR = "/Volumes/P5Plus/yunshu-build/snapshot014-b2"

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
JOB_CAP_MIN = 20.0  # gpuq job cap (default --timeout); a job above it is killed
JOB_MARGIN = 0.7  # planned chunks stay <= 70% of the cap (>= 30% margin for estimate error)
JOB_LIMIT_MIN = JOB_CAP_MIN * JOB_MARGIN  # 14 min: the chunk packing limit
JB = fm.TIMING["job_budget"]  # itemized per-job budget terms (fullcmp_cell_timing.json)
TIMEOUT_FACTOR = float(JB["margin_factor"]["value"])  # timeout = 1.3 x itemized total (>= 30% margin)
GATE_PRE_MIN = (
    float(JB["gate_retries"]["value"])
    * (float(JB["gate_wait_s"]["value"]) + float(JB["window_s"]["value"]))
    / 60.0
)  # quiet pre-gate, worst case: every attempt waits and checks
POST_MIN = float(JB["window_s"]["value"]) / 60.0  # one quiet_post snapshot per cell
FINALIZE_MIN = float(JB["finalize_s"]["value"]) / 60.0  # telemetry/sampler/macmon stop per cell
SHUTDOWN_MIN = fm.OVERHEAD_S / 60.0  # memory idle + shutdown per cell (measured)
STALL_FLOOR_MIN = float(JB["stall_floor_min"]["value"])
SHORT_LANE_MIN = float(JB["short_lane_min"]["value"])  # gpuq: timeout <= this is the short lane
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
    if part is not None and row.get("batch") == SMOKE_BATCH:
        # the smoke's usage-shape evidence: the first raw usage chunk lands in the session header
        extra = [*(extra or []), "--log-usage-sample"]
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
    # both names: the real matrix's conc-same sweep and the --smoke-conc same-prompt sweep
    same = [] if row.get("sweep") not in ("conc-same", "smoke-conc-same") else ["--conc-same"]
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


def startup_s(engine: str) -> float:
    """Model-load and ready budget of one server start, seconds: max(1.5 x the longest ready_s seen on
    disk, the engine default). The default is a floor: a header on disk can be too early (a killed
    job never writes one)."""
    defaults = JB["startup_default_s"]["value"]
    floor = float(defaults.get(engine, defaults["default"]))
    seen = JB["startup_seen_s"].get(engine)
    if seen:
        return max(float(JB["startup_factor"]["value"]) * float(seen["value"]), floor)
    return floor


# Engines whose mixed (one long + short concurrent) cell can run silent for minutes: a stall limit of
# MIXED_SLOW_STALL_MIN applies to their mixed jobs even with heartbeats (defense in depth).
MIXED_SLOW_ENGINES = ("mlx-vlm-server", "splash-1.5.0", "tf-1.1.8")
MIXED_SLOW_STALL_MIN = 10.0
MIXED_PROBE_MIN = float(JB["mixed_probe_s"]["value"]) / 60.0  # the concurrent probe of a mixed cell


def cell_budget(row: dict) -> dict:
    """Itemized minutes of one cell for one rep: its own server start, the modelled work (the
    manifest estimate minus the manifest's own boot and idle, so nothing is counted twice), the
    memory idle and shutdown, the quiet_post snapshot and the finalize."""
    ready = fm.ENGINE_PERF[row["engine"]]["ready_s"]
    work = max(0.0, cell_est(row) - (ready + fm.OVERHEAD_S) / 60.0)
    if row.get("turns") == "mixed":
        work += MIXED_PROBE_MIN  # the probe repeats the N concurrent requests, long prefill included
    return {
        "startup": startup_s(row["engine"]) / 60.0,
        "cells": work,
        "shutdown": SHUTDOWN_MIN,
        "post": POST_MIN,
        "finalize": FINALIZE_MIN,
    }


def job_budget(rows: list[dict], prior_min: float = 0.0) -> dict:
    """One job's itemized budget (minutes, one rep). total = gate_pre + every cell's terms.
    timeout = 1.3 x total, never below 1.3 x prior_min (the longest same-kind run), capped at the
    20 min cap. stall = silence limit: >= 5 min, >= 2 x startup, >= 2 x the gate windows. lane is
    the short lane only when gpuq would classify the timeout as short (<= 10 min), which by
    construction means total with the 30% margin is under 10 min."""
    parts = [cell_budget(r) for r in rows]
    out = {k: sum(p[k] for p in parts) for k in ("startup", "cells", "shutdown", "post", "finalize")}
    out["gate_pre"] = GATE_PRE_MIN
    out["total"] = GATE_PRE_MIN + sum(out[k] for k in ("startup", "cells", "shutdown", "post", "finalize"))
    want = TIMEOUT_FACTOR * max(out["total"], prior_min)
    out["timeout"] = min(JOB_CAP_MIN, math.ceil(round(want * 10, 6)) / 10)  # 0.1 min steps
    out["stall"] = max(
        STALL_FLOOR_MIN,
        2.0 * startup_s(rows[0]["engine"]) / 60.0,
        2.0 * GATE_PRE_MIN,
    )
    if any(r.get("turns") == "mixed" for r in rows) and rows[0]["engine"] in MIXED_SLOW_ENGINES:
        out["stall"] = max(out["stall"], MIXED_SLOW_STALL_MIN)
    out["lane"] = "short" if out["timeout"] <= SHORT_LANE_MIN else "normal"
    out["n_cells"] = len(rows)
    return out


def pack_chunks(rows: list[dict], cap: float = JOB_LIMIT_MIN) -> list[list[dict]]:
    """Consecutive chunks (in id order) whose job_budget total is <= cap, the 70% planning limit by
    default. Cells of one group share a job, not a server: each cell is its own tfbench invocation
    with its own server start (budgeted per cell). A single row above the limit gets its own chunk
    (the caller reports it: over_margin)."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for row in sorted(rows, key=lambda r: r["id"]):
        if cur and job_budget(cur + [row])["total"] > cap:
            chunks.append(cur)
            cur = []
        cur.append(row)
    if cur:
        chunks.append(cur)
    return chunks


def prior_wall_min(batch: str, engine: str) -> float:
    """Longest measured wall of an earlier job of the same batch and engine (fm.TIMING history)."""
    return max(
        (
            float(h["wall_min"]["value"])
            for h in fm.TIMING["job"]["history"]
            if h["batch"] == batch and h["engine"] == engine
        ),
        default=0.0,
    )


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
    over_cap: bool = False  # est above the 20 min cap: refused
    over_margin: bool = False  # est above the 70% planning limit: a single cell, run alone
    timeout_min: float = JOB_CAP_MIN
    stall_min: float = STALL_FLOOR_MIN
    lane: str = "normal"
    budget: dict = field(default_factory=dict)
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
            "--timeout",
            f"{self.timeout_min:g}",
            "--stall",
            f"{self.stall_min:g}",
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
    profile: bool = False,
) -> list[str]:
    """Env assignments every harness command carries (pinned Yunshu tree, isolated HOME).
    Observe is decided here, never left to tfbench's default: a timed cell exports TFB_OBSERVE=0
    (YUNSHU_OBSERVE off); only a profile cell exports TFB_OBSERVE=1 with TFB_PROFILE=1.
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
        f"TFB_WORK={TFB_WORK_DIR}",
        f"TFB_MLXLM_PY={pin['python_executable']}",
        f"TFB_OUT={TFOUT_ROOT}",
        f"TFB_OBSERVE={'1' if profile else '0'}",
        f"TFB_HARNESS_COMMIT={HARNESS_COMMIT}",
        f"TFB_HARNESS_TREE={HARNESS_TREE}",
    ]
    if profile:
        env.append("TFB_PROFILE=1")
    if label:
        env.append(f"TFB_JOB_LABEL={label}")
    # tfbench fails closed without these (fullcmp_provenance.REQUIRED_ENV); the group is the
    # manifest cell group, which the audit matches against each header.
    env.append(f"TFB_RUN_ID={RUN_ID}")
    if group:
        env.append(f"TFB_GROUP={group}")
    return env + list(extra)


PROMPT_VARIANT_DIR = Path("/Volumes/Micron/yunshu-fullcmp/smoke-prompts")  # smoke variants (N > 0)
PROMPT_SHA_TABLE = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/prompt_files.sha256.txt"
)


def _sha_table(path: Path = PROMPT_SHA_TABLE) -> dict[str, str]:
    """name -> sha256 from the prompt table ('<sha>  <name>' lines, sha256sum format)."""
    table: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                sha, name = line.split(maxsplit=1)
                table[name.strip()] = sha
    return table


def prompt_corpus_problems(
    rows: list[dict],
    work_dir: str = TFB_WORK_DIR,
    variant_dir: Path = PROMPT_VARIANT_DIR,
    sha_table: dict[str, str] | None = None,
) -> list[str]:
    """Fail closed before any job is written. Every (kind, ctx, variant) the given cells read must have
    its prompt file (variant 0: the TFB_WORK corpus; variant N: the smoke variant dir) and a sha256 row in
    the prompt table that matches the file. Family kinds read the prose file, as tfbench does.
    Each problem names the (kind, ctx, variant). An empty list means every cell's prompt is present."""
    prompts = Path(work_dir) / "prompts"
    if not prompts.is_dir():
        return [f"TFB_WORK prompt dir missing: {prompts}"]
    table = _sha_table() if sha_table is None else sha_table
    need: dict[str, tuple[str, int, int]] = {}
    for r in rows:
        if "ctx_tokens" not in r or not r.get("prompt_kind"):
            continue  # behaviour / agent rows carry no decode prompt
        base = "prose" if r["prompt_kind"] in fm.FAMILY_KINDS else r["prompt_kind"]
        n = prompt_variant(r)
        name = f"{base}-{int(r['ctx_tokens'])}" + (f"-v{n}" if n else "") + ".txt"
        need.setdefault(name, (base, int(r["ctx_tokens"]), n))
    problems = []
    for name in sorted(need):
        kind, ctx, n = need[name]
        where = "" if not n else f" (kind={kind} ctx={ctx} variant={n})"
        path = (prompts if n == 0 else variant_dir) / name
        if not path.is_file():
            problems.append(f"missing prompt file {path}{where}")
            continue
        if name not in table:
            problems.append(f"no sha256 row for {name} in {PROMPT_SHA_TABLE}{where}")
            continue
        import hashlib  # noqa: PLC0415

        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != table[name]:
            problems.append(f"sha256 mismatch for {path}: table {table[name]} file {got}{where}")
    return problems


def raw_path(engine: str, cell_id: str, rep: int) -> Path:
    return RAW_ROOT / engine / f"{cell_id}-r{rep}.jsonl"


def profile_path(engine: str, cell_id: str) -> Path:
    """Layer-2 profile rows: outside <engine>/<cell>-rN.jsonl, so the audit never reads them."""
    return RAW_ROOT / "profile" / engine / f"{cell_id}-prof.jsonl"


LOG_USAGE_SAMPLE = False  # --log-usage-sample: tfbench keeps the first raw usage chunk (smokes)


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
            profile=True,
        )
        rep = 0
    else:
        out = raw_path(row["engine"], row["id"], rep)
        env = cell_env(
            row["engine"], pin, label, group=comparison_group(row), profile=False
        )
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
        *(["--log-usage-sample"] if LOG_USAGE_SAMPLE and not profile else []),
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
                    budget = job_budget(chunk, prior_wall_min(batch, engine))
                    est = budget["total"]
                    over = est > JOB_CAP_MIN + 1e-9
                    margin = est > JOB_LIMIT_MIN + 1e-9
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
                            over_margin=margin,
                            timeout_min=budget["timeout"],
                            stall_min=budget["stall"],
                            lane=budget["lane"],
                            budget=budget,
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


_GPUQ_SHORT_LABEL = re.compile(r"(^|[-_])(tiny|smoke|dry)([-_]|$)", re.I)


def one_lane(jobs: list[Job]) -> list[Job]:
    """One gpuq lane for every job of a batch. gpuq picks a pending job by priority, then SHORT
    jobs first (timeout <= 10 min), then owner round-robin, then submission order (gpuq.py _pick),
    and it runs short jobs between long ones (_interleave_choice). A batch that mixes lanes is
    therefore reordered (2026-11-03 wire-1106: gpu_exclusive_rotation). If any job is normal, every
    job is normal: its timeout is raised above the 10 min short limit, so gpuq keeps submission
    order (one owner, one priority, same lane = FIFO). Fails loudly if the lanes still differ."""
    if any(j.lane == "normal" for j in jobs):
        for j in jobs:
            j.lane = "normal"
            j.timeout_min = max(j.timeout_min, SHORT_LANE_MIN + 0.5)
    lanes = {j.lane for j in jobs}
    if len(lanes) > 1:
        raise ValueError(f"batch mixes gpuq lanes {sorted(lanes)}: submission order would not hold")
    # gpuq also treats a label with a tiny/smoke/dry word as short (gpuq.py _SHORT, _is_short)
    label_short = {bool(_GPUQ_SHORT_LABEL.search(j.label)) for j in jobs}
    if len(label_short) > 1:
        raise ValueError("batch mixes short-named and normal-named labels: gpuq would reorder it")
    return jobs


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
    jobs = one_lane(jobs)
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


def expected_records(row: dict) -> int:
    """Expected request-record rows in ONE rep's file (the audit checks each rep file on its own).
    Decode: one record per planned phase. Conc (and mixed, conc 8 = one long + 7 short): CONC_TRIALS
    records, each holding conc requests. Real-matrix rows get the same counts: they returned None
    before 2026-11-03, and the record_count guard then failed every real cell ('expected None')."""
    if int(row["conc"]) > 1:
        return int(row["conc"]) * CONC_TRIALS
    return len(row.get("phases") or ())


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
        # unique per invocation: gpuq refuses a duplicate active label, so a second run of the
        # same batch while a first preflight is queued would be refused at submit
        f"fullcmp-preflight-{batch}-{uuid.uuid4().hex[:8]}",
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
            # a refused submit stops the batch: nothing after it is queued, and the caller exits
            print(
                f"SUBMIT REFUSED label={job.label} rc={res.returncode}: {res.stderr.strip()}",
                file=sys.stderr,
            )
            bad += 1
            break
        jid = res.stdout.strip().splitlines()[-1]
        ids.append(jid)
        JOB_OF_ID[jid] = job
        print(
            f"submitted {jid} ({job.est_min:.1f} min, {len(job.cells)} cells)",
            flush=True,
        )
    return ids, bad


JOB_OF_ID: dict[str, "Job"] = {}  # submitted job id -> its Job (engine, label) for the summary


def job_record(jid: str, env: dict | None = None) -> dict:
    """gpuq's own record of a job ($GPUQ_DIR/jobs/<id>.json): state and rc. {} when unreadable."""
    root = Path((env or gpuq_env()).get("GPUQ_DIR", GPUQ_DIR_DEFAULT))
    try:
        return json.loads((root / "jobs" / f"{jid}.json").read_text())
    except (OSError, ValueError):
        return {}


def job_summary(ids: list[str], env: dict | None = None) -> list[tuple[str, str, object, str, bool]]:
    """(job id, engine, rc, state, ok) per job. ok only for state done with rc 0; no record = failed."""
    rows = []
    for jid in ids:
        rec = job_record(jid, env)
        job = JOB_OF_ID.get(jid)
        engine = job.engine if job else str(rec.get("label", "?"))
        state, rc = rec.get("state", "missing"), rec.get("rc")
        rows.append((jid, engine, rc, state, state == "done" and rc == 0))
    return rows


def wait_all(ids: list[str]) -> int:
    """One blocking gpuq wait, then every job's own rc and state: a per-job table, and nonzero if ANY
    job failed or timed out (the wait's exit code alone reported one failure for six)."""
    if not ids:
        return 0
    env = gpuq_env()
    res = subprocess.run([GPUQ, "wait", *ids], env=env)
    rows = job_summary(ids, env)
    print("job id | engine | rc | state", flush=True)
    for jid, engine, rc, state, _ok in rows:
        print(f"{jid} | {engine} | {rc} | {state}", flush=True)
    failing = sorted({engine for _jid, engine, _rc, _st, ok in rows if not ok})
    bad = sum(1 for row in rows if not row[4])
    if failing:
        print(
            f"FAILED: {bad} of {len(ids)} job(s) failed or timed out; engines: {', '.join(failing)}",
            file=sys.stderr,
            flush=True,
        )
    return 1 if (res.returncode != 0 or failing) else 0


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
        flags = " [over-cap: above 20 min, refused]" if job.over_cap else ""
        flags += " [over-margin: above the 70% limit, run alone]" if job.over_margin else ""
        print(f"JOB {job.label}{flags}")
        print(f"  group: {job.engine}/{job.group} rep {job.rep}")
        print(f"  cells: {' '.join(job.cell_ids)}")
        b = job.budget
        print(
            f"  budget_min: startup {b['startup']:.2f} gate_pre {b['gate_pre']:.2f} "
            f"cells {b['cells']:.2f} post {b['post']:.2f} finalize {b['finalize']:.2f} "
            f"shutdown {b['shutdown']:.2f} total {b['total']:.2f}"
        )
        print(f"  est_min: {job.est_min:.2f} timeout_min: {job.timeout_min:g} "
              f"stall_min: {job.stall_min:g} lane: {job.lane}")
        print(f"  command: {shlex.join(job.submit_argv())}")
    total = sum(j.est_min for j in plan.jobs)
    print(
        f"TOTAL {plan.batch}: {len(plan.jobs)} jobs, {total:.1f} min = {total / 60:.2f} h "
        f"(back-to-back same-engine across rep boundaries: {back_to_back(plan.jobs)})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


HARNESS_WORK = Path("/Volumes/Micron/yunshu-fullcmp/harness")  # snapshots: harness-<HEAD7>/
HARNESS_COMMIT = ""  # set by use_harness(): the snapshot's commit (tfbench header: harness_commit)
HARNESS_TREE = ""  # the snapshot's HEAD^{tree} (tfbench header: harness_tree_sha)
HARNESS_ARCHIVE = ("scripts/research", "scripts/dev", "python", "tests/unit")
HARNESS_DIRTY_PATHS = ("scripts/research", "scripts/dev", "python", "tests/unit")


class HarnessDirty(RuntimeError):
    """The live worktree has uncommitted harness changes: nothing may be submitted from it."""


def _git(args: list[str], repo: Path) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, check=True
    ).stdout


def live_dirty_paths(repo: Path = REPO_ROOT) -> list[str]:
    out = _git(["status", "--porcelain", "--", *HARNESS_DIRTY_PATHS], repo).decode()
    return [line for line in out.splitlines() if line.strip()]


def snapshot_harness(work: Path, repo: Path = REPO_ROOT) -> Path:
    """Freeze the harness at HEAD into work/harness-<HEAD7>/ (read-only, reused when it exists for
    the same commit, never deleted). Refuses when the live worktree has uncommitted harness changes
    or the snapshot's recorded commit does not match HEAD."""
    import io
    import tarfile

    dirty = live_dirty_paths(repo)
    if dirty:
        raise HarnessDirty(
            "live worktree has uncommitted harness changes (commit or stash first): "
            + "; ".join(dirty[:8])
        )
    head = _git(["rev-parse", "HEAD"], repo).decode().strip()
    dest = Path(work) / f"harness-{head[:7]}"
    marker = dest / ".snapshot_commit"
    if not marker.exists():
        tmp = Path(work) / f".tmp-{head[:7]}-{os.getpid()}"
        if tmp.exists():
            raise HarnessDirty(f"snapshot temp dir already exists: {tmp}")
        tmp.mkdir(parents=True)
        data = _git(["archive", "--format=tar", head, *HARNESS_ARCHIVE], repo)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(tmp)
        (tmp / ".snapshot_commit").write_text(head + "\n")
        tree = _git(["rev-parse", "HEAD^{tree}"], repo).decode().strip()
        (tmp / ".snapshot_tree").write_text(tree + "\n")
        manifest, excluded = _tree_blobs(head, repo)
        for name in excluded:  # the tar extract wrote them: drop before the set is fixed
            victim = tmp / name
            if victim.is_file() or victim.is_symlink():
                victim.unlink()
        (tmp / ".snapshot_manifest").write_bytes(
            b"".join(
                os.fsencode(f"{blob}\t{path}\n") for path, blob in sorted(manifest.items())
            )
        )
        (tmp / ".snapshot_excluded").write_bytes(
            b"".join(os.fsencode(p) + b"\n" for p in sorted(excluded))
        )
        for dirpath, dirnames, filenames in os.walk(tmp):
            for name in dirnames + filenames:
                p = Path(dirpath) / name
                os.chmod(p, p.stat().st_mode & ~0o222)
        os.chmod(tmp, tmp.stat().st_mode & ~0o222)
        os.rename(tmp, dest)
    if marker.read_text().strip() != head:
        raise HarnessDirty(f"snapshot {dest} records {marker.read_text().strip()}, HEAD is {head}")
    verify_snapshot(dest, repo)
    return dest


# A path with a control character, a newline, a quote, a backslash, '|' or '$' is not a harness
# file: it is a mangled name (a shell substitution that became a file). Such entries are excluded
# from the snapshot set and logged in .snapshot_excluded, never silently copied or compared.
JUNK_PATH = re.compile(rb'[\x00-\x1f\x7f"\\|$]')


def _tree_blobs(commit: str, repo: Path) -> tuple[dict[str, str], list[str]]:
    """(path -> blob id, excluded paths) of every harness file at `commit`. NUL-separated listing:
    names are raw bytes, never quoted. Paths are kept as os.fsdecode strings (surrogateescape)."""
    raw = _git(["ls-tree", "-r", "-z", commit, "--", *HARNESS_ARCHIVE], repo)
    blobs, excluded = {}, []
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        meta, path_b = entry.split(b"\t", 1)
        if JUNK_PATH.search(path_b):
            excluded.append(os.fsdecode(path_b))
            continue
        blobs[os.fsdecode(path_b)] = meta.split()[2].decode()
    return blobs, excluded


def _file_blobs(root: Path, paths: list[str], repo: Path) -> dict[str, str]:
    """path -> blob id of the files under root, hashed by git itself (git hash-object)."""
    if not paths:
        return {}
    feed = b"".join(os.fsencode(str(root / p)) + b"\n" for p in paths)
    res = subprocess.run(
        ["git", "hash-object", "--stdin-paths"],
        cwd=str(root), input=feed, capture_output=True, check=True,
    )
    return dict(zip(paths, res.stdout.decode().split()))


def verify_snapshot(dest: Path, repo: Path = REPO_ROOT) -> None:
    """Proof that the snapshot is HEAD: its files hash to the blob ids git holds for HEAD, its
    manifest matches the tree, and no file was added, removed or edited. Raises HarnessDirty."""
    dest = Path(dest)
    manifest_path = dest / ".snapshot_manifest"
    try:
        recorded = {}
        for line in manifest_path.read_text().splitlines():
            blob, path = line.split("\t", 1)
            recorded[path] = blob
        head = (dest / ".snapshot_commit").read_text().strip()
    except OSError as exc:
        raise HarnessDirty(f"snapshot {dest} has no readable manifest/commit: {exc}") from exc
    if _tree_blobs(head, repo)[0] != recorded:
        raise HarnessDirty(f"snapshot manifest {manifest_path} does not match git's tree at {head[:7]}")
    present = []
    for dirpath, _dirs, files in os.walk(dest):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, dest)
            if not os.path.basename(rel).startswith(".snapshot_"):
                present.append(rel)
    present.sort()
    if present != sorted(recorded):
        extra = sorted(set(present) - set(recorded))
        missing = sorted(set(recorded) - set(present))
        raise HarnessDirty(f"snapshot file set differs: extra {extra[:5]} missing {missing[:5]}")
    actual = _file_blobs(dest, present, repo)  # hashed by git from raw bytes
    bad = [p for p in present if actual[p] != recorded[p]]
    if bad:
        raise HarnessDirty(f"snapshot files differ from HEAD: {bad[:8]}")


def snapshot_tree(root: Path) -> str:
    return (Path(root) / ".snapshot_tree").read_text().strip()


def use_harness(root: Path) -> None:
    """Point every job-building path at the snapshot. HARNESS_PY (the venv) and the pinned Yunshu
    tree are not harness code and stay as they are."""
    global SCRIPTS, TFBENCH, AUDIT, REPS, PREFLIGHT, QUIET, PROFILE_SCRIPT, HARNESS_COMMIT, HARNESS_TREE
    SCRIPTS = Path(root) / "scripts" / "research"
    TFBENCH = SCRIPTS / "tfbench.py"
    AUDIT = SCRIPTS / "fullcmp_audit.py"
    REPS = SCRIPTS / "fullcmp_reps.py"
    PREFLIGHT = SCRIPTS / "fullcmp_preflight.py"
    QUIET = SCRIPTS / "fullcmp_quiet.py"
    PROFILE_SCRIPT = SCRIPTS / "fullcmp_profile.py"
    HARNESS_COMMIT = (Path(root) / ".snapshot_commit").read_text().strip()
    HARNESS_TREE = snapshot_tree(Path(root))


# Explicit files only: collecting all of tests/unit fails on unrelated modules missing from the venv.
QUICK_TEST_FILES = (
    "tests/unit/test_fullcmp_*.py",
    "tests/unit/test_tfbench_*.py",
    "tests/unit/test_bench_snapshot.py",
    "tests/unit/test_observe_mode.py",
)


def quick_test_paths(root: Path) -> list[str]:
    """The files pre_submit_checks collects (absolute, under root), each pattern expanded, sorted."""
    out: list[str] = []
    for pat in QUICK_TEST_FILES:
        out += sorted(str(p) for p in Path(root).glob(pat))
    return out


def pre_submit_checks(root: Path, repo: Path = REPO_ROOT) -> str:
    """Before any job is written or submitted. (1) The snapshot still equals HEAD (verify_snapshot).
    (2) The static undefined-name check runs on the snapshot: that is what the jobs import.
    (3) The fullcmp_quick subset runs in the live worktree at the same HEAD, which is clean (the
    snapshot refuses a dirty tree), so the tests exercise the code the snapshot contains. The
    snapshot has no .git and lacks data files the tests read, so the subset cannot run there.
    Returns '' when all pass, else the failure."""
    try:
        verify_snapshot(Path(root), repo)
    except HarnessDirty as exc:
        return f"snapshot integrity: {exc}"
    if os.environ.get("TFB_PRESUBMIT_ACTIVE") == "1":
        # a pre-submit's own pytest reached here (through a driver test): refuse, never recurse
        return "nested pre-submit refused (TFB_PRESUBMIT_ACTIVE=1)"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TFB_PRESUBMIT_ACTIVE="1", TFB_PRESUBMIT_SKIP="1")
    static = [HARNESS_PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
              str(Path(root) / "tests" / "unit" / "test_fullcmp_static.py")]
    quick = [HARNESS_PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "fullcmp_quick",
             *quick_test_paths(Path(repo))]
    for cmd, cwd, env_ in (
        (static, Path(root), dict(env, PYTHONPATH=str(Path(root) / "python"))),
        (quick, Path(repo), dict(env, PYTHONPATH=str(Path(repo) / "python"))),
    ):
        res = subprocess.run(cmd, cwd=str(cwd), env=env_, capture_output=True, text=True)
        if res.returncode != 0:
            return f"{' '.join(cmd[-3:])} rc={res.returncode}\n{res.stdout[-2000:]}{res.stderr[-1000:]}"
    return ""


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
    ap.add_argument(
        "--log-usage-sample",
        action="store_true",
        help="every cell also passes tfbench --log-usage-sample (the first raw usage chunk lands in the session header)",
    )
    a = ap.parse_args(argv)
    global LOG_USAGE_SAMPLE
    LOG_USAGE_SAMPLE = a.log_usage_sample
    configure(a.raw_root, a.work_dir)
    # Every job of this invocation runs from one frozen snapshot of the harness at HEAD, never the
    # live worktree (2026-11-04: a harness edit landed mid-batch and a NameError killed a job).
    test_run = os.environ.get("TFB_PRESUBMIT_SKIP") == "1" or os.environ.get("TFB_PRESUBMIT_ACTIVE") == "1"
    if test_run:
        # a unit-test run calls main() on the live tree: no git reads, no snapshot (the snapshot
        # code is tested by test_fullcmp_snapshot.py)
        snap = REPO_ROOT
        print("harness snapshot skipped (test run)", flush=True)
    else:
        try:
            snap = snapshot_harness(HARNESS_WORK, REPO_ROOT)
        except HarnessDirty as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 2
        use_harness(snap)
    if test_run:
        # inside a test run or a pre-submit's own pytest: never a second pre-submit (that recursed)
        print("pre-submit skipped: TFB_PRESUBMIT_SKIP/ACTIVE set (test or nested run)", flush=True)
        quick = ""
    else:
        quick = pre_submit_checks(snap, REPO_ROOT)
    if quick:
        print("refusing: pre-submit checks failed on the snapshot:", file=sys.stderr)
        print(quick, file=sys.stderr)
        return 2
    if not test_run:
        print(f"harness snapshot: {snap} (commit {HARNESS_COMMIT})", flush=True)

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
    # Every per-engine table, resolved for every selected engine (fullcmp_engine_tables): a miss
    # refuses here, before any job is written, so dry-run, self-check and submission all stop.
    import fullcmp_engine_tables  # lazy: that module imports this driver's siblings

    misses = fullcmp_engine_tables.check_all(
        engines or sorted({r.get("engine") for r in rows if r.get("batch") == a.batch})
    )
    if misses:
        print(fullcmp_engine_tables.format_misses(misses), file=sys.stderr)
        return 2
    plan = plan_batch(
        rows, a.batch, pin, engines=engines, rep=a.rep, profile_rep=a.profile_rep
    )
    problems = prompt_corpus_problems(plan.runnable)
    if problems:
        for p in problems:
            print(f"refusing: {p}", file=sys.stderr)
        return 2
    # Provenance before anything is printed, checked or submitted: every selected engine must resolve
    # its version, census release date and binary sha256, and the model its census weight shas.
    prov = fp.dry_run_problems(
        [j.engine for j in plan.jobs],
        fp.load_json(fp.CENSUS_PATH),
        pin,
        MODEL_PATH,
    )
    if prov:
        for p in prov:
            print(f"refusing: provenance: {p}", file=sys.stderr)
        return 2

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
        for job in plan.jobs:  # the corpus each generated job body exports (every cell)
            works = sorted(set(re.findall(r"TFB_WORK=(\S+)", job.script)))
            print(f"  job body {job.label}: TFB_WORK={','.join(works) or 'MISSING'}")
        for job in plan.jobs:  # the observe state each job body exports (timed 0, -prof 1)
            obs = sorted(set(re.findall(r"TFB_OBSERVE=(\S+)", job.script)))
            print(f"  job body {job.label}: TFB_OBSERVE={','.join(obs) or 'MISSING'}")
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
    if bad:
        print(f"{a.batch}: submit refused, {len(ids)} job(s) already queued", file=sys.stderr)
        return 1
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
