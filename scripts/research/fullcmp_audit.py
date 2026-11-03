#!/usr/bin/env python3
"""Audit a full fair-comparison result directory (FULLCMP_SPEC.md, "Hard guards").

Layout audited (written by the runner, checked here; nothing is trusted from the runner):

    <root>/<engine>/<cell>-r<rep>.jsonl    one file per engine x cell x rep
    <manifest>.json                         expected-cell manifest, written before the run

Each jsonl file is one session header row (part == "session", first row) followed by one
request row per HTTP request (part == "decode"). Anything that is not a complete, measured
row is a violation. Every violation is printed as "<guard>: <cell>: <reason>".

Usage:
    python scripts/research/fullcmp_audit.py --root DIR --manifest MANIFEST.json
    python scripts/research/fullcmp_audit.py --root DIR --manifest MANIFEST.json \
        --recompute --table FULLCMP_TABLE.md

Exit status: 0 when all 15 guards pass (and the table recomputes, with --recompute),
1 otherwise. This module is CPU-only: it reads json files and never touches MLX or a server.

--recompute parses FULLCMP_TABLE.md: one "### <model> — <metric>" section per model x metric
(metric in METRICS), a header row "| engine | <column> ... |", and cells "<median> ± <spread>"
(or "invalid: reason"). Sections whose heading contains "derived" are skipped (derived values
live in their own labeled section). Median = median over the 3 reps of the per-rep median of
the matching request rows; spread = max - min over reps.

Record schema (FULLCMP_SCHEMA) and the fields bench_engines / tfbench do not emit yet are
documented below and exposed via schema_fields_not_emitted().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_engines  # noqa: E402  (manifest id <-> harness id registry)
import fullcmp_attribution as fat  # noqa: E402  (layer-1 / layer-2 attribution schema)
import fullcmp_engaged as fe  # noqa: E402  (engaged spec mode from evidence)

# ---------------------------------------------------------------------------
# Record schema. "manifest" is the expected-cell file; "header" is row 0 of each jsonl
# (part "session"); "request" is one row per HTTP request (part "decode").
# ---------------------------------------------------------------------------
FULLCMP_SCHEMA = {
    "manifest": {
        "run_id": "str, one id per run; every header must carry it (stale files fail)",
        "reps": "int, reps per cell (3); file <cell>-r<rep>.jsonl for rep in range(reps)",
        "pins": "{engine: {version, release_date, binary_sha256}}; yunshu: {version, release_date, git_sha}",
        "yunshu": "{git_sha, python}: pinned Yunshu tree sha and <tree>/.venv/bin/python",
        "controls": "{aa_delta_pct, aa_noise_pct, injected_delay_detected}: A/A and injected-delay controls",
        "groups": "{group: {order: [engine, ...]}}: rep-0 interleave order; rep r rotates it by r",
        "cells": "[{engine, model, column, cell, group, ctx, kind, prompt_id, max_tokens, phases, expected_records, spec_mode, weight_group, rep_decision?, need_rep3?, reps?}]",
        "rep policy (per cell, optional)": "rep_decision: ok | need_rep3 | unstable | pending (fullcmp_reps.py status). need_rep3 or unstable -> 3 reps; ok -> 2 reps (reps, default the manifest reps); pending -> incomplete. A cell without rep_decision keeps the manifest reps (old manifests).",
    },
    "header": {
        "part": '"session"',
        "run_id": "str, see manifest",
        "engine": "str, engine id (folder name)",
        "model": "str, model id of the cell",
        "cell": "str, cell id (file stem)",
        "group": "str, comparison group (same prompt, same ctx/kind across engines)",
        "rep": "int, rep index",
        "version": "str, engine version as installed",
        "release_date": "YYYY-MM-DD, upstream release date",
        "git_sha": "str, engine git sha (Yunshu: the pinned tree)",
        "binary_sha256": "64 hex, sha256 of the engine binary (competitors)",
        "cmd": "list, argv the server was started with",
        "python": "str, venv python used (Yunshu only)",
        "requested_spec_mode": "str, mode configured for the run",
        "spec_mode_expected": "str, mode the cell expects",
        "engaged_spec_mode": "str, mode parsed from the engine's own log",
        "engaged_log_line": "str, the log line the engaged mode was parsed from",
        "weights": "list of {file, sha256, bits, group_size} the engine loaded",
        "fresh_process": "bool, server started cold for this cell group",
        "home": "str, isolated HOME of this engine",
        "job_id": "str, gpuq job id",
        "job_start_s": "float, job start (epoch s)",
        "job_end_s": "float, job end (epoch s)",
        "gpu_exclusive": "bool, no other GPU job ran during the job",
        "quiet_pre": "dict, quiet-machine snapshot before the server started (fullcmp_quiet)",
        "quiet_post": "dict, quiet-machine snapshot after the cell, server still up",
        "contended": "bool, machine contention flag",
        "device": "str, machine id",
        "observe": "bool, Yunshu only: observe mode the server ran with (timed false, profile true)",
        "yunshu_observe": "str|null, the YUNSHU_OBSERVE value passed to the server env (\"1\" or null)",
        "profile": "bool, true only in a profile rep (TFB_PROFILE=1)",
    },
    "request": {
        "part": '"decode"',
        "phase": "str, cold | warm | turn2",
        "ctx": "int, context length",
        "kind": "str, code | prose",
        "cell": "str, cell id",
        "prompt_file": "str, prompt source file",
        "harness": "str, harness name and version",
        "prompt_ids_sha": "str, sha256 of the prompt token ids the server reported",
        "pt": "int, prompt token count",
        "max_tokens": "int, requested output budget",
        "ct": "int, completion tokens actually produced",
        "temperature": "number, must be 0 (greedy)",
        "stop": "list, stop strings",
        "ttft_s": "float, time to first token",
        "total_s": "float, total request time",
        "dec_tps": "float, decode tokens per second",
        "cached": "int, server-reported cached prompt tokens (cold: must be 0)",
        "finish": "str, stop | length; anything else is an incomplete stream",
        "sha": "hex, output sha (text, reasoning, tool calls, finish)",
        "contended": "bool, per-request contention flag",
        "error": "absent on measured rows; present = failed request (never a number)",
    },
}

# Fields bench_engines / tfbench (scripts/research/tfbench.py `send`, `emit`, `part_decode`;
# scripts/research/bench_engines.py `meta_row`) emit at b70535990. Anything in FULLCMP_SCHEMA
# that is not in this set is a field the runner must add before the audit can pass.
EMITTED_BY_BENCH_ENGINES = {
    "header": {
        "part",
        "engine",
        "product",
        "version",
        "git_sha",
        "flags",
        "drafter",
        "spec_mode",
        "spec_mode_expected",
        "checkpoint",
        "weights_vs_oQ4e",
        "snapshot_rep",
        "rep",
        "ready_s",
        "cmd",
        "env",
        "requested_spec_mode",
        "engaged_spec_mode",
        "engine_probe",
        "tag",
        "device",
        "contended",
        "execution_device",
        "remote_host",
    },
    "request": {
        "part",
        "ctx",
        "kind",
        "phase",
        "reference_prompt_tokens",
        "pt",
        "ct",
        "cached",
        "ttft_s",
        "total_s",
        "dec_tps",
        "finish",
        "sha",
        "reasoning_chars",
        "lp_tokens",
        "lps",
        "head",
        "rep4",
        "energy",
        "joules_per_token",
        "gpu_watts_mean",
        "xy",
        "contended",
        "device",
        "rehearsals_completed",
    },
}

METRICS = {
    "decode_tps": ("cold", "dec_tps"),
    "cold_ttft": ("cold", "ttft_s"),
    "warm_ttft": ("warm", "ttft_s"),
    "turn2_ttft": ("turn2", "ttft_s"),
    "turn2_decode_tps": ("turn2", "dec_tps"),
}

PROVENANCE_KEYS = (
    "harness",
    "max_tokens",
    "prompt_file",
    "ctx",
    "kind",
    "temperature",
    "stop",
)
TABLE_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?\s*$")


def schema_fields_not_emitted() -> dict:
    """Schema fields that bench_engines / tfbench do not emit yet (runner must add them)."""
    out = {}
    for section in ("header", "request"):
        missing = sorted(
            set(FULLCMP_SCHEMA[section]) - EMITTED_BY_BENCH_ENGINES[section]
        )
        out[section] = missing
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class CellFile:
    cell: dict
    rep: int
    path: Path
    header: dict | None
    rows: list  # request rows (part == "decode")
    # parts that wrote request rows but no part_done(complete=True): a killed or timed-out job
    truncated: tuple = ()


@dataclass
class Run:
    manifest: dict
    files: list  # CellFile for every expected file that exists
    missing: list  # (cell, rep, path) for expected files that do not exist
    extra: list  # Path of jsonl files on disk that the manifest does not plan
    parse_errors: list  # str, one per unparseable line (truncated stream = error)
    profile_root: object = None  # <profile_root>/<engine>/<cell>-prof.jsonl: layer-2 rows
    full: bool = False  # full mode: every cell needs a profile rep (missing = violation)
    behaviour: dict = field(default_factory=dict)  # engine -> rows of <root>/behaviour/<engine>.jsonl


@dataclass
class GuardResult:
    name: str
    checked: int = 0
    violations: list = field(default_factory=list)

    def fail(self, cell, msg):
        self.violations.append(f"{self.name}: {cell}: {msg}")


TELEMETRY_SUFFIX = ".telemetry.jsonl"


def telemetry_path(cell_path: Path) -> Path:
    """<cell>-rN.jsonl -> <cell>-rN.telemetry.jsonl (written by fullcmp_telemetry.Sampler)."""
    return cell_path.with_name(cell_path.name[: -len(".jsonl")] + TELEMETRY_SUFFIX)


def _read_jsonl(path: Path):
    rows, errors = [], []
    for no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"measured_only: {path}: line {no} unparseable ({exc.msg})")
    return rows, errors


REP_DECISIONS = ("ok", "need_rep3", "unstable", "pending")


def expected_reps(cell: dict, manifest: dict) -> int:
    """Reps this cell must have. The rep-3 rule is per cell (fullcmp_reps.py): need_rep3 (or an
    unstable verdict, which exists only after rep 3) -> 3; a recorded 'ok' (rep 3 not required)
    -> 2; no decision recorded -> the cell's reps, else the manifest reps."""
    if cell.get("need_rep3") or cell.get("rep_decision") in ("need_rep3", "unstable"):
        return 3
    return int(cell.get("reps", manifest["reps"]))


DECODE_PART = "decode"
CONC_PART = "conc"
CONC_LAYER1_REASON = (
    "conc record carries relative ttfts/total_s per request, no per-request epoch timestamps "
    "or per-round/server counters (tfbench conc part)"
)


MIXED_PART = "mixed"
MIXED_ROLES = ("long", "short")


def conc_request_rows(rec: dict, part: str = CONC_PART) -> list[dict]:
    """One request row per request of a tfbench `conc` (or `mixed`) record: one record = n
    simultaneous requests of one trial. Per-request lists are read by index; a missing entry stays
    None so the measured_only / record_count guards flag it instead of counting it."""
    n = rec.get("n")
    if not isinstance(n, int) or n < 1:
        return [{"part": part, "phase": f"{part}?", "error": f"{part} record n={n!r}"}]
    rows = []
    for i in range(n):

        def at(key):
            v = rec.get(key)
            return v[i] if isinstance(v, list) and i < len(v) else None

        attr = {"cache_hit_tokens": at("cached")} if at("cached") is not None else {}
        unavailable = {
            f: CONC_LAYER1_REASON
            for f in list(fat.LAYER1_TIMESTAMPS) + list(fat.LAYER1_COUNTERS)
            if f not in attr and f != "cache_hit_tokens"
        }
        roles = rec.get("roles") if part == MIXED_PART else None
        rows.append(
            {
                "part": part,
                "phase": part,
                "role": roles[i] if isinstance(roles, list) and i < len(roles) else None,
                "same_prompt": rec.get("same_prompt"),
                "attr": attr,
                "attr_unavailable": unavailable,
                "n": n,
                "trial": rec.get("trial"),
                "ttft_s": at("ttfts"),
                "dec_tps": at("per_req_dec"),
                "total_s": at("total_s"),
                "ct": at("cts"),
                "pt": at("pts"),
                "cached": at("cached"),
                "sha": at("shas"),
                "finish": at("finish"),
                "prompt_ids_sha": at("prompt_ids_sha"),
                "max_tokens": rec.get("max_tokens"),
                "temperature": rec.get("temperature"),
                "stop": rec.get("stop"),
                "harness": rec.get("harness"),
                "ctx": rec.get("ctx"),
                "kind": rec.get("kind"),
                "prompt_file": rec.get("prompt_file"),
            }
        )
    # record-level aggregates ride on every request row of the record (checked by usage_fields)
    agg = {k: rec.get(k) for k in ("wall_s", "total_tokens", "agg_tps")}
    decs = rec.get("per_req_dec")
    if part == MIXED_PART and isinstance(decs, list):
        decs = decs[1:]  # the mixed aggregates summarise the short requests (request 0 is the long one)
    for row in rows:
        row["agg"] = agg
        row["per_req_dec_all"] = decs
        for k in ("dec_min", "dec_max", "dec_median", "fair_max_min"):
            row[k] = rec.get(k)
    return rows


def load_run(root, manifest: dict) -> Run:
    root = Path(root)
    expected = {}
    for cell in manifest["cells"]:
        for rep in range(expected_reps(cell, manifest)):
            expected[root / cell["engine"] / f"{cell['cell']}-r{rep}.jsonl"] = (
                cell,
                rep,
            )
    files, missing, parse_errors = [], [], []
    for path, (cell, rep) in expected.items():
        if not path.is_file():
            missing.append((cell, rep, path))
            continue
        rows, errs = _read_jsonl(path)
        parse_errors += errs
        header = next((r for r in rows if r.get("part") == "session"), None)
        body = [r for r in rows if r.get("part") == "decode"]
        for r in rows:
            if r.get("part") == CONC_PART:
                body += conc_request_rows(r)
            elif r.get("part") == MIXED_PART:
                body += conc_request_rows(r, MIXED_PART)
        # tfbench writes part_done after each part finishes; a part with request rows but no
        # part_done was cut short (gpuq timeout or kill), so its cell is incomplete
        started = {r.get("part") for r in rows if r.get("part") in (DECODE_PART, CONC_PART, MIXED_PART)}
        done = {r.get("which") for r in rows if r.get("part") == "part_done" and r.get("complete") is True}
        files.append(CellFile(cell, rep, path, header, body, truncated=tuple(sorted(started - done))))
    extra = [
        p
        for p in sorted(root.glob("*/*.jsonl"))
        if p not in expected
        and not p.name.endswith(TELEMETRY_SUFFIX)
        and p.parent.name != BEHAVIOUR_DIR
    ]
    behaviour = {}
    for p in sorted((root / BEHAVIOUR_DIR).glob("*.jsonl")):
        rows, errs = _read_jsonl(p)
        parse_errors += errs
        behaviour[p.stem] = rows
    return Run(manifest, files, missing, extra, parse_errors, behaviour=behaviour)


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rotate(order, rep):
    k = rep % len(order)
    return list(order[k:]) + list(order[:k])


# ---------------------------------------------------------------------------
# The 12 hard guards. Each takes a Run and returns a GuardResult.
# ---------------------------------------------------------------------------
# sha256 of each corpus prompt FILE (`<sha>  <name>` lines). tfbench records the same bytes as
# prompt_file_sha256 on every decode row; prompt_sha256 is the rendered request and is not a file hash.
PROMPT_SHA_FILE = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/prompt_files.sha256.txt"
)


def load_prompt_file_table() -> dict:
    """basename -> sha256 of the prompt file. A missing or unreadable table raises (fail closed)."""
    table = {}
    for line in PROMPT_SHA_FILE.read_text().splitlines():
        if line.strip():
            sha, name = line.split(maxsplit=1)
            table[name.strip()] = sha
    return table


def check_prompt_identity(run: Run) -> GuardResult:
    """Guard 1: prompt token count + sha identical across engines in a comparison group.

    Compared on cold and warm rows only: turn2 prompts embed each engine's own reply, so
    they are recorded per engine and not required to match.
    Also, every decode row's prompt_file_sha256 (the bytes the cell read) must equal the entry for
    its prompt file in PROMPT_SHA_FILE; a missing entry, a missing sha or a mismatch fails the cell.
    """
    res = GuardResult("prompt_identity")
    try:
        table = load_prompt_file_table()
    except (OSError, ValueError) as exc:
        res.checked += 1
        for cf in run.files:
            res.fail(cf.cell["cell"], f"prompt sha table unreadable ({PROMPT_SHA_FILE}): {exc}")
        table = None
    for cf in run.files:
        if table is None:
            break
        res.checked += 1
        problems = []
        for r in cf.rows:
            if r.get("part") != "decode":
                continue
            name = Path(str(r.get("prompt_file") or "")).name
            got, want = r.get("prompt_file_sha256"), table.get(name)
            if want is None:
                problems.append(f"prompt file {name!r} has no entry in the sha table")
            elif not isinstance(got, str) or not got:
                problems.append(f"prompt_file_sha256 not recorded for {name!r}")
            elif got != want:
                problems.append(f"prompt file {name!r} sha {got[:12]} != table {want[:12]}")
        for msg in sorted(set(problems)):
            res.fail(cf.cell["cell"], msg)
    groups = defaultdict(list)  # group -> [(cell id, engine, row)]
    for cf in run.files:
        for r in cf.rows:
            if r.get("phase") in ("cold", "warm"):
                groups[cf.cell["group"]].append((cf.cell["cell"], cf.cell["engine"], r))
    cells_of = defaultdict(set)
    for cf in run.files:
        cells_of[cf.cell["group"]].add(cf.cell["cell"])
    for group, items in groups.items():
        res.checked += 1
        seen = set()
        for cell, eng, r in items:
            sha, pt = r.get("prompt_ids_sha"), r.get("pt")
            if not isinstance(sha, str) or not sha or not isinstance(pt, int):
                res.fail(cell, f"[{eng}] prompt token count/sha not recorded")
            seen.add((pt, sha))
        if len(seen) > 1:
            for cell in sorted(cells_of[group]):
                res.fail(
                    cell,
                    f"group {group}: engines disagree on prompt (pt, sha) {sorted(map(str, seen))}",
                )
    return res


def check_provenance_identity(run: Run) -> GuardResult:
    """Guard 2: a comparison group only mixes identical provenance."""
    res = GuardResult("provenance_identity")
    per_group = defaultdict(dict)  # group -> {cell: provenance tuple}
    for cf in run.files:
        res.checked += 1
        if not cf.rows:
            continue
        prov_set = set()
        for r in cf.rows:
            missing = [k for k in PROVENANCE_KEYS if k not in r]
            if missing:
                res.fail(cf.cell["cell"], f"provenance fields missing: {missing}")
            prov_set.add(
                json.dumps([r.get(k) for k in PROVENANCE_KEYS], sort_keys=True)
            )
            if r.get("max_tokens") != cf.cell["max_tokens"]:
                res.fail(
                    cf.cell["cell"],
                    f"max_tokens {r.get('max_tokens')} != planned {cf.cell['max_tokens']}",
                )
            # conc requests are a fixed prose/code mix by design (tfbench concurrent_prompt): their
            # kind is "mixed", so only ctx is compared for them.
            kind_ok = (
                r.get("part") in (CONC_PART, MIXED_PART) or r.get("kind") == cf.cell["kind"]
            )
            if r.get("ctx") != cf.cell["ctx"] or not kind_ok:
                res.fail(cf.cell["cell"], "ctx/kind differs from the planned cell")
        if len(prov_set) > 1:
            res.fail(cf.cell["cell"], "provenance differs between rows of one cell")
        per_group[cf.cell["group"]][(cf.cell["engine"], cf.cell["cell"])] = (
            prov_set.pop() if prov_set else None
        )
    for group, by_key in per_group.items():
        if len(set(by_key.values())) > 1:
            for _, cell in by_key:
                res.fail(
                    cell,
                    f"provenance differs within group {group} (harness/max_tokens/prompt/sampling)",
                )
    return res


def manifest_engine(eng: str) -> str:
    """Manifest engine id for a folder or harness id (inverse of bench_engines.MANIFEST_ENGINES).
    Yunshu is manifest id 'yunshu' and harness id 'yunshu-main'; its pin comes from yunshu_pin.json
    through the manifest's 'yunshu' section."""
    if eng in bench_engines.MANIFEST_ENGINES:
        return eng
    inverse = {v: k for k, v in bench_engines.MANIFEST_ENGINES.items()}
    return inverse.get(eng, eng)


def check_pinned_versions(run: Run) -> GuardResult:
    """Guard 3: version, release date, binary sha256 / pinned tree, and this run's run_id."""
    res = GuardResult("pinned_versions")
    m = run.manifest
    pins = m.get("pins", {})
    yun = m.get("yunshu", {})
    for cf in run.files:
        h = cf.header
        if h is None:
            continue
        res.checked += 1
        cell = cf.cell["cell"]
        eng = manifest_engine(cf.cell["engine"])
        pin = pins.get(eng)
        if pin is None:
            res.fail(cell, f"no pin for engine {eng}")
            continue
        if h.get("run_id") != m.get("run_id"):
            res.fail(cell, f"run_id {h.get('run_id')!r} is not this run (stale file?)")
        if not h.get("version") or h.get("version") != pin.get("version"):
            res.fail(
                cell, f"version {h.get('version')!r} != pinned {pin.get('version')!r}"
            )
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(h.get("release_date"))):
            res.fail(cell, "release_date not recorded")
        if eng == "yunshu":
            if h.get("git_sha") != yun.get("git_sha") or pin.get("git_sha") != yun.get(
                "git_sha"
            ):
                res.fail(cell, "Yunshu tree sha is not the pinned tree")
            cmd = h.get("cmd") or []
            if h.get("python") != yun.get("python"):
                res.fail(cell, "Yunshu header python is not the pinned venv python")
            # stub_engine (test only, set by the e2e test): the server process is the stub, so its
            # command line cannot be the pinned python. Production manifests never set it.
            # the header digest is the launched python (cmd[0]); it must be the pinned digest
            if not re.fullmatch(r"[0-9a-f]{64}", str(h.get("binary_sha256"))) or h.get(
                "binary_sha256"
            ) != pin.get("binary_sha256"):
                res.fail(cell, "Yunshu binary sha256 missing or not the pinned python")
            if m.get("stub_engine") is not True:
                if not cmd or cmd[0] != yun.get("python"):
                    res.fail(cell, "Yunshu not started by its pinned venv python")
                if "-m" not in cmd or "yunshu_cli" not in cmd:
                    res.fail(cell, "Yunshu not started as python -m yunshu_cli")
        elif m.get("stub_engine") is not True and (
            not re.fullmatch(r"[0-9a-f]{64}", str(h.get("binary_sha256")))
            or h.get("binary_sha256") != pin.get("binary_sha256")
        ):
            # stub_engine (test only): the launched process is the stub, not the engine binary
            res.fail(cell, "binary sha256 missing or not the pinned binary")
    return res


def check_manifest_completeness(run: Run) -> GuardResult:
    """Guard 4: every planned cell x rep exists; no unplanned file; no duplicate cells."""
    res = GuardResult("manifest_completeness")
    cells = run.manifest["cells"]
    counts = Counter((c["engine"], c["cell"]) for c in cells)
    for (eng, cid), n in counts.items():
        res.checked += 1
        if n > 1:
            res.fail(cid, f"duplicate ({eng}, {cid}) in manifest")
    pins = run.manifest.get("pins", {})
    for c in cells:
        res.checked += 1
        if c["engine"] not in pins:
            res.fail(c["cell"], f"engine {c['engine']} has no pin")
    for c in cells:
        res.checked += 1
        dec = c.get("rep_decision")
        if dec is not None and dec not in REP_DECISIONS:
            res.fail(c["cell"], f"rep_decision {dec!r} is not one of {REP_DECISIONS}")
        elif dec == "pending":
            res.fail(
                c["cell"],
                "rep decision pending (fullcmp_reps.py has not decided rep 3)",
            )
    for cell, rep, path in run.missing:
        res.fail(cell["cell"], f"r{rep} missing ({path})")
    for path in run.extra:
        res.fail(path.stem, f"unplanned file {path} (not in manifest)")
    return res


def check_record_count(run: Run) -> GuardResult:
    """Guard 5: expected record count, every phase present, complete streams, output tokens."""
    import fullcmp_manifest as fm

    res = GuardResult("record_count")
    for cf in run.files:
        cell = cf.cell["cell"]
        res.checked += 1
        if cf.header is None:
            res.fail(cell, f"r{cf.rep}: no session header row")
        for part in cf.truncated:
            res.fail(
                cell,
                f"r{cf.rep}: truncated: part '{part}' has request rows but no part_done "
                "(the job timed out or was killed; the cell is incomplete)",
            )
        exp = cf.cell["expected_records"]  # per rep file (fullcmp_run.expected_records)
        if len(cf.rows) != exp:
            res.fail(cell, f"r{cf.rep}: {len(cf.rows)} request rows, expected {exp}")
        if set(r.get("phase") for r in cf.rows) != set(cf.cell["phases"]):
            res.fail(
                cell,
                f"r{cf.rep}: phases {sorted(map(str, set(r.get('phase') for r in cf.rows)))} != planned {cf.cell['phases']}",
            )
        # a request-family reply ends where its shape ends (a tool call, a JSON object): its
        # validity is check_family_validity's job, so only the stream's completeness is checked here
        family = fm.is_family_cell(cf.cell.get("kind"), cf.cell.get("thinking"), cf.cell.get("media"))
        for r in cf.rows:
            if r.get("finish") not in ("stop", "length", "tool_calls") if family else r.get("finish") not in ("stop", "length"):
                res.fail(
                    cell,
                    f"r{cf.rep} {r.get('phase')}: incomplete stream (finish={r.get('finish')!r})",
                )
            if family:
                if not r.get("ct"):
                    res.fail(cell, f"r{cf.rep} {r.get('phase')}: no output tokens")
                continue
            if r.get("ct") != cf.cell["max_tokens"]:
                res.fail(
                    cell,
                    f"r{cf.rep} {r.get('phase')}: output tokens {r.get('ct')} != expected {cf.cell['max_tokens']}",
                )
    return res


def check_engaged_mode(run: Run) -> GuardResult:
    """Guard 6: the engaged spec mode is derived from evidence (fullcmp_engaged) in the cell's full
    engine log (split at the header's ready offset) and its per-request records. The header's final
    engaged mode, log line and source must equal the evidence. Fails closed: no table, 'unverified'
    (loaded but not engaged, or no evidence), a mode other than the configured one, a missing ready
    offset, and a speculative mode whose per-request acceptance is zero or absent."""
    res = GuardResult("engaged_mode")
    table = fe.load_table()
    for cf in run.files:
        h = cf.header
        if h is None:
            continue
        res.checked += 1
        want, cell = cf.cell["spec_mode"], cf.cell["cell"]
        if not want:
            # no expected mode in the manifest: a violation, never a crash (fail closed)
            res.fail(cell, f"r{cf.rep}: manifest has no expected spec mode for this cell")
            continue
        engine = cf.cell.get("engine")
        if engine == "yunshu":
            slice_path = server_log_slice_path(cf.path)
            log = slice_path.read_text(errors="replace") if slice_path.is_file() else ""
            ev = fe.yunshu_runner_evidence(log, h.get("server_log_ready_offset"))
            m = None
        else:
            if table is None:
                res.fail(cell, f"r{cf.rep}: engaged evidence table missing or malformed (fail closed)")
                continue
            m = fe.mode_entry(table, engine, want)
            if m is None:
                res.fail(cell, f"r{cf.rep}: no evidence entry for {engine}/{want} (fail closed)")
                continue
            slice_path = server_log_slice_path(cf.path)
            log = slice_path.read_text(errors="replace") if slice_path.is_file() else ""
            ready = h.get("server_log_ready_offset")
            if not isinstance(ready, int):
                res.fail(cell, f"r{cf.rep}: server_log_ready_offset missing (fail closed)")
                continue
            ev = fe.evaluate(engine, want, log, cf.rows, table,
                             probe_text=h.get("engine_probe"), ready_offset=ready)
        got = ev["engaged_mode"]
        if got == fe.UNVERIFIED:
            kind = ev.get("evidence_kind")
            why = "loaded only (startup line, no per-request evidence)" if kind == "startup_only" else "no evidence"
            res.fail(cell, f"r{cf.rep}: engaged mode unverified: {why}")
        elif got != want:
            res.fail(cell, f"r{cf.rep}: engaged evidence says {got!r}, configured {want!r}")
        for key in ("requested_spec_mode", "spec_mode_expected"):
            if h.get(key) != want:
                res.fail(cell, f"r{cf.rep}: {key}={h.get(key)!r}, configured {want!r}")
        if h.get("engaged_spec_mode") != got:
            res.fail(cell, f"r{cf.rep}: header engaged_spec_mode={h.get('engaged_spec_mode')!r}, evidence says {got!r}")
        if h.get("engaged_log_line") != ev["matched_line"]:
            res.fail(cell, f"r{cf.rep}: engaged_log_line is not the evidence line")
        if engine != "yunshu" and h.get("engaged_evidence_source") != ev.get("source"):
            res.fail(cell, f"r{cf.rep}: engaged_evidence_source={h.get('engaged_evidence_source')!r}, evidence says {ev.get('source')!r}")
        if (problem := fe.acceptance_problem(got, m or {}, cf.rows)) is not None:
            res.fail(cell, f"r{cf.rep}: {problem}")
    return res


def _observe_problems(header: dict) -> list[str]:
    """Observe rule for one Yunshu session header: timed (profile false) runs observe OFF, profile
    runs observe ON; the resolved YUNSHU_OBSERVE value must agree. A missing field is a problem."""
    out = []
    obs, prof = header.get("observe"), header.get("profile")
    if not isinstance(obs, bool):
        out.append("observe field missing or not bool")
    if not isinstance(prof, bool):
        out.append("profile field missing or not bool")
    if isinstance(obs, bool) and isinstance(prof, bool):
        if obs != prof:
            out.append(f"observe={obs} but profile={prof} (timed rows observe false, profile rows true)")
        want_env = "1" if obs else None
        if header.get("yunshu_observe") != want_env:
            out.append(f"yunshu_observe={header.get('yunshu_observe')!r}, observe={obs}")
    return out


def check_observe_state(run: Run) -> GuardResult:
    """Guard 7b (lead ruling 2026-11-03): a Yunshu timed rep runs with observe OFF (TFB_OBSERVE=0,
    YUNSHU_OBSERVE unset); observe, stage timers and xctrace only in the profile rep (observe ON).
    Every Yunshu session header carries observe, profile and yunshu_observe; a missing field, a
    timed row with observe on, or a profile row with observe off fails the cell (fail closed)."""
    res = GuardResult("observe_state")
    for cf in run.files:
        cell, engine = cf.cell["cell"], cf.cell.get("engine")
        if engine != "yunshu":
            continue
        res.checked += 1
        for problem in _observe_problems(cf.header or {}):
            res.fail(cell, f"r{cf.rep}: {problem}")
    if run.profile_root is None:
        return res
    seen = set()
    for cf in run.files:
        cell, engine = cf.cell["cell"], cf.cell.get("engine")
        if engine != "yunshu" or (engine, cell) in seen:
            continue
        seen.add((engine, cell))
        path = Path(run.profile_root) / str(engine) / f"{cell}-prof.jsonl"
        if not path.is_file():
            continue  # a missing profile rep is reported by check_attribution_fields
        rows, _errs = _read_jsonl(path)
        session = next((r for r in rows if r.get("part") == "session"), None)
        res.checked += 1
        if session is None:
            res.fail(cell, "profile rep: no session header")
            continue
        for problem in _observe_problems(session):
            res.fail(cell, f"profile rep: {problem}")
        if session.get("profile") is not True:
            res.fail(cell, "profile rep: session profile field is not true")
    return res


def check_weight_identity(run: Run) -> GuardResult:
    """Guard 7: weight sha256 + bits + group size recorded; equal within a weight_group."""
    res = GuardResult("weight_identity")
    by_group = defaultdict(list)  # weight_group -> [(cell, canon)]
    for cf in run.files:
        h = cf.header
        if h is None:
            continue
        res.checked += 1
        cell = cf.cell["cell"]
        w = h.get("weights")
        if not isinstance(w, list) or not w:
            res.fail(cell, f"r{cf.rep}: no weight sha256 recorded")
            continue
        entries = [e for e in w if isinstance(e, dict)]
        for e in entries:
            if not (
                re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256")))
                and isinstance(e.get("bits"), int)
                and isinstance(e.get("group_size"), int)
            ):
                res.fail(cell, f"r{cf.rep}: weight entry lacks sha256/bits/group_size")
        canon = json.dumps(
            sorted(
                json.dumps([e.get("sha256"), e.get("bits"), e.get("group_size")])
                for e in entries
            )
        )
        if cf.cell.get("weight_group"):
            by_group[cf.cell["weight_group"]].append((cell, canon))
    for wg, items in by_group.items():
        res.checked += 1
        if len({c for _, c in items}) > 1:
            for cell, _ in items:
                res.fail(cell, f"weights differ within weight_group {wg}")
    return res


def check_cold_cache(run: Run) -> GuardResult:
    """Guard 8: cold cells are a fresh process with server-reported cached_tokens == 0."""
    res = GuardResult("cold_cache")
    for cf in run.files:
        cell = cf.cell["cell"]
        if cf.header is not None:
            res.checked += 1
            if cf.header.get("fresh_process") is not True:
                res.fail(cell, f"r{cf.rep}: not a fresh process")
        for r in cf.rows:
            res.checked += 1
            cached, phase = r.get("cached"), r.get("phase")
            if phase == "cold":
                if (
                    isinstance(cached, bool)
                    or cached != 0
                    or not isinstance(cached, int)
                ):
                    res.fail(
                        cell,
                        f"r{cf.rep} cold: cached={cached!r}, must be server-reported 0",
                    )
            elif phase == "restart":
                # the relaunched server serves the cold prompt from the disk prefix tier: a cache
                # miss here means the restart did not keep the cache (HOME wiped, tier not used)
                if isinstance(cached, bool) or not isinstance(cached, int) or cached <= 0:
                    res.fail(
                        cell,
                        f"r{cf.rep} restart: cached={cached!r}, the restarted server must hit "
                        "the disk tier (cached > 0)",
                    )
            elif phase in ("warm", "turn2", CONC_PART, MIXED_PART):
                if isinstance(cached, bool) or not isinstance(cached, int):
                    res.fail(cell, f"r{cf.rep} {phase}: cached count not recorded")
            else:
                res.fail(cell, f"r{cf.rep}: unknown phase {phase!r}")
    return res


def check_gpu_exclusive_rotation(run: Run) -> GuardResult:
    """Guard 9: each job ran alone on the GPU; engines interleave in rotated order per rep."""
    res = GuardResult("gpu_exclusive_rotation")
    slots = defaultdict(list)  # (group, rep) -> [(start, end, engine, cell)]
    for cf in run.files:
        h = cf.header
        if h is None:
            continue
        cell = cf.cell["cell"]
        res.checked += 1
        if h.get("gpu_exclusive") is not True:
            res.fail(cell, f"r{cf.rep}: gpu_exclusive not recorded True")
        if h.get("contended") is not False:
            res.fail(cell, f"r{cf.rep}: contended not recorded False")
        s, e = h.get("job_start_s"), h.get("job_end_s")
        if not (is_num(s) and is_num(e) and s < e):
            res.fail(cell, f"r{cf.rep}: job timestamps missing or not start < end")
            continue
        slots[(cf.cell["group"], cf.rep)].append((s, e, cf.cell["engine"], cell))
    for (group, rep), items in sorted(slots.items()):
        res.checked += 1
        items.sort()
        anchor = items[0][3]
        for prev, nxt in zip(items, items[1:], strict=False):
            if nxt[0] < prev[1]:
                res.fail(nxt[3], f"rep {rep}: overlaps {prev[2]} on the GPU")
        order = run.manifest.get("groups", {}).get(group, {}).get("order")
        if not order:
            res.fail(anchor, f"group {group} has no interleave order in manifest")
            continue
        # Per engine: consecutive cells of one engine are one visit (an engine with two cells
        # runs them back to back), so the observed interleave is the engine runs in time order.
        observed = []
        for i in items:
            if not observed or observed[-1] != i[2]:
                observed.append(i[2])
        expected = rotate(order, rep)
        if observed != expected:
            res.fail(anchor, f"rep {rep}: interleave {observed} != rotated {expected}")
    return res


def check_output_sha(run: Run) -> GuardResult:
    """Guard 10: output sha recorded per request; >=2 prompts per (model, ctx, kind)."""
    res = GuardResult("output_sha")
    for cf in run.files:
        for r in cf.rows:
            res.checked += 1
            if not re.fullmatch(r"[0-9a-f]{16,64}", str(r.get("sha"))):
                res.fail(
                    cf.cell["cell"],
                    f"r{cf.rep} {r.get('phase')}: output sha not recorded",
                )
    prompts = defaultdict(set)
    cells_of = defaultdict(list)
    for c in run.manifest["cells"]:
        key = (c["model"], c["ctx"], c["kind"])
        prompts[key].add(c["prompt_id"])
        cells_of[key].append(c["cell"])
    for key, ids in prompts.items():
        res.checked += 1
        if len(ids) < 2:
            for cid in cells_of[key]:
                res.fail(cid, f"only {len(ids)} prompt for {key}; need >= 2 per kind")
    return res


def check_family_validity(run: Run) -> GuardResult:
    """Guard 12: every reply of a request-family cell (tool-chat / cjk / json-schema, thinking on,
    image input) is valid: tool JSON and schema, JSON-schema conformance, CJK text, the reasoning
    split, accepted image parts. A row flagged invalid, or one missing the flag, fails closed."""
    import fullcmp_manifest as fm

    res = GuardResult("family_validity")
    for cf in run.files:
        c = cf.cell
        if not fm.is_family_cell(c.get("kind"), c.get("thinking"), c.get("media")):
            continue
        for r in cf.rows:
            res.checked += 1
            if r.get("valid") is not True:
                why = r.get("invalid_reason") or "validity not recorded"
                res.fail(c["cell"], f"r{cf.rep} {r.get('phase')}: family reply invalid: {why}")
    # zero checked rows pass vacuously only when the manifest has no family cell; a family cell in
    # the manifest with no audited rows means its replies were never validated, which fails closed
    family_cells = [
        c["cell"]
        for c in (run.manifest.get("cells") or [])
        if fm.is_family_cell(c.get("kind"), c.get("thinking"), c.get("media"))
    ]
    if res.checked == 0 and family_cells:
        res.fail(
            family_cells[0],
            f"{len(family_cells)} family cell(s) in the manifest but no family rows were audited",
        )
    return res


def check_measured_only(run: Run) -> GuardResult:
    """Guard 11: every number comes from a complete raw row; errors and truncations fail."""
    res = GuardResult("measured_only")
    res.violations += list(run.parse_errors)
    for cf in run.files:
        for r in cf.rows:
            res.checked += 1
            cell = cf.cell["cell"]
            if "error" in r:
                res.fail(cell, f"r{cf.rep} {r.get('phase')}: error row ({r['error']})")
            for k in ("ttft_s", "total_s", "dec_tps"):
                if not is_num(r.get(k)):
                    res.fail(
                        cell,
                        f"r{cf.rep} {r.get('phase')}: {k} is not a measured number ({r.get(k)!r})",
                    )
    return res


def check_fairness_controls(run: Run) -> GuardResult:
    """Guard 12: greedy sampling, isolated HOME per engine, A/A and injected-delay controls."""
    res = GuardResult("fairness_controls")
    ctl = run.manifest.get("controls") or {}
    aa, noise = ctl.get("aa_delta_pct"), ctl.get("aa_noise_pct")
    if run.manifest.get("smoke") is True:
        # smoke: a harness check, not a table; the controls come from the ctl-1103 runs, not here
        res.checked += 1
    else:
        res.checked += 1
        if not (is_num(aa) and is_num(noise) and abs(aa) <= noise):
            res.fail("controls", f"A/A delta {aa!r} not inside noise {noise!r}")
        res.checked += 1
        if ctl.get("injected_delay_detected") is not True:
            res.fail("controls", "injected delay was not detected by the harness")
    # HOME is isolated per server process. The harness starts one server per gpuq job (a chunk of
    # cells cold-starts its server again), so the job id is the server key: each job has exactly one
    # HOME, no HOME is the real one, and no two servers share a HOME. Several servers of one engine
    # (one per job) are expected; each gets its own HOME.
    homes = defaultdict(set)  # job_id -> {home}
    for cf in run.files:
        for r in cf.rows:
            res.checked += 1
            t = r.get("temperature")
            if not (is_num(t) and t == 0):
                res.fail(cf.cell["cell"], f"temperature {t!r}; greedy (0) required")
        if cf.header is not None:
            homes[cf.header.get("job_id")].add(cf.header.get("home"))
    real_home = os.path.expanduser("~")
    owner = {}  # home -> job_id
    for job, hs in sorted(homes.items(), key=lambda kv: str(kv[0])):
        res.checked += 1
        if job is None or len(hs) != 1 or None in hs:
            res.fail(
                f"job {job}",
                f"HOME not one isolated directory per server: {sorted(map(str, hs))}",
            )
            continue
        (home,) = hs
        if os.path.abspath(str(home)) == os.path.abspath(real_home):
            res.fail(f"job {job}", "server uses the real HOME, not an isolated one")
            continue
        if home in owner and owner[home] != job:
            res.fail(f"job {job}", f"HOME shared with server job {owner[home]}")
        owner[home] = job
    return res


def check_telemetry(run: Run) -> GuardResult:
    """Guard 13: every cell has a telemetry series; flags thermal throttling and swap growth.

    Missing telemetry, a series without samples, or an unreadable file is a violation (unless the
    manifest sets "telemetry_required": false). A cell whose sampler saw a thermal throttle
    (CPU_Speed_Limit < 100 or a recorded thermal/performance warning, at start, end or any sample)
    is flagged. A cell whose swap used grew during the cell (last sample minus first) is flagged.
    """
    import fullcmp_telemetry as tm

    res = GuardResult("telemetry")
    required = run.manifest.get("telemetry_required", True)
    for cf in run.files:
        cell = cf.cell["cell"]
        tpath = telemetry_path(cf.path)
        if not tpath.is_file():
            if required:
                res.fail(cell, f"r{cf.rep}: missing telemetry file {tpath.name}")
            continue
        res.checked += 1
        try:
            header, samples, marks, footer = tm.read_telemetry(tpath)
        except (OSError, json.JSONDecodeError) as exc:
            res.fail(cell, f"r{cf.rep}: telemetry unreadable ({exc})")
            continue
        if header is None:
            res.fail(cell, f"r{cf.rep}: telemetry has no header")
        if footer is None:
            res.fail(
                cell,
                f"r{cf.rep}: telemetry has no footer (sampler did not stop cleanly)",
            )
        if not samples:
            res.fail(cell, f"r{cf.rep}: telemetry has no samples")
            continue
        flagged = []
        for where, th in (
            ("start", (header or {}).get("thermal_start")),
            ("end", (footer or {}).get("thermal_end")),
        ):
            if (th or {}).get("throttled"):
                flagged.append(where)
        if any((smp.get("thermal") or {}).get("throttled") for smp in samples):
            flagged.append("sample")
        if flagged:
            res.fail(
                cell, f"r{cf.rep}: thermal throttling flagged ({', '.join(flagged)})"
            )
        growth = tm.summarize(samples, marks)["overall"].get("swap_used_delta_bytes")
        if growth is not None and growth > 0:
            res.fail(cell, f"r{cf.rep}: swap grew by {growth} bytes during the cell")
    return res


def quiet_limits_of(rec: dict) -> tuple[float, float]:
    """(single-holder limit, total limit) a snapshot was taken against (recorded in it), else the
    gate defaults."""
    import fullcmp_quiet as fq

    single = rec.get("max_single_pct")
    total = rec.get("max_total_pct")
    return (
        float(single) if is_num(single) else fq.DEFAULT_MAX_SINGLE_PCT,
        float(total) if is_num(total) else fq.DEFAULT_MAX_TOTAL_PCT,
    )


def check_quiet_machine(run: Run) -> GuardResult:
    """Guard 14: every cell's quiet_pre and quiet_post snapshot (session header) exists, its total
    foreign CPU is below the total limit, and no non-system holder sustained >= the single-holder
    limit (both recorded in the snapshot; re-derived from `holders`, not trusted from `ok`). A
    missing snapshot is a violation (fail closed); the holders are named in the message."""
    res = GuardResult("quiet_machine")
    for cf in run.files:
        cell = cf.cell["cell"]
        res.checked += 1
        head = cf.header or {}
        for key in ("quiet_pre", "quiet_post"):
            rec = head.get(key)
            total = rec.get("total_foreign_pct") if isinstance(rec, dict) else None
            if not is_num(total):
                res.fail(cell, f"r{cf.rep}: no {key} snapshot")
                continue
            if run.manifest.get("stub_engine") is True:
                # stub_engine (test only): the engine is a stub, so the host's CPU load says
                # nothing about a measurement; the e2e tests run beside other pytest and agent
                # processes. The snapshot must still exist (fail closed above).
                continue
            single, limit = quiet_limits_of(rec)
            noise = {"WindowServer", "kernel_task", "mds_stores"}
            violators = [
                h
                for h in rec.get("holders") or []
                if is_num(h.get("pct"))
                and h["pct"] >= single
                and h.get("name") not in noise
            ]
            if total >= limit or violators:
                holders = ", ".join(
                    f"{h.get('name')} (pid {h.get('pid')}, {h.get('pct')}%)"
                    for h in violators
                ) or "no single holder"
                res.fail(
                    cell,
                    f"r{cf.rep}: {key} foreign CPU {total}% (limit {limit}%), "
                    f"holder limit {single}%: {holders}",
                )
    return res


ROUND_OUTLIER_RATIO = 1.5


def round_ms(row: dict):
    """Per-round decode time of one speculative request: xy.decode_ms / xy.speculative.rounds.
    None when the row has no speculative round count (not comparable, not flagged)."""
    xy = row.get("xy") or {}
    rounds = (xy.get("speculative") or {}).get("rounds")
    dec = xy.get("decode_ms")
    if is_num(rounds) and rounds > 0 and is_num(dec) and dec > 0:
        return dec / rounds
    return None


def round_series(rows) -> dict:
    """{(phase, ctx, kind, occurrence): round_ms} for one rep's decode rows. Shared with
    fullcmp_reps, which applies the same rule to 2-rep cells (rep-3 trigger)."""
    seen: dict = {}
    out = {}
    for row in rows:
        if "phase" not in row:
            continue
        base = (row.get("phase"), row.get("ctx"), row.get("kind"))
        n = seen.get(base, 0)
        seen[base] = n + 1
        val = round_ms(row)
        if val is not None:
            out[base + (n,)] = val
    return out


def check_round_time_outlier(run: Run) -> GuardResult:
    """Guard 15: a request whose per-round time differs by more than 1.5x from the median of the
    same request's other reps is flagged (these cells feed the rep-3 rule). Needs another rep
    with a round count; requests without speculative round counts are not compared."""
    import statistics

    res = GuardResult("round_time_outlier")
    by_cell: dict = defaultdict(dict)
    for cf in run.files:
        by_cell[cf.cell["cell"]][cf.rep] = round_series(cf.rows)
    for cell, reps in by_cell.items():
        res.checked += 1
        keys = {k for series in reps.values() for k in series}
        for key in sorted(keys, key=str):
            vals = {rep: s[key] for rep, s in reps.items() if key in s}
            for rep, v in sorted(vals.items()):
                others = [w for r2, w in vals.items() if r2 != rep]
                if not others:
                    continue
                m = statistics.median(others)
                ratio = max(v / m, m / v)
                if ratio > ROUND_OUTLIER_RATIO:
                    res.fail(
                        cell,
                        f"r{rep}: {key[0]} ctx{key[1]} {key[2]} per-round {v:.2f} ms is "
                        f"{ratio:.2f}x the median of the other reps ({m:.2f} ms)",
                    )
    return res


def server_log_slice_path(cell_path) -> Path:
    """<cell>-rN.server.log beside <cell>-rN.jsonl (tfbench.server_log_path, kept in step)."""
    text = str(cell_path)
    stem = text[: -len(".jsonl")] if text.endswith(".jsonl") else text
    return Path(stem + ".server.log")


def server_log_slice_problem(cell_path):
    """None when the cell's engine-log slice exists and is non-empty, else the reason (fail closed)."""
    path = server_log_slice_path(cell_path)
    if not path.is_file():
        return f"engine log slice missing: {path}"
    if path.stat().st_size == 0:
        return f"engine log slice empty: {path}"
    return None


def check_attribution_fields(run: Run) -> GuardResult:
    """Guard 16 (lead ruling 2026-11-03, two layers).
    Layer 1: every timed row (decode and conc) carries the layer-1 fields for its engine, each as a
    value or an explicit 'unavailable (reason)'. A MISSING field fails the cell (fail closed).
    Observe-only fields are never required here.
    Layer 2: when run.profile_root is set, each cell's profile rep (<profile_root>/<engine>/
    <cell>-prof.jsonl) must carry the layer-2 fields for its engine. In full mode (run.full) a cell
    with no profile rep is a violation 'profile rep missing'."""
    import fullcmp_attribution as fa_attr

    res = GuardResult("attribution_fields")
    for cf in run.files:
        cell = cf.cell["cell"]
        res.checked += 1
        engine = cf.cell.get("engine")
        try:
            result = fa_attr.check_cell(engine, cf.rows, None)
        except KeyError as exc:
            res.fail(cell, f"r{cf.rep}: {exc}")
            continue
        miss = [m for m in result["missing"] if not m.startswith("profile:")]
        if not cf.rows:
            res.fail(cell, f"r{cf.rep}: no decode rows")
        elif miss:
            res.fail(cell, f"r{cf.rep}: attribution fields MISSING: {', '.join(miss)}")
        slice_problem = server_log_slice_problem(cf.path)
        if slice_problem:
            res.fail(cell, f"r{cf.rep}: {slice_problem}")
    if run.profile_root is None:
        return res
    seen = set()
    for cf in run.files:
        cell, engine = cf.cell["cell"], cf.cell.get("engine")
        if (engine, cell) in seen:
            continue
        seen.add((engine, cell))
        path = Path(run.profile_root) / str(engine) / f"{cell}-prof.jsonl"
        if not path.is_file():
            if run.full:
                res.fail(cell, "profile rep missing: no " + str(path))
            continue
        slice_problem = server_log_slice_problem(path)
        if slice_problem:
            res.fail(cell, f"profile rep: {slice_problem}")
        rows, _errs = _read_jsonl(path)
        prof = next((r for r in rows if r.get("part") == "profile"), None)
        if prof is None:
            res.fail(cell, "profile rep missing: no part=profile row in " + str(path))
            continue
        res.checked += 1
        try:
            cov = fa_attr.check_profile(engine, prof)
        except KeyError as exc:
            res.fail(cell, f"profile rep: {exc}")
            continue
        miss2 = fa_attr.missing(cov)
        if miss2:
            res.fail(cell, f"profile rep lacks layer 2: {', '.join(miss2)}")
    return res


BEHAVIOUR_DIR = "behaviour"  # <root>/behaviour/<engine>.jsonl: B4 checks (fullcmp_behaviour.py)


def check_behaviour(run: Run) -> GuardResult:
    """Guard 18 (B4): each engine in manifest["behaviour"] = {engine: {check: "pass"|"unsupported"}}
    has one file <root>/behaviour/<engine>.jsonl with a row per expected check. The status must be
    the expected one: a check expected to pass that fails (or is unsupported) is a violation, and
    so is a route the census calls unsupported that answers. A missing file or check fails closed.
    A behaviour file for an engine the manifest does not plan is a violation too."""
    res = GuardResult("behaviour")
    expected = run.manifest.get("behaviour") or {}
    for eng, want in expected.items():
        res.checked += 1
        rows = run.behaviour.get(eng)
        if rows is None:
            res.fail(eng, f"behaviour file missing: {BEHAVIOUR_DIR}/{eng}.jsonl")
            continue
        by_check = {r.get("check"): r for r in rows if r.get("part") == "behaviour"}
        for check, status in want.items():
            res.checked += 1
            row = by_check.get(check)
            if row is None:
                res.fail(f"{eng}.{check}", "behaviour check not recorded")
            elif row.get("status") != status:
                reason = row.get("reason") or json.dumps(row.get("detail"), sort_keys=True)
                res.fail(
                    f"{eng}.{check}",
                    f"status {row.get('status')!r}, expected {status!r}: {reason}",
                )
    for eng in run.behaviour:
        if eng not in expected:
            res.checked += 1
            res.fail(eng, f"unplanned behaviour file {BEHAVIOUR_DIR}/{eng}.jsonl")
    return res


def check_usage_fields(run: Run) -> GuardResult:
    """Guard 0 (lead ruling 2026-11-03): every request row carries a real completion count and
    prompt count (int, never None), and a cached-token count that is an int or an explicit
    'cached_unavailable' reason. Conc requests also carry the record aggregates: the summary of
    the per-request decode rates (dec_min / dec_max / dec_median / fair_max_min) must match the
    per-request lists, and agg_tps must equal total_tokens / wall_s (1 %)."""
    res = GuardResult("usage_fields")
    for cf in run.files:
        cell = cf.cell["cell"]
        for r in cf.rows:
            if r.get("part") not in ("decode", CONC_PART, MIXED_PART):
                continue
            res.checked += 1
            for key in ("ct", "pt"):
                v = r.get(key)
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    res.fail(cell, f"r{cf.rep}: {key}={v!r} is not a recorded count")
            cached = r.get("cached")
            if isinstance(cached, bool) or (cached is not None and not isinstance(cached, int)):
                res.fail(cell, f"r{cf.rep}: cached={cached!r} is not a count")
            if cached is None and not str(r.get("cached_unavailable") or "").strip():
                res.fail(cell, f"r{cf.rep}: cached is None with no cached_unavailable reason")
            if r.get("part") in (CONC_PART, MIXED_PART):
                for key in ("dec_min", "dec_max", "dec_median", "fair_max_min", "agg"):
                    if r.get(key) is None:
                        res.fail(cell, f"r{cf.rep}: conc aggregate {key} is missing or None")
                decs = r.get("per_req_dec_all")
                if isinstance(decs, list) and decs and all(isinstance(d, (int, float)) for d in decs):
                    if r.get("dec_min") != min(decs) or r.get("dec_max") != max(decs):
                        res.fail(cell, f"r{cf.rep}: conc dec_min/dec_max differ from per-request decode")
                    if r.get("dec_median") != statistics.median(decs):
                        res.fail(cell, f"r{cf.rep}: conc dec_median differs from per-request decode")
                    if r.get("fair_max_min") != round(max(decs) / min(decs), 3):
                        res.fail(cell, f"r{cf.rep}: conc fair_max_min differs from max/min")
                agg = r.get("agg") or {}
                if agg.get("wall_s") and agg.get("total_tokens") is not None and agg.get("agg_tps") is not None:
                    want = agg["total_tokens"] / agg["wall_s"]
                    if abs(agg["agg_tps"] - want) > 0.01 * max(want, 1e-9) + 0.05:
                        res.fail(cell, f"r{cf.rep}: conc agg_tps {agg['agg_tps']} != total/wall {want:.1f}")
    return res


def check_mixed_same_prompt(run: Run) -> GuardResult:
    """Guard (mixed load and same-prompt conc): a mixed record holds exactly one long request and
    n-1 short ones, and a same-prompt conc record sends one prompt n times (one prompt_ids_sha)."""
    res = GuardResult("mixed_same_prompt")
    for cf in run.files:
        cell = cf.cell["cell"]
        res.checked += 1  # every cell file is checked, mixed or not
        mixed = [r for r in cf.rows if r.get("part") == MIXED_PART]
        groups = defaultdict(list)
        for r in mixed:
            res.checked += 1
            if r.get("role") not in MIXED_ROLES:
                res.fail(cell, f"r{cf.rep} mixed: role {r.get('role')!r} is not long/short")
            groups[r.get("trial")].append(r)
        for trial, rows in groups.items():
            longs = [r for r in rows if r.get("role") == "long"]
            if len(longs) != 1 or len(rows) != rows[0].get("n"):
                res.fail(
                    cell,
                    f"r{cf.rep} mixed trial {trial}: {len(longs)} long of {len(rows)} requests, "
                    f"record n={rows[0].get('n')}",
                )
        for r in cf.rows:
            if r.get("part") == CONC_PART and r.get("same_prompt") is True:
                res.checked += 1
                if r.get("prompt_ids_sha") is None:
                    res.fail(cell, f"r{cf.rep} same-prompt conc: prompt_ids_sha missing")
        same = defaultdict(set)
        for r in cf.rows:
            if r.get("part") == CONC_PART and r.get("same_prompt") is True:
                same[r.get("trial")].add(r.get("prompt_ids_sha"))
        for trial, shas in same.items():
            if len(shas) != 1:
                res.fail(cell, f"r{cf.rep} same-prompt conc trial {trial}: {len(shas)} prompts")
    return res


GUARD_CHECKS = (
    check_usage_fields,
    check_prompt_identity,
    check_provenance_identity,
    check_pinned_versions,
    check_manifest_completeness,
    check_record_count,
    check_engaged_mode,
    check_observe_state,
    check_weight_identity,
    check_cold_cache,
    check_gpu_exclusive_rotation,
    check_output_sha,
    check_family_validity,
    check_measured_only,
    check_fairness_controls,
    check_telemetry,
    check_quiet_machine,
    check_attribution_fields,
    check_round_time_outlier,
    check_behaviour,
    check_mixed_same_prompt,
)
# the guard names, in run order (fullcmp_audit's report and its tests read these)
GUARDS = tuple(check.__name__.removeprefix("check_") for check in GUARD_CHECKS)


def run_guards(root, manifest_path, profile_root=None, full=False):
    if full and profile_root is None:
        raise ValueError("full mode needs --profile-root (layer-2 profile rows)")
    manifest = json.loads(Path(manifest_path).read_text())
    run = load_run(root, manifest)
    run.profile_root = profile_root
    run.full = full
    return run, [check(run) for check in GUARD_CHECKS]


def violations_by_cell(results) -> dict:
    out = defaultdict(list)
    for res in results:
        for v in res.violations:
            parts = v.split(": ", 2)
            if len(parts) == 3:
                out[parts[1]].append(v)
    return out


# ---------------------------------------------------------------------------
# --recompute: rebuild every table number from raw rows and diff against the table
# ---------------------------------------------------------------------------
@dataclass
class TableEntry:
    model: str
    metric: str
    engine: str
    column: str
    text: str
    line: int


def _clean(cell: str) -> str:
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)  # [text](link) -> text
    return cell.replace("*", "").replace("`", "").strip()


def parse_table(text: str):
    entries, problems = [], []
    model = metric = header = None
    skip = False
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            header, skip = None, "derived" in line.lower()
            parts = re.split(r"\s+[—-]\s+", line.lstrip("#").strip(), maxsplit=1)
            if skip:
                model = metric = None
            elif len(parts) != 2:
                model = metric = None
                problems.append(f"line {no}: heading must be '<model> — <metric>'")
            else:
                model, metric = parts[0].strip(), parts[1].strip()
                if metric not in METRICS:
                    problems.append(f"line {no}: unknown metric {metric!r}")
                    model = metric = None
            continue
        if not line.startswith("|") or model is None:
            continue
        cells = [_clean(c) for c in line.strip("|").split("|")]
        if all(c and set(c) <= set("-: ") for c in cells):
            continue  # markdown separator row
        if header is None:
            header = cells
            if not header or header[0].lower() != "engine":
                problems.append(f"line {no}: table header must start with 'engine'")
                header = None
            continue
        if header is None:
            continue
        for col, txt in zip(header[1:], cells[1:], strict=False):
            entries.append(TableEntry(model, metric, cells[0], col, txt, no))
    return entries, problems


def recompute_metric(run: Run, cell: dict, metric: str):
    """Median over reps of the per-rep median, and spread (max - min over reps)."""
    phase, fld = METRICS[metric]
    rep_vals = []
    for cf in run.files:
        if cf.cell["cell"] != cell["cell"] or cf.cell["engine"] != cell["engine"]:
            continue
        vals = [
            r[fld] for r in cf.rows if r.get("phase") == phase and is_num(r.get(fld))
        ]
        if not vals:
            raise ValueError(f"no {phase} {fld} rows in r{cf.rep}")
        rep_vals.append(statistics.median(vals))
    if not rep_vals:
        raise ValueError("no raw files")
    return statistics.median(rep_vals), max(rep_vals) - min(rep_vals)


def recompute_table(run: Run, table_text: str, by_cell: dict):
    """Returns (checked, violations) for every table number against the raw jsonl."""
    entries, problems = parse_table(table_text)
    out = [f"recompute: table: {p}" for p in problems]
    checked = 0
    index = {(e.engine, e.model, e.column, e.metric): e for e in entries}
    planned = {(c["engine"], c["model"], c["column"]) for c in run.manifest["cells"]}
    metrics_of = defaultdict(set)
    for e in entries:
        metrics_of[e.model].add(e.metric)
        if (e.engine, e.model, e.column) not in planned:
            out.append(
                f"measured_only: {e.engine}/{e.column}: table number has no planned cell (line {e.line})"
            )
    for c in run.manifest["cells"]:
        cid = c["cell"]
        viol = by_cell.get(cid, [])
        for metric in sorted(metrics_of.get(c["model"], ())):
            checked += 1
            e = index.get((c["engine"], c["model"], c["column"], metric))
            if e is None:
                out.append(f"manifest_completeness: {cid}: {metric} missing from table")
                continue
            if e.text.lower().startswith("invalid"):
                if not viol:
                    out.append(
                        f"recompute: {cid}: {metric} marked invalid but every guard passes"
                    )
                continue
            m = TABLE_NUM.match(e.text)
            if m is None:
                out.append(
                    f"recompute: {cid}: {metric} is neither a number nor 'invalid: reason' ({e.text!r})"
                )
                continue
            if viol:
                out.append(
                    f"recompute: {cid}: {metric} shows a number but the cell fails guards"
                )
                continue
            try:
                med, spread = recompute_metric(run, c, metric)
            except ValueError as exc:
                out.append(f"measured_only: {cid}: {metric}: {exc}")
                continue
            if abs(float(m.group(1)) - round(med, 1)) > 0.051:
                out.append(
                    f"recompute: {cid}: {metric} table {m.group(1)} != raw median {med:.3f}"
                )
            if (
                m.group(2) is not None
                and abs(float(m.group(2)) - round(spread, 1)) > 0.051
            ):
                out.append(
                    f"recompute: {cid}: {metric} table spread {m.group(2)} != raw spread {spread:.3f}"
                )
    return checked, out


# ---------------------------------------------------------------------------
# Derived efficiency (labeled DERIVED everywhere: never a measured value)
# A decode ROUND reads the weights once and verifies R rows (R = 1 for plain autoregressive decode;
# R = draft rows for MTP / DFlash speculative decode). Tokens per round T = tokens / rounds.
#   rounds/s      = tok/s / T
#   bytes/round   = weight bytes + KV bytes read (at avg ctx) + GDN state bytes
#   MBU           = bytes/round x rounds/s / peak BW          (peak = stable GEMV statistic)
#   FLOPs/round   = R x (2 x active params + attention FLOPs at avg ctx)
#   MFU           = FLOPs/round x rounds/s / peak FLOP/s       (peak = measured bf16 27B prefill)
# T and R come from the request row / session header when the record carries them. If T is absent
# for a speculative engine, MBU/MFU are None ("unavailable (tokens_per_round missing)"), never a
# guess. If R is absent, T is used as a labeled LOWER BOUND for R.
# Inputs are measured (tok/s, prompt/completion tokens, ttft) or come from the tables below.
# A value whose input is not sourced is None with the reason, never a guess.
# ---------------------------------------------------------------------------
PEAKS_PATH = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/peak_measured.json"
)

# Spec-sheet values: SECONDARY only (column "vs spec sheet"). Never the MBU/MFU denominator.
SPEC_SHEET = {
    "M5 Max": {
        "bw_gbs": 560.0,
        "bw_source": "IMPLIED: docs/research/ctx1m/ROOFLINE.md (560 GB/s implied by the measured "
        "29 ms 27B floor). Not an Apple spec figure.",
        "tflops_bf16": None,
        "flops_source": "UNSOURCED: no peak FLOP/s figure in the repo or the spec.",
    },
}


class MeasuredPeaksError(RuntimeError):
    """The measured-peaks file is absent or invalid. MBU/MFU are refused, never estimated."""


def load_measured_peaks(path=PEAKS_PATH) -> dict:
    """Denominators from scripts/research/probe_peak.py (measured on this machine, gpuq job).

    decode_gbs  = 4-bit quantized GEMV at batch 1, STABLE statistic = top_half_median (MBU denominator;
                  the record's "stable_stat" names it). max / median / p10-p90 spread kept for reference.
    bf16_tflops = max of the bf16 matmul at the 27B prefill shape (MFU denominator)
    read_gbs    = max of the sustained read bandwidth (reference only, not a denominator)
    Fails closed with MeasuredPeaksError when the file is missing, invalid, or lacks a needed result
    (a GEMV result without the stable statistic means the file predates the stable probe).
    """
    import probe_peak as pp  # scripts/research on sys.path (script dir or test insert)

    p = Path(path)
    if not p.is_file():
        raise MeasuredPeaksError(
            f"measured peaks file missing: {p}. Run scripts/research/probe_peak.py through gpuq "
            "first; MBU/MFU do not fall back to the spec sheet."
        )
    try:
        rep = json.loads(p.read_text())
        pp.validate_report(rep)
    except (ValueError, json.JSONDecodeError) as exc:
        raise MeasuredPeaksError(f"measured peaks file invalid: {p}: {exc}") from exc
    by = {r["name"]: r for r in rep["results"]}
    needed = ("qgemv_w4g64_M1", "matmul_prefill27b_bf16", "bw_read_sum_bf16")
    missing = [n for n in needed if n not in by]
    if missing:
        raise MeasuredPeaksError(f"measured peaks file lacks {missing}: {p}")
    if "stable" not in by["qgemv_w4g64_M1"] or not (by["qgemv_w4g64_M1"]["stable"] > 0):
        raise MeasuredPeaksError(
            f"measured peaks file has no stable GEMV statistic (predates probe_peak "
            f"robust stats): {p}"
        )

    def pick(name):
        r = by[name]
        return {
            "max": r["max"],
            "median": r["median"],
            "spread_pct": r["spread_pct"],
            "n": r["n"],
        }

    g = by["qgemv_w4g64_M1"]
    decode = {
        **pick("qgemv_w4g64_M1"),
        "stable": g["stable"],
        "stable_stat": g["stable_stat"],
        "p10": g["p10"],
        "p90": g["p90"],
        "spread_p10_p90_pct": g["spread_p10_p90_pct"],
    }
    return {
        "source": str(p),
        "label": rep["label"],
        "mlx_version": rep["mlx_version"],
        "chip": rep["chip"],
        "date": rep["date"],
        "decode_gbs": decode,
        "bf16_tflops": pick("matmul_prefill27b_bf16"),
        "read_gbs": pick("bw_read_sum_bf16"),
    }


# Model table. Values from the checkpoints' config.json (text_config) unless noted.
MODELS = {
    "q27-4bit": {
        "name": "Qwen3.8-27B-oQ4e-mtp",
        "active_params": 27.74e9,  # task spec: 27.74B text params (dense: all active)
        "weight_bytes": 15.74e9,  # task spec: 15.74 GB weights (oQ4e)
        "n_layers": 64,
        "full_attn_layers": 16,  # layer_types == full_attention (every 4th)
        "linear_layers": 48,
        "n_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "kv_dtype_bytes": 2,  # bf16 KV (int8 KV is a separate lossy option)
        "gdn_state": {"heads": 48, "key_dim": 128, "value_dim": 128, "dtype_bytes": 4},
        "config": "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp/config.json",
    },
    "flashnext": {
        "name": "Qwen3.8-Flash-Next-oQ3e-mtp",
        # NOT in the repo: MoE (512 experts, top-10) active-parameter count and per-token expert
        # bytes are not recorded. MBU/MFU stay None until they are sourced.
        "active_params": None,
        "weight_bytes": None,
        "n_layers": 48,
        "full_attn_layers": 12,
        "linear_layers": 36,
        "n_heads": 24,
        "kv_heads": 2,
        "head_dim": 256,
        "kv_dtype_bytes": 2,
        "gdn_state": None,  # not extracted; KV term excludes the indexer (indexer_kv_heads=1, dim 128)
        "config": "/Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ3e-mtp/config.json",
    },
}


def arch_from_config(path) -> dict:
    """Layer counts from a checkpoint config.json (text_config), for the model table check."""
    cfg = json.loads(Path(path).read_text())
    tc = cfg.get("text_config", cfg)
    types = tc.get("layer_types") or []
    return {
        "n_layers": tc.get("num_hidden_layers"),
        "full_attn_layers": sum(1 for t in types if t == "full_attention"),
        "linear_layers": sum(1 for t in types if t == "linear_attention"),
        "n_heads": tc.get("num_attention_heads"),
        "kv_heads": tc.get("num_key_value_heads"),
        "head_dim": tc.get("head_dim"),
        "gdn_state": {
            "heads": tc.get("linear_num_value_heads"),
            "key_dim": tc.get("linear_key_head_dim"),
            "value_dim": tc.get("linear_value_head_dim"),
            "dtype_bytes": 4 if str(tc.get("mamba_ssm_dtype", "")) == "float32" else 2,
        }
        if tc.get("linear_num_value_heads")
        else None,
    }


def kv_bytes_per_token(model: dict, ctx: float) -> float:
    """KV bytes READ for one decode step at context length ctx (full-attention layers only)."""
    return (
        model["full_attn_layers"]
        * 2
        * model["kv_heads"]
        * model["head_dim"]
        * model["kv_dtype_bytes"]
        * ctx
    )


def gdn_state_bytes(model: dict) -> float | None:
    g = model.get("gdn_state")
    if not g:
        return None
    return (
        model["linear_layers"]
        * g["heads"]
        * g["key_dim"]
        * g["value_dim"]
        * g["dtype_bytes"]
    )


def attn_flops_per_token(model: dict, ctx: float) -> float:
    """Attention matmul FLOPs for one token at context ctx: QK^T and AV, 4 * heads * head_dim * ctx per layer."""
    return model["full_attn_layers"] * 4 * model["n_heads"] * model["head_dim"] * ctx


AR_MODES = frozenset({"", "off", "none", "ar", "autoregressive"})


def is_speculative(engaged_mode) -> bool | None:
    """True / False from the engaged spec mode string; None when the mode is not recorded."""
    if not isinstance(engaged_mode, str):
        return None
    return engaged_mode.strip().lower() not in AR_MODES


def derived_decode(
    model_id: str,
    peaks: dict,
    tok_s: float,
    pt: int,
    ct: int,
    hw: str = "M5 Max",
    speculative: bool | None = False,
    tokens_per_round: float | None = None,
    rows_per_round: float | None = None,
) -> dict:
    """MBU and MFU for decode at the average context (pt + ct/2). Labeled DERIVED.

    Per round (see the formula block above): weights are read once per round, not once per token.
    tokens_per_round: measured tokens / rounds from the record. None for plain AR decode means 1;
    None for a speculative engine means unavailable (reported as None, never estimated).
    mbu uses the stable 4-bit GEMV statistic (peaks["decode_gbs"]["stable"]); mbu_vs_spec is the
    same bytes against the spec-sheet bandwidth, kept only as a labeled secondary column.
    """
    m = MODELS[model_id]
    ctx = pt + ct / 2.0
    out = {
        "label": "DERIVED",
        "ctx_avg": ctx,
        "tokens_per_round": None,
        "tokens_per_round_src": None,
        "mbu": None,
        "mbu_vs_spec": None,
        "mfu": None,
        "reason": [],
    }
    if tokens_per_round is not None:
        if not (is_num(tokens_per_round) and tokens_per_round >= 1):
            out["reason"].append(f"tokens_per_round invalid: {tokens_per_round!r}")
            return _unavailable_decode(m, peaks, hw, tok_s, ctx, out)
        tpr, out["tokens_per_round_src"] = float(tokens_per_round), "record"
    elif speculative is False:
        tpr, out["tokens_per_round_src"] = 1.0, "autoregressive (1 token per round)"
    else:
        out["reason"].append("unavailable (tokens_per_round missing)")
        return _unavailable_decode(m, peaks, hw, tok_s, ctx, out)
    out["tokens_per_round"] = tpr
    rounds_s = tok_s / tpr
    out["rounds_per_s"] = rounds_s
    if m["weight_bytes"] is None:
        out["reason"].append("weight bytes per token not in repo")
    else:
        kv_gdn = kv_bytes_per_token(m, ctx) + (gdn_state_bytes(m) or 0)
        bytes_round = m["weight_bytes"] + kv_gdn
        out["bytes_per_round"] = bytes_round
        stable = peaks["decode_gbs"]["stable"]
        out["peak_decode_gbs"] = stable
        out["peak_decode_stat"] = peaks["decode_gbs"]["stable_stat"]
        out["mbu"] = bytes_round * rounds_s / (stable * 1e9)
        spec = SPEC_SHEET[hw]["bw_gbs"]
        out["mbu_vs_spec"] = bytes_round * rounds_s / (spec * 1e9)
    out.update(_mfu(m, peaks, hw, rounds_s, ctx, out, rows_per_round, tpr))
    return out


def _unavailable_decode(m, peaks, hw, tok_s, ctx, out) -> dict:
    out.update(_mfu(m, peaks, hw, None, ctx, out, None, None))
    out["mbu"] = None
    out["mbu_vs_spec"] = None
    return out


def derived_prefill(
    model_id: str, peaks: dict, pt: int, ttft_s: float, hw: str = "M5 Max"
) -> dict:
    """MFU for prefill: prompt tokens over ttft (includes the first sampled token, so a slight
    under-estimate). Attention at the average prefill context pt/2. MBU is not defined for
    compute-bound prefill and is not reported."""
    m = MODELS[model_id]
    out = {"label": "DERIVED", "reason": []}
    if not ttft_s or ttft_s <= 0:
        return {**out, "mfu": None, "reason": ["ttft missing"]}
    out.update(_mfu(m, peaks, hw, pt / ttft_s, pt / 2.0, out))
    out["prefill_tok_s"] = pt / ttft_s
    return out


def _mfu(
    m: dict,
    peaks: dict,
    hw: str,
    rounds_s: float | None,
    ctx: float,
    out: dict,
    rows_per_round: float | None = None,
    tokens_per_round: float | None = None,
) -> dict:
    """MFU = FLOPs per round x rounds/s / peak. A round verifies R rows: R from the record
    (rows_per_round); else tokens_per_round as a labeled lower bound; else 1 (single forward)."""
    if m["active_params"] is None:
        out["reason"].append("active params not in repo")
        return {"mfu": None, "mfu_vs_spec": None, "reason": out["reason"]}
    if rounds_s is None:
        return {"mfu": None, "mfu_vs_spec": None, "reason": out["reason"]}
    if rows_per_round is not None:
        if not (is_num(rows_per_round) and rows_per_round >= 1):
            return {
                "mfu": None,
                "mfu_vs_spec": None,
                "reason": out["reason"]
                + [f"rows_per_round invalid: {rows_per_round!r}"],
            }
        rows, rows_src = float(rows_per_round), "record"
    elif tokens_per_round is not None:
        rows, rows_src = float(tokens_per_round), "lower bound (tokens_per_round)"
    else:
        rows, rows_src = 1.0, "1 row per round"
    flops_round = rows * (2 * m["active_params"] + attn_flops_per_token(m, ctx))
    res = {
        "flops_per_round": flops_round,
        "rows_per_round": rows,
        "rows_per_round_src": rows_src,
        "mfu": flops_round * rounds_s / (peaks["bf16_tflops"]["max"] * 1e12),
        "mfu_vs_spec": None,
        "peak_bf16_tflops": peaks["bf16_tflops"]["max"],
        "reason": out["reason"],
    }
    spec_tf = SPEC_SHEET[hw]["tflops_bf16"]
    if spec_tf is None:
        res["reason"] = out["reason"] + ["spec-sheet FLOP/s unsourced"]
    else:
        res["mfu_vs_spec"] = flops_round * rounds_s / (spec_tf * 1e12)
    return res


def derived_rows(run: Run, peaks: dict, hw: str = "M5 Max") -> list:
    """One DERIVED row per (cell, rep, phase) that has a model table entry and measured inputs."""
    rows = []
    for cf in run.files:
        model_id = cf.cell.get("model")
        if model_id not in MODELS:
            continue
        for r in cf.rows:
            pt, ct = r.get("pt"), r.get("ct")
            if not (is_num(pt) and is_num(ct) and ct > 0):
                continue
            base = {"cell": cf.cell["cell"], "rep": cf.rep, "phase": r.get("phase")}
            if is_num(r.get("dec_tps")):
                # tokens/rows per round: the request row wins, then the session header; absent
                # fields stay absent (derived_decode then reports unavailable for spec engines).
                hdr = cf.header or {}
                spec = is_speculative(hdr.get("engaged_spec_mode"))
                tpr = next(
                    (
                        src.get("tokens_per_round")
                        for src in (r, hdr)
                        if src.get("tokens_per_round") is not None
                    ),
                    None,
                )
                rpr = next(
                    (
                        src.get("rows_per_round")
                        for src in (r, hdr)
                        if src.get("rows_per_round") is not None
                    ),
                    None,
                )
                rows.append(
                    {
                        **base,
                        "metric": "decode",
                        **derived_decode(
                            model_id,
                            peaks,
                            r["dec_tps"],
                            int(pt),
                            int(ct),
                            hw,
                            speculative=spec,
                            tokens_per_round=tpr,
                            rows_per_round=rpr,
                        ),
                    }
                )
            if is_num(r.get("ttft_s")):
                rows.append(
                    {
                        **base,
                        "metric": "prefill",
                        **derived_prefill(model_id, peaks, int(pt), r["ttft_s"], hw),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--root", required=True, help="result dir: <root>/<engine>/<cell>-rN.jsonl"
    )
    p.add_argument("--manifest", required=True, help="expected-cell manifest json")
    p.add_argument(
        "--recompute",
        action="store_true",
        help="also recompute the table from raw rows",
    )
    p.add_argument("--table", help="FULLCMP_TABLE.md (required with --recompute)")
    p.add_argument(
        "--derived",
        action="store_true",
        help="print DERIVED MBU/MFU rows (labeled derived; never counted as measured)",
    )
    p.add_argument(
        "--peaks",
        default=str(PEAKS_PATH),
        help="measured peaks json (scripts/research/probe_peak.py); required with --derived",
    )
    p.add_argument(
        "--profile-root",
        help="layer-2 profile rows: <profile-root>/<engine>/<cell>-prof.jsonl (checked when given)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="full mode: every cell needs its profile rep (requires --profile-root)",
    )
    a = p.parse_args(argv)
    if a.full and not a.profile_root:
        p.error("--full needs --profile-root")
    if a.recompute and not a.table:
        p.error("--recompute needs --table")
    peaks = None
    if a.derived:
        try:
            peaks = load_measured_peaks(a.peaks)
        except MeasuredPeaksError as exc:
            print(f"DERIVED REFUSED: {exc}")
            return 1
    run, results = run_guards(a.root, a.manifest, a.profile_root, a.full)
    if a.recompute:
        checked, rv = recompute_table(
            run, Path(a.table).read_text(), violations_by_cell(results)
        )
        results.append(GuardResult("recompute", checked, rv))
    total = 0
    for r in results:
        total += len(r.violations)
        status = "PASS" if not r.violations else "FAIL"
        print(
            f"{r.name}: {status} ({r.checked} checked, {len(r.violations)} violations)"
        )
        for v in r.violations:
            print(f"  {v}")
    if a.derived:
        print(
            "DERIVED (not measured): MBU / MFU from measured inputs + model/hardware tables"
        )
        print(f"denominators: {json.dumps(peaks)}")
        for d in derived_rows(run, peaks):
            print(json.dumps(d, default=str))
    passed = sum(1 for r in results if not r.violations)
    print(f"TOTAL: {passed}/{len(results)} checks passed, {total} violations")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
