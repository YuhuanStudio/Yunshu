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

Exit status: 0 when all 12 guards pass (and the table recomputes, with --recompute),
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

GUARDS = (
    "prompt_identity",
    "provenance_identity",
    "pinned_versions",
    "manifest_completeness",
    "record_count",
    "engaged_mode",
    "weight_identity",
    "cold_cache",
    "gpu_exclusive_rotation",
    "output_sha",
    "measured_only",
    "fairness_controls",
    "telemetry",
)

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
        "cells": "[{engine, model, column, cell, group, ctx, kind, prompt_id, max_tokens, phases, expected_records, spec_mode, weight_group}]",
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
        "contended": "bool, machine contention flag",
        "device": "str, machine id",
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


@dataclass
class Run:
    manifest: dict
    files: list  # CellFile for every expected file that exists
    missing: list  # (cell, rep, path) for expected files that do not exist
    extra: list  # Path of jsonl files on disk that the manifest does not plan
    parse_errors: list  # str, one per unparseable line (truncated stream = error)


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


def load_run(root, manifest: dict) -> Run:
    root = Path(root)
    expected = {}
    for cell in manifest["cells"]:
        for rep in range(int(manifest["reps"])):
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
        files.append(CellFile(cell, rep, path, header, body))
    extra = [
        p
        for p in sorted(root.glob("*/*.jsonl"))
        if p not in expected and not p.name.endswith(TELEMETRY_SUFFIX)
    ]
    return Run(manifest, files, missing, extra, parse_errors)


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rotate(order, rep):
    k = rep % len(order)
    return list(order[k:]) + list(order[:k])


# ---------------------------------------------------------------------------
# The 12 hard guards. Each takes a Run and returns a GuardResult.
# ---------------------------------------------------------------------------
def check_prompt_identity(run: Run) -> GuardResult:
    """Guard 1: prompt token count + sha identical across engines in a comparison group.

    Compared on cold and warm rows only: turn2 prompts embed each engine's own reply, so
    they are recorded per engine and not required to match.
    """
    res = GuardResult("prompt_identity")
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
            if r.get("ctx") != cf.cell["ctx"] or r.get("kind") != cf.cell["kind"]:
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
        cell, eng = cf.cell["cell"], cf.cell["engine"]
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
            if (
                h.get("python") != yun.get("python")
                or not cmd
                or cmd[0] != yun.get("python")
            ):
                res.fail(cell, "Yunshu not started by its pinned venv python")
            if "-m" not in cmd or "yunshu_cli" not in cmd:
                res.fail(cell, "Yunshu not started as python -m yunshu_cli")
        elif not re.fullmatch(r"[0-9a-f]{64}", str(h.get("binary_sha256"))) or h.get(
            "binary_sha256"
        ) != pin.get("binary_sha256"):
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
    for cell, rep, path in run.missing:
        res.fail(cell["cell"], f"r{rep} missing ({path})")
    for path in run.extra:
        res.fail(path.stem, f"unplanned file {path} (not in manifest)")
    return res


def check_record_count(run: Run) -> GuardResult:
    """Guard 5: expected record count, every phase present, complete streams, output tokens."""
    res = GuardResult("record_count")
    for cf in run.files:
        cell = cf.cell["cell"]
        res.checked += 1
        if cf.header is None:
            res.fail(cell, f"r{cf.rep}: no session header row")
        exp = cf.cell["expected_records"]
        if len(cf.rows) != exp:
            res.fail(cell, f"r{cf.rep}: {len(cf.rows)} request rows, expected {exp}")
        if set(r.get("phase") for r in cf.rows) != set(cf.cell["phases"]):
            res.fail(
                cell,
                f"r{cf.rep}: phases {sorted(map(str, set(r.get('phase') for r in cf.rows)))} != planned {cf.cell['phases']}",
            )
        for r in cf.rows:
            if r.get("finish") not in ("stop", "length"):
                res.fail(
                    cell,
                    f"r{cf.rep} {r.get('phase')}: incomplete stream (finish={r.get('finish')!r})",
                )
            if r.get("ct") != cf.cell["max_tokens"]:
                res.fail(
                    cell,
                    f"r{cf.rep} {r.get('phase')}: output tokens {r.get('ct')} != expected {cf.cell['max_tokens']}",
                )
    return res


def check_engaged_mode(run: Run) -> GuardResult:
    """Guard 6: engaged spec mode parsed from the engine's own log equals the configured mode."""
    res = GuardResult("engaged_mode")
    for cf in run.files:
        h = cf.header
        if h is None:
            continue
        res.checked += 1
        want, cell = cf.cell["spec_mode"], cf.cell["cell"]
        for key in ("requested_spec_mode", "spec_mode_expected", "engaged_spec_mode"):
            if h.get(key) != want:
                res.fail(cell, f"r{cf.rep}: {key}={h.get(key)!r}, configured {want!r}")
        line = h.get("engaged_log_line")
        if not isinstance(line, str) or want.lower() not in line.lower():
            res.fail(cell, f"r{cf.rep}: engaged_log_line does not show mode {want!r}")
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
            elif phase in ("warm", "turn2"):
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
        observed, expected = [i[2] for i in items], rotate(order, rep)
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
    res.checked += 1
    if not (is_num(aa) and is_num(noise) and abs(aa) <= noise):
        res.fail("controls", f"A/A delta {aa!r} not inside noise {noise!r}")
    res.checked += 1
    if ctl.get("injected_delay_detected") is not True:
        res.fail("controls", "injected delay was not detected by the harness")
    homes = defaultdict(set)
    for cf in run.files:
        for r in cf.rows:
            res.checked += 1
            t = r.get("temperature")
            if not (is_num(t) and t == 0):
                res.fail(cf.cell["cell"], f"temperature {t!r}; greedy (0) required")
        if cf.header is not None:
            homes[cf.cell["engine"]].add(cf.header.get("home"))
    real_home = os.path.expanduser("~")
    unique = []
    for eng, hs in sorted(homes.items()):
        res.checked += 1
        if len(hs) != 1 or None in hs:
            res.fail(
                f"engine {eng}",
                f"HOME not one isolated directory: {sorted(map(str, hs))}",
            )
            continue
        (home,) = hs
        if home == real_home:
            res.fail(f"engine {eng}", "uses the real HOME, not an isolated one")
        unique.append(home)
    if len(set(unique)) != len(unique):
        res.fail("controls", "two engines share one HOME")
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


GUARD_CHECKS = (
    check_prompt_identity,
    check_provenance_identity,
    check_pinned_versions,
    check_manifest_completeness,
    check_record_count,
    check_engaged_mode,
    check_weight_identity,
    check_cold_cache,
    check_gpu_exclusive_rotation,
    check_output_sha,
    check_measured_only,
    check_fairness_controls,
    check_telemetry,
)


def run_guards(root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    run = load_run(root, manifest)
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
#   MBU  = (weight bytes read per token + KV bytes read per token + GDN state bytes) x tok/s / peak BW
#   MFU  = (2 x active params + attention FLOPs per token) x tok/s / peak FLOPs
# Inputs are measured (tok/s, prompt/completion tokens, ttft) or come from the tables below.
# A value whose input is not sourced is None with the reason, never a guess.
# ---------------------------------------------------------------------------
HARDWARE = {
    "M5 Max": {
        "peak_bw_gbs": 560.0,
        "bw_source": "IMPLIED: docs/research/ctx1m/ROOFLINE.md (BW = 560 GB/s, implied by the measured "
        "29 ms 27B floor). Not an Apple spec figure; replace when the spec value is verified.",
        "peak_tflops_bf16": None,
        "flops_source": "UNSOURCED: no peak FLOP/s figure in the repo or the spec. The ROOFLINE 40 TFLOP/s "
        "is measured effective prefill, not a peak. MFU is None until a sourced peak is entered.",
    },
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


def derived_decode(model_id: str, hw: str, tok_s: float, pt: int, ct: int) -> dict:
    """MBU and MFU for decode at the average context (pt + ct/2). Labeled DERIVED."""
    m, h = MODELS[model_id], HARDWARE[hw]
    ctx = pt + ct / 2.0
    out = {"label": "DERIVED", "ctx_avg": ctx, "mbu": None, "mfu": None, "reason": []}
    if m["weight_bytes"] is None:
        out["reason"].append("weight bytes per token not in repo")
    else:
        kv_gdn = kv_bytes_per_token(m, ctx) + (gdn_state_bytes(m) or 0)
        bytes_tok = m["weight_bytes"] + kv_gdn
        out["bytes_per_token"] = bytes_tok
        out["mbu"] = bytes_tok * tok_s / (h["peak_bw_gbs"] * 1e9)
    out.update(_mfu(m, h, tok_s, ctx, out))
    return out


def derived_prefill(model_id: str, hw: str, pt: int, ttft_s: float) -> dict:
    """MFU for prefill: prompt tokens over ttft (includes the first sampled token, so a slight
    under-estimate). Attention at the average prefill context pt/2. MBU is not defined for
    compute-bound prefill and is not reported."""
    m, h = MODELS[model_id], HARDWARE[hw]
    out = {"label": "DERIVED", "reason": []}
    if not ttft_s or ttft_s <= 0:
        return {**out, "mfu": None, "reason": ["ttft missing"]}
    out.update(_mfu(m, h, pt / ttft_s, pt / 2.0, out))
    out["prefill_tok_s"] = pt / ttft_s
    return out


def _mfu(m: dict, h: dict, tok_s: float, ctx: float, out: dict) -> dict:
    if m["active_params"] is None:
        out["reason"].append("active params not in repo")
        return {"mfu": None, "reason": out["reason"]}
    if h["peak_tflops_bf16"] is None:
        out["reason"].append("peak FLOP/s unsourced")
        return {"mfu": None, "reason": out["reason"]}
    flops_tok = 2 * m["active_params"] + attn_flops_per_token(m, ctx)
    return {
        "mfu": flops_tok * tok_s / (h["peak_tflops_bf16"] * 1e12),
        "reason": out["reason"],
    }


def derived_rows(run: Run, hw: str = "M5 Max") -> list:
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
                rows.append(
                    {
                        **base,
                        "metric": "decode",
                        **derived_decode(model_id, hw, r["dec_tps"], int(pt), int(ct)),
                    }
                )
            if is_num(r.get("ttft_s")):
                rows.append(
                    {
                        **base,
                        "metric": "prefill",
                        **derived_prefill(model_id, hw, int(pt), r["ttft_s"]),
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
    a = p.parse_args(argv)
    if a.recompute and not a.table:
        p.error("--recompute needs --table")
    run, results = run_guards(a.root, a.manifest)
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
        for d in derived_rows(run):
            print(json.dumps(d, default=str))
    passed = sum(1 for r in results if not r.violations)
    print(f"TOTAL: {passed}/{len(results)} checks passed, {total} violations")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
