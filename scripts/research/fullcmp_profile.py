#!/usr/bin/env python3
"""Layer-2 profile rows for --profile-rep (FULLCMP_SPEC "Attribution data", layer 2).

A profile job runs each cell once more with the engine's own profiling switch on. It never feeds
table numbers: its rows live under <raw root>/profile/<engine>/<cell>-prof.jsonl, which the audit
glob (<engine>/<cell>-rN.jsonl) does not read.

Yunshu: YUNSHU_DEBUG_TREE_PROFILE=1 (the stage timer in tree_verify.py: a sync after every stage)
writes "DFlash round profile", "DFlash host between rounds" and "Tree verify stage profile" lines to
the server log. They are parsed into ``stages``. Stages the log does not carry stay absent, so the
attribution check reports them MISSING.

Competitors: no profiling switch is known from the census, so no env is added. The profile row keeps
the lines of the engine's own log that look like timing or stats, as-is (``competitor_stats``).

Usage (called by the job chain after each profile cell):
    fullcmp_profile.py --cell <cell-prof.jsonl> --engine <engine>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROUND_RE = re.compile(
    r"DFlash round profile: (\d+) rounds, ([\d.]+) nodes/round, draft ([\d.]+) ms, "
    r"verify ([\d.]+) ms, commit\+host ([\d.]+) ms"
)
HOST_RE = re.compile(
    r"DFlash host between rounds: consumer \(yield\) ([\d.]+) ms/round"
)
# Phase B round driver (yunshu-pb2): no per-stage profile line; it logs the whole-round time over a window of
# rounds and, per request, the time inside decode steps and between steps.
PB2_WINDOW_RE = re.compile(
    r"Round driver DFlash window of (\d+) rounds \(total (\d+)\): rows/round ([\d.]+), ([\d.]+) ms/round, "
    r"nodes/row ([\d.]+), tokens/row-round ([\d.]+)"
)
PB2_IDLE_RE = re.compile(
    r"round driver idle: (\d+) decode steps, ([\d.]+) ms in a step, ([\d.]+) ms between steps"
)
TREE_RE = re.compile(r"Tree verify stage profile \((\d+) rounds, ms/round\): (\{.*\})")
COMPETITOR_RE = re.compile(r"(?i)(stat|prof|timing|tok/s|\bms\b)")
# Structured layer-2 line: stage fields the profile lines do not carry, plus the host-sync counter
# and the op census (fullcmp_attribution.LAYER2_YUNSHU). The engine must print it for --full to pass.
LAYER2_RE = re.compile(r"^YUNSHU_LAYER2 (\{.*\})\s*$", re.M)
# manifest ids of the Yunshu trees: dev0 and the Phase B preview (same server and profile lines)
YUNSHU_ENGINES = ("yunshu", "yunshu-pb2")


def profile_env(engine: str) -> list[str]:
    """Env assignments that switch the engine's own profiling on (none for competitors)."""
    if engine in YUNSHU_ENGINES:
        return ["YUNSHU_DEBUG_TREE_PROFILE=1"]
    return []


def parse_yunshu_stages(text: str) -> dict:
    """Stage dict from the Yunshu profile lines. Only what the log says; nothing is derived."""
    stages: dict = {}
    m = ROUND_RE.search(text)
    if m:
        rounds, nodes, draft, verify, commit = m.groups()
        stages["decode_round"] = {
            "rounds": int(rounds),
            "nodes_per_round": float(nodes),
            "drafter": float(draft),
            "verify_total": float(verify),
            "commit": float(commit),
        }
    h = HOST_RE.search(text)
    if h:
        stages.setdefault("decode_round", {})["host_yield"] = float(h.group(1))
    t = TREE_RE.search(text)
    if t:
        try:
            parts = json.loads(t.group(2).replace("'", '"'))
        except ValueError:
            parts = {}
        if parts:
            stages.setdefault("decode_round", {})["tree_stage_ms"] = {
                str(k): float(v) for k, v in parts.items()
            }
    return stages


def parse_pb2_stages(text: str) -> dict:
    """Stage dict from the Phase B round driver lines: decode steps and the step-weighted mean of the
    time inside a step and between steps (one 'round driver idle' line per request), plus the mean
    ms/round over the logged windows. Only what the log says (means of logged numbers, nothing modelled)."""
    idle = [(int(a), float(b), float(c)) for a, b, c in PB2_IDLE_RE.findall(text)]
    win = [float(m[3]) for m in PB2_WINDOW_RE.findall(text)]
    if not idle and not win:
        return {}
    node: dict = {}
    steps = sum(n for n, _, _ in idle)
    if steps:
        node["rounds"] = steps
        node["step_ms"] = sum(n * a for n, a, _ in idle) / steps
        node["host_between_steps_ms"] = sum(n * b for n, _, b in idle) / steps
    if win:
        node["window_ms_per_round"] = sum(win) / len(win)
    return {"decode_round": node}


def parse_yunshu_layer2(text: str) -> dict:
    """The YUNSHU_LAYER2 line (last one wins): {"stages", "host_sync_count", "op_census"} as
    printed, or {} when the engine did not print it. Malformed JSON is ignored (reported missing)."""
    found = LAYER2_RE.findall(text)
    if not found:
        return {}
    try:
        obj = json.loads(found[-1])
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def competitor_stat_lines(text: str) -> list[str]:
    return [
        ln.rstrip()
        for ln in text.splitlines()
        if COMPETITOR_RE.search(ln) and ln.strip()
    ]


# Why a layer-2 field is absent from a Yunshu profile row. The dev0 and Phase B preview trees print the
# "DFlash round profile", "host between rounds" and "Tree verify stage profile" lines only; the finer
# stage timers and the per-step counters are not emitted by those trees (no YUNSHU_LAYER2 line), so the row
# says so per field instead of leaving it absent. The per-step counters come from a Metal trace capture
# of the profile request, taken as a separate profile batch.
UNAVAILABLE_REASONS = {
    "stages.decode_round": (
        "unavailable: not emitted by this tree's profile log (dev0 prints drafter, verify_total, commit, "
        "host_yield and tree_stage_ms; the Phase B round driver prints step_ms and ms between steps; "
        "neither prints a YUNSHU_LAYER2 line)"
    ),
    "stages.prefill": "unavailable: prefill stage timers are not emitted by this tree's profile log",
    "stages.ttft": "unavailable: TTFT stage timers are not emitted by this tree's profile log",
    "host_sync_count": "unavailable: not an engine log counter on this tree; comes from the Metal trace capture",
    "op_census": "unavailable: not an engine log counter on this tree; comes from the Metal trace capture",
}


# Required layer-2 fields the "DFlash round profile" line does emit: absent means the log lacked the line.
EMITTED_BY_THE_ROUND_LINES = frozenset({"stages.decode_round.drafter", "stages.decode_round.commit"})


def unavailable_layer2(row: dict) -> dict:
    """{field: reason} for each required layer-2 field the row does not carry (explicit, never silent)."""
    import fullcmp_attribution as fa_attr

    out = {}
    stages = row.get("stages") or {}
    for field in fa_attr.required_layer2(row.get("engine", "yunshu")):
        if field.startswith("stages."):
            node, ok = stages, True
            for part in field.split(".", 1)[1].split("."):
                if not isinstance(node, dict) or part not in node:
                    ok = False
                    break
                node = node[part]
            present = ok and node is not None
            key = ".".join(field.split(".")[:2])
        else:
            present = row.get(field) is not None
            key = field
        # only what the tree cannot emit: the round line's own fields (drafter, commit) missing is a real gap
        # the dev0 round line carries drafter and commit: absent there is a real gap. The Phase B driver logs
        # whole-round and between-step times instead (step_ms in its stage dict), so those two are unavailable
        round_line_tree = "step_ms" not in (stages.get("decode_round") or {})
        if not present and not (round_line_tree and field in EMITTED_BY_THE_ROUND_LINES):
            out[field] = UNAVAILABLE_REASONS.get(key, "unavailable: not emitted by this engine")
    return out


def refresh_legacy_row(row: dict) -> dict:
    """A Yunshu profile row written before the Phase B parser and the per-field reasons existed: re-read its
    engine log (the path is in the row) and recompute stages and reasons. An unreadable log leaves the row
    unchanged apart from the reasons, so the guard still reports what the row lacks."""
    row = dict(row)
    try:
        text = Path(str(row.get("engine_log_path"))).read_text(errors="replace")
    except OSError:
        text = ""
    stages = parse_yunshu_stages(text) or parse_pb2_stages(text)
    if stages:
        row["stages"] = stages
    row["layer2_unavailable"] = unavailable_layer2(row)
    return row


def _engine_log_path(cell: Path) -> str | None:
    """The engine log of a cell: the session header's engine_log_path (every cell type carries it), else the
    attribution rows' or a decode row's (sequential cells). Conc and mixed cells have no decode rows."""
    rows = [json.loads(line) for line in cell.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("part") == "session" and row.get("engine_log_path"):
            return row["engine_log_path"]
    for row in rows:
        path = (row.get("attr") or {}).get("engine_log_path") or row.get("engine_log_path")
        if path:
            return path
    return None


def append_profile_row(cell: Path, engine: str) -> dict:
    """Read the engine log of this profile cell and append one part=profile row to the file.
    A cell without a log path, or an unreadable log, appends an error row (fail closed)."""
    row: dict = {
        "part": "profile",
        "engine": engine,
        "cell": cell.stem,
        "profile_env": profile_env(engine),
    }
    log_path = _engine_log_path(cell)
    concurrent = any(
        json.loads(line).get("part") in ("conc", "mixed") for line in cell.read_text().splitlines() if line.strip()
    )
    if not log_path and concurrent:
        # a concurrent cell with no recorded log: a labelled result, not an error (the stages need the log)
        row["stages_unavailable"] = "stages unavailable for concurrent path (no engine_log_path recorded)"
    elif not log_path:
        row["error"] = "no engine_log_path in cell file"
    else:
        try:
            text = Path(log_path).read_text(errors="replace")
        except OSError as exc:
            row["error"] = f"engine log unreadable: {exc}"
        else:
            row["engine_log_path"] = log_path
            if engine in YUNSHU_ENGINES:
                stages = parse_yunshu_stages(text) or parse_pb2_stages(text)
                l2 = parse_yunshu_layer2(text)
                for name, fields in (l2.get("stages") or {}).items():
                    if isinstance(fields, dict):
                        stages.setdefault(name, {}).update(fields)
                row["stages"] = stages
                for key in ("host_sync_count", "op_census"):
                    if l2.get(key) is not None:
                        row[key] = l2[key]
                row["layer2_unavailable"] = unavailable_layer2(row)
            else:
                lines = competitor_stat_lines(text)
                row["competitor_stats"] = lines or None
    if concurrent and log_path and "error" not in row:
        row["concurrent"] = True  # stages parsed from the engine log of the whole concurrent cell
    with cell.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cell", required=True, type=Path)
    ap.add_argument("--engine", required=True)
    a = ap.parse_args(argv)
    row = append_profile_row(a.cell, a.engine)
    print(json.dumps({k: row.get(k) for k in ("part", "cell", "error")}), flush=True)
    return 1 if row.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
