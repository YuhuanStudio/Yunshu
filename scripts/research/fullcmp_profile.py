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
TREE_RE = re.compile(r"Tree verify stage profile \((\d+) rounds, ms/round\): (\{.*\})")
COMPETITOR_RE = re.compile(r"(?i)(stat|prof|timing|tok/s|\bms\b)")
# Structured layer-2 line: stage fields the profile lines do not carry, plus the host-sync counter
# and the op census (fullcmp_attribution.LAYER2_YUNSHU). The engine must print it for --full to pass.
LAYER2_RE = re.compile(r"^YUNSHU_LAYER2 (\{.*\})\s*$", re.M)


def profile_env(engine: str) -> list[str]:
    """Env assignments that switch the engine's own profiling on (none for competitors)."""
    if engine == "yunshu":
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


def _engine_log_path(cell: Path) -> str | None:
    for line in cell.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = (row.get("attr") or {}).get("engine_log_path")
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
    if not log_path:
        row["error"] = "no engine_log_path in cell file"
    else:
        try:
            text = Path(log_path).read_text(errors="replace")
        except OSError as exc:
            row["error"] = f"engine log unreadable: {exc}"
        else:
            row["engine_log_path"] = log_path
            if engine == "yunshu":
                stages = parse_yunshu_stages(text)
                l2 = parse_yunshu_layer2(text)
                for name, fields in (l2.get("stages") or {}).items():
                    if isinstance(fields, dict):
                        stages.setdefault(name, {}).update(fields)
                row["stages"] = stages
                for key in ("host_sync_count", "op_census"):
                    if l2.get(key) is not None:
                        row[key] = l2[key]
            else:
                lines = competitor_stat_lines(text)
                row["competitor_stats"] = lines or None
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
