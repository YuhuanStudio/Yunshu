"""Engaged speculative mode from evidence, never from the request (fullcmp, fail closed).

A cell's engaged mode is derived only from evidence recorded per engine in an evidence table:

    {engine: {"mode": "dflash" | "mtp" | "off" | "ar",
              "evidence": [{"kind": "log_regex" | "metrics" | "response_field" | "acceptance_stat",
                            "pattern_or_path": str, "source_ref": str, "real_example": str,
                            "negative_example": str, "per_request": bool}, ...],
              "fallback": [regex, ...]        # optional: a log line that says the engine fell back to AR
              "log_source": str, "confidence": str, "notes": str}}

Evidence kinds:
- log_regex: the engine's own server log (the cell's .server.log slice) matches the pattern.
- metrics: the pattern matches the engine's /metrics or status probe text recorded for the cell.
- response_field: a dotted path into the per-request records (x_yunshu.speculative fields as the
  harness flattens them) exists with a truthy value.
- acceptance_stat: a dotted path into the per-request records holds a number above zero.

Result: {"engaged_mode": <mode> | "ar" | "unverified", "matched_line", "evidence_kind", "stats"}.
A fallback line in the log wins over positive evidence ("ar"). No evidence is "unverified", and
the audit fails on it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_TABLE = Path(__file__).with_name("fullcmp_engaged_evidence.json")
MODES = frozenset({"dflash", "mtp", "off", "ar"})
AR_MODES = frozenset({"ar", "off"})
KINDS = frozenset({"log_regex", "metrics", "response_field", "acceptance_stat"})
EVIDENCE_KEYS = ("kind", "pattern_or_path", "source_ref", "real_example", "negative_example", "per_request")
UNVERIFIED = "unverified"


class EvidenceError(ValueError):
    """The evidence table is malformed. Callers treat it as no evidence (fail closed)."""


def validate_table(table) -> list[str]:
    """Every schema problem in an evidence table (empty list when valid)."""
    if not isinstance(table, dict):
        return ["table is not a JSON object"]
    problems = []
    for engine, entry in table.items():
        where = f"{engine}"
        if not isinstance(entry, dict):
            problems.append(f"{where}: entry is not an object")
            continue
        if entry.get("mode") not in MODES:
            problems.append(f"{where}: mode {entry.get('mode')!r} not in {sorted(MODES)}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            problems.append(f"{where}: evidence must be a non-empty list")
            evidence = []
        for i, ev in enumerate(evidence):
            if not isinstance(ev, dict):
                problems.append(f"{where}: evidence[{i}] is not an object")
                continue
            missing = [k for k in EVIDENCE_KEYS if k not in ev]
            if missing:
                problems.append(f"{where}: evidence[{i}] missing {missing}")
            if ev.get("kind") not in KINDS:
                problems.append(f"{where}: evidence[{i}] kind {ev.get('kind')!r} not in {sorted(KINDS)}")
            elif ev["kind"] in ("log_regex", "metrics"):
                try:
                    re.compile(str(ev.get("pattern_or_path", "")))
                except re.error as exc:
                    problems.append(f"{where}: evidence[{i}] bad regex: {exc}")
            elif not str(ev.get("pattern_or_path", "")).strip():
                problems.append(f"{where}: evidence[{i}] empty path")
            if not str(ev.get("source_ref", "")).strip():
                problems.append(f"{where}: evidence[{i}] has no source_ref")
        for pat in entry.get("fallback", []) or []:
            try:
                re.compile(str(pat))
            except re.error as exc:
                problems.append(f"{where}: fallback bad regex: {exc}")
    return problems


def load_table(path=DEFAULT_TABLE):
    """The validated evidence table, or None when the file is missing or malformed."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        table = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return None if validate_table(table) else table


def _line_at(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return text[line_start : line_end if line_end != -1 else len(text)]


def _dotted(row, path: str):
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def acceptance_paths(entry) -> list[str]:
    paths = [ev["pattern_or_path"] for ev in entry.get("evidence", []) if ev.get("kind") == "acceptance_stat"]
    return paths or ["attr.accepted"]


def evaluate(engine, log: str, rows, table, probe_text: str | None = None) -> dict:
    """Engaged mode of one cell from its engine log slice, its request rows and its status probe."""
    log = log or ""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    stats = {"requests": len(rows)}
    unverified = {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": None, "stats": stats}
    entry = table.get(engine) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        stats["table"] = "no entry for engine"
        return unverified
    for pat in entry.get("fallback", []) or []:
        m = re.search(pat, log)
        if m:
            return {
                "engaged_mode": "ar",
                "matched_line": _line_at(log, m.start(), m.end()),
                "evidence_kind": "fallback",
                "stats": stats,
            }
    for ev in entry.get("evidence", []):
        kind, pat = ev.get("kind"), str(ev.get("pattern_or_path", ""))
        hit = None
        if kind == "log_regex":
            m = re.search(pat, log)
            if m:
                hit = _line_at(log, m.start(), m.end())
        elif kind == "metrics":
            m = re.search(pat, probe_text or "")
            if m:
                hit = _line_at(probe_text or "", m.start(), m.end())
        elif kind == "response_field":
            for row in rows:
                value = _dotted(row, pat)
                if value not in (None, False, 0, "", [], {}):
                    hit = f"{pat}={value!r}"
                    break
        elif kind == "acceptance_stat":
            positives = sum(1 for row in rows if _is_positive(_dotted(row, pat)))
            stats["acceptance_positive_rows"] = positives
            if positives:
                hit = f"{pat} > 0 in {positives} request(s)"
        if hit is not None:
            return {
                "engaged_mode": entry["mode"],
                "matched_line": hit,
                "evidence_kind": kind,
                "stats": stats,
            }
    return unverified


def acceptance_problem(engaged_mode, entry, rows) -> str | None:
    """For a speculative mode: why its per-request acceptance stats do not prove drafting, or None."""
    if engaged_mode in AR_MODES or engaged_mode == UNVERIFIED:
        return None
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return "speculative mode but no per-request records to check acceptance"
    paths = acceptance_paths(entry or {})
    if any(_is_positive(_dotted(row, p)) for row in rows for p in paths):
        return None
    return f"speculative mode {engaged_mode!r} but every request's acceptance stat ({', '.join(paths)}) is zero or absent"


YUNSHU_RUNNER = re.compile(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b")


def yunshu_runner_evidence(log: str) -> dict:
    """Yunshu's own evidence: the initialized VLM batch runner line (the last one wins)."""
    log = log or ""
    found = list(YUNSHU_RUNNER.finditer(log))
    if not found:
        return {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": None, "stats": {}}
    m = found[-1]
    return {
        "engaged_mode": m.group(1),
        "matched_line": _line_at(log, m.start(), m.end()),
        "evidence_kind": "log_regex",
        "stats": {"runner_lines": len(found)},
    }
