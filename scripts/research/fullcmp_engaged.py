"""Engaged speculative mode from evidence, never from the request (fullcmp, fail closed).

Table (scripts/research/fullcmp_engaged_evidence.json, built by fullcmp_engaged_merge.py):

    {"_meta": {...},
     "engines": {engine: {"modes": {mode: {"evidence": [...], "fallback": [regex, ...]}},
                          "log_source": str, "notes": str, "confidence": str}}}

The cell's EXPECTED mode selects the mode entry; an engine with no configured mode (empty
"modes") can never be verified. Evidence item: {kind, pattern_or_path, source_ref, real_example,
negative_example, per_request, startup_sufficient}. Kinds:
- log_regex: the engine log matches. The log is split at the ready offset (the byte offset after
  readiness): a match before it is source "startup_log", after it "slice".
- metrics: the pattern matches the engine's status/metrics probe text (source "metrics").
- response_field: a dotted path into the per-request records holds a truthy value.
- acceptance_stat: a dotted path into the per-request records holds a number above zero.

Rules: a fallback line anywhere in the log -> "ar". Engagement needs per-request evidence when
the mode has any per_request entry (a per_request log line after readiness, a response field, an
acceptance stat or a metric); otherwise any non-startup hit counts. A startup-only line proves
"loaded", not "engaged", unless that entry sets startup_sufficient. Loaded-only is "unverified"
with evidence_kind "startup_only". No hit is "unverified".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_TABLE = Path(__file__).with_name("fullcmp_engaged_evidence.json")
MODES = frozenset({"dflash", "mtp", "off", "ar"})
AR_MODES = frozenset({"ar", "off"})
KINDS = frozenset({"log_regex", "metrics", "response_field", "acceptance_stat", "acceptance_regex",
                    "response_field_probe"})
REQUIRED = ("kind", "pattern_or_path", "source_ref", "real_example", "negative_example", "per_request")
UNVERIFIED = "unverified"
# tfbench main() sends this many readiness requests (short "Say hi") before the measured parts; their
# done lines are the first lines after the ready offset and never evidence for a measured request.
READINESS_REQUESTS = 2
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


class EvidenceError(ValueError):
    """The evidence table is malformed. Callers treat it as no evidence (fail closed)."""


def validate_table(table) -> list[str]:
    """Every schema problem in an evidence table (empty list when valid)."""
    if not isinstance(table, dict) or not isinstance(table.get("engines"), dict):
        return ["table must be an object with an 'engines' object"]
    problems = []
    for engine, entry in table["engines"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("modes"), dict):
            problems.append(f"{engine}: entry needs a 'modes' object")
            continue
        for mode, m in entry["modes"].items():
            where = f"{engine}/{mode}"
            if mode not in MODES:
                problems.append(f"{where}: mode not in {sorted(MODES)}")
            if not isinstance(m, dict) or not isinstance(m.get("evidence"), list) or not m["evidence"]:
                problems.append(f"{where}: evidence must be a non-empty list")
                continue
            for i, ev in enumerate(m["evidence"]):
                if not isinstance(ev, dict):
                    problems.append(f"{where}: evidence[{i}] is not an object")
                    continue
                missing = [k for k in REQUIRED if k not in ev]
                if missing:
                    problems.append(f"{where}: evidence[{i}] missing {missing}")
                    continue
                if ev["kind"] not in KINDS:
                    problems.append(f"{where}: evidence[{i}] kind {ev['kind']!r} not in {sorted(KINDS)}")
                elif ev["kind"] in ("log_regex", "metrics", "acceptance_regex"):
                    try:
                        rx = re.compile(str(ev["pattern_or_path"]))
                    except re.error as exc:
                        problems.append(f"{where}: evidence[{i}] bad regex: {exc}")
                        rx = None
                    if ev["kind"] == "acceptance_regex":
                        # a numeric capture group; the hit needs value > min
                        if rx is None or ev.get("group") not in rx.groupindex or not isinstance(
                            ev.get("min"), (int, float)
                        ):
                            problems.append(f"{where}: evidence[{i}] acceptance_regex needs group (a named group) and numeric min")
                elif ev["kind"] == "response_field_probe":
                    paths = ev.get("paths")
                    if not isinstance(paths, list) or not paths or not all(
                        isinstance(p, str) and _DOTTED.match(p) for p in paths
                    ):
                        problems.append(f"{where}: evidence[{i}] response_field_probe needs dotted 'paths'")
                elif not _DOTTED.match(str(ev["pattern_or_path"])):
                    problems.append(f"{where}: evidence[{i}] path {ev['pattern_or_path']!r} is not a dotted field path")
                if not str(ev.get("source_ref", "")).strip():
                    problems.append(f"{where}: evidence[{i}] has no source_ref")
            for pat in m.get("fallback", []) or []:
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


def mode_entry(table, engine, mode):
    """The mode entry the expected mode selects, or None (no entry: unverifiable)."""
    if not isinstance(table, dict):
        return None
    eng = (table.get("engines") or {}).get(engine)
    if not isinstance(eng, dict):
        return None
    m = (eng.get("modes") or {}).get(mode)
    return m if isinstance(m, dict) else None


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


def acceptance_paths(m) -> list[str]:
    paths = [ev["pattern_or_path"] for ev in (m or {}).get("evidence", []) if ev.get("kind") == "acceptance_stat"]
    return paths or ["attr.accepted"]


def evaluate(engine, mode, log, rows, table=None, probe_text=None, ready_offset=None, probe_response=None) -> dict:
    if table is None:  # no table passed: the default file (None when missing: unverified)
        table = load_table()
    """Engaged mode of one cell for its expected mode, from the engine log (whole server log),
    the request rows and the status probe. ready_offset: the log byte offset after readiness
    (None: the whole log counts as request-time evidence)."""
    log = log or ""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    stats = {"requests": len(rows)}

    def result(mode_out, line, kind, source):
        eng = (table or {}).get("engines", {}).get(engine) or {}
        return {"engaged_mode": mode_out, "matched_line": line, "evidence_kind": kind,
                "source": source, "stats": stats, "label": eng.get("engaged_label")}

    if mode in AR_MODES:
        drafting = ar_drafting_line(log, ready_offset)
        if drafting is not None:
            return result(UNVERIFIED, drafting, "ar_drafting", "slice")
    m = mode_entry(table, engine, mode)
    if m is None and mode in AR_MODES:
        return result("ar", None, "ar_no_drafting", "slice")
    if m is None:
        stats["table"] = f"no entry for {engine}/{mode}"
        return result(UNVERIFIED, None, None, None)
    for pat in m.get("fallback", []) or []:
        hit = re.search(pat, log)
        if hit:
            return result("ar", _line_at(log, hit.start(), hit.end()), "fallback", "slice")
    per_request_kinds = any(ev.get("per_request") for ev in m["evidence"])
    good, loaded = [], []
    for ev in m["evidence"]:
        kind, pat = ev["kind"], str(ev["pattern_or_path"])
        hits = []  # (line, source)
        if kind == "log_regex":
            for hit in re.finditer(pat, log):
                src = "slice"
                if ready_offset is not None and hit.start() < ready_offset:
                    src = "startup_log"
                hits.append((_line_at(log, hit.start(), hit.end()), src))
                if src == "slice":
                    break
        elif kind == "metrics":
            hit = re.search(pat, probe_text or "")
            if hit:
                hits.append((_line_at(probe_text or "", hit.start(), hit.end()), "metrics"))
        elif kind == "response_field":
            for row in rows:
                value = _dotted(row, pat)
                if value not in (None, False, 0, "", [], {}):
                    hits.append((f"{pat}={value!r}", "response_field"))
                    break
        elif kind == "acceptance_stat":
            positives = sum(1 for row in rows if _is_positive(_dotted(row, pat)))
            stats["acceptance_positive_rows"] = positives
            if positives:
                hits.append((f"{pat} > 0 in {positives} request(s)", "acceptance_stat"))
        elif kind == "response_field_probe":
            # one non-streaming probe request after the cell (tfbench): the serving path returns the
            # per-request draft stats only in a non-streaming response (mlx-vlm 0.7.11, openai.py:2759)
            values = [_dotted(probe_response or {}, p) for p in ev.get("paths", [])]
            if values and all(_is_positive(v) for v in values):
                hits.append((f"probe {ev['paths']}={values}", "probe"))
            stats["probe"] = values
        elif kind == "acceptance_regex":
            # Per MEASURED request: every row of the cell needs its own done line, after the readiness
            # requests, with the same prompt count (named group 'prompt' when the line has one,
            # otherwise the k-th row takes the k-th line) and a value above min. A readiness ping
            # that drafted, or a missing done line, never counts for a measured row.
            per_row, why = _correlate(ev, log, rows, ready_offset)
            if why is None and rows:
                # this item proves drafting per measured row: its matches are the cell's per-row evidence
                # (an earlier item's failure to match is not a problem of the cell)
                stats["per_row_matched"] = len(per_row)
                stats["per_row"] = per_row
                stats.pop("per_row_problem", None)
                hits.append((per_row[-1]["line"], "slice"))
            elif why is not None and "per_row" not in stats:
                stats["per_row_problem"] = why
        for line, src in hits:
            if src == "startup_log":
                if ev.get("startup_sufficient"):
                    good.append((line, kind, src))
                else:
                    loaded.append((line, kind, src))
            elif per_request_kinds and not ev.get("per_request"):
                loaded.append((line, kind, src))
            else:
                good.append((line, kind, src))
    if good:
        line, kind, src = good[0]
        return result(mode, line, kind, src)
    if loaded:
        line, kind, src = loaded[0]
        return result(UNVERIFIED, line, "startup_only", src)
    return result(UNVERIFIED, None, None, None)


def _correlate(ev, log, rows, ready_offset):
    """(per-row matches, None) when every measured row has its own positive done line, else
    ([], reason)."""
    rx, group, minimum = re.compile(str(ev["pattern_or_path"])), ev.get("group"), float(ev.get("min"))
    start = ready_offset or 0
    cands = []
    for hit in rx.finditer(log, start):
        try:
            value = float(hit.group(group))
        except (TypeError, ValueError):
            continue
        prompt = None
        if "prompt" in rx.groupindex and hit.group("prompt") is not None:
            prompt = int(hit.group("prompt").replace(",", ""))  # splash prints 1,024
        cands.append({"line": _line_at(log, hit.start(), hit.end()), "value": value, "prompt": prompt})
    cands = cands[READINESS_REQUESTS:]  # the readiness pings take the first done lines
    measured = [r for r in rows if isinstance(r.get("pt"), int)]
    if not measured or len(measured) != len(rows):
        return [], "measured rows without a prompt count"
    used, per_row = set(), []
    for k, row in enumerate(measured):
        found = None
        for j, c in enumerate(cands):
            if j in used:
                continue
            if c["prompt"] is not None and c["prompt"] != row["pt"]:
                continue
            if c["prompt"] is None and j != _next_free(used):
                continue
            found = j
            break
        if found is None:
            return per_row, f"row {k} (prompt {row['pt']}): no done line"
        used.add(found)
        c = cands[found]
        if not c["value"] > minimum:
            return per_row, f"row {k} (prompt {row['pt']}): done line value {c['value']} not above {minimum}"
        per_row.append({"row": k, "prompt": row["pt"], "line": c["line"], "value": c["value"],
                        "group": ev.get("group")})
    return per_row, None


def _next_free(used):
    n = 0
    while n in used:
        n += 1
    return n


_DRAFTED = re.compile(r"accepted=\d+/(\d+)")
_ROUNDS = re.compile(r"rounds=(\d+)|(\d+) cycles|cycles=(\d+)")
COUNT_GROUPS = ("accepted", "accepts", "acc")  # a numeric group that is an accepted-token count


def attr_rows_from(per_row, rows) -> list[dict]:
    """One attribution row per measured request, from the matched evidence line (single source of the
    per-request draft fields). A field the line does not state is 'unavailable (reason)', never 0 or
    an estimate. tokens_per_round = emitted tokens (the row's ct) / rounds."""
    out = []
    for pr in per_row:
        k = pr["row"]
        line = pr.get("line") or ""
        value = float(pr["value"])
        if pr.get("group") in COUNT_GROUPS:
            accepted = int(value) if value.is_integer() else value
        else:
            accepted = "unavailable (line states a tokens-per-cycle value, not an accepted count)"
        m = _DRAFTED.search(line)
        drafted = int(m.group(1)) if m else "unavailable (line states no drafted count)"
        m = _ROUNDS.search(line)
        rounds = int(next(g for g in m.groups() if g)) if m else "unavailable (line states no round count)"
        ct = rows[k].get("ct") if k < len(rows) else None
        if isinstance(rounds, int) and rounds > 0 and isinstance(ct, int):
            tpr = round(ct / rounds, 3)
        else:
            tpr = "unavailable (no rounds or no emitted-token count)"
        out.append({"row": k, "accepted": accepted, "drafted": drafted, "rounds": rounds,
                    "tokens_per_round": tpr, "matched_line": line})
    return out


HANDOFF_FINISH = "late-join-handoff"
CALIBRATION_FINISH = "batch-standard-calibration"  # oMLX decoding without drafting under concurrency
NON_TERMINAL_FINISH = frozenset({HANDOFF_FINISH, CALIBRATION_FINISH})
TERMINAL_FINISH = frozenset({"length", "stop"})


def _merge_segments(raw):
    """oMLX prints a running request's stats in segments: 'late-join-handoff' (another request joined) and
    'batch-standard-calibration' (decoding without drafting), interleaved across uids. Every segment of a uid
    merges into one request, up to its terminal finish (length or stop), whatever the interleaving; tokens,
    accepted and drafted are summed; the calibration tokens are kept (calibration_tokens). A uid with no
    terminal segment is unmatched, and an unknown finish kind fails closed; both carry 'bad' with the line."""
    out, pending = [], {}
    for seg in raw:
        uid, fin = seg.get("uid"), seg.get("finish")
        if uid is None:
            out.append(seg)
            continue
        if fin in NON_TERMINAL_FINISH:
            pending.setdefault(uid, []).append(seg)
            continue
        if fin in TERMINAL_FINISH:
            parts = pending.pop(uid, [])
            segs = parts + [seg]
            merged = dict(seg)
            merged["ct"] = sum(p.get("ct") or 0 for p in segs)
            merged["value"] = sum(p.get("value") or 0 for p in segs if p.get("finish") != CALIBRATION_FINISH)
            merged["calibration_tokens"] = sum(p.get("ct") or 0 for p in parts if p.get("finish") == CALIBRATION_FINISH)
            merged["segments"] = len(segs)
            out.append(merged)
            continue
        out.append(dict(seg, bad=f"unknown oMLX finish kind {fin!r} for uid {uid}"))
    for uid, parts in pending.items():
        out.append(dict(parts[-1], bad=f"uid {uid}: {len(parts)} non-terminal segment(s) and no terminal finish"))
    return out


def _done_candidates(item, log, ready_offset, skip=READINESS_REQUESTS):
    """Every match of the item's regex after the readiness requests, in log order (oMLX's split segments
    are merged per uid first)."""
    rx = re.compile(str(item["pattern_or_path"]))
    group = item.get("group")
    names = rx.groupindex
    raw = []
    for hit in rx.finditer(log or "", ready_offset or 0):
        try:
            value = float(hit.group(group))
        except (TypeError, ValueError, IndexError):
            continue
        prompt = int(hit.group("prompt").replace(",", "")) if "prompt" in names and hit.group("prompt") else None
        ct = None
        for key in ("ct", "tokens"):
            if key in names and hit.group(key):
                ct = int(hit.group(key).replace(",", ""))
                break
        raw.append({"line": _line_at(log, hit.start(), hit.end()), "value": value, "prompt": prompt, "ct": ct,
                    "uid": hit.group("uid") if "uid" in names else None,
                    "finish": hit.group("finish") if "finish" in names else None})
    if "uid" in names and "finish" in names:
        raw = _merge_segments(raw)
    cands = [{"line": c["line"], "value": c["value"], "prompt": c["prompt"], "ct": c["ct"],
              "bad": c.get("bad"), "calibration_tokens": c.get("calibration_tokens", 0),
              "calibration_share": (c.get("calibration_tokens", 0) / c["ct"]) if c.get("ct") else 0.0}
             for c in raw]
    if item.get("request_line"):
        return _request_line_candidates(item, log, ready_offset, skip, cands)
    return cands[skip:]


def _request_line_candidates(item, log, ready_offset, skip, mtp):
    """oMLX: one candidate per request, from the server's own 'Chat completion' line (prompt length, completion
    tokens, finish). The MTP uid groups (merged, in terminal order) supply the acceptance and calibration of the
    same request: paired with the chat lines by order (both are printed at completion). A count mismatch between
    the two makes every candidate bad (fail closed)."""
    rx = re.compile(str(item["request_line"]))
    chats = []
    for hit in rx.finditer(log or "", ready_offset or 0):
        chats.append({"line": _line_at(log, hit.start(), hit.end()),
                      "prompt": int(hit.group("prompt").replace(",", "")),
                      "ct": int(hit.group("tokens").replace(",", ""))})
    mtp_ok = [m for m in mtp if not m.get("bad")]
    if len(chats) != len(mtp):
        bad = f"{len(chats)} Chat completion lines and {len(mtp)} MTP request groups"
        return [{"line": chats[i]["line"] if i < len(chats) else "", "value": 0.0, "prompt": None, "ct": None,
                 "bad": bad, "calibration_tokens": 0, "calibration_share": 0.0} for i in range(max(1, len(chats)))][skip:]
    out = []
    for chat, m in zip(chats, mtp):
        out.append({"line": chat["line"], "value": m["value"], "prompt": chat["prompt"], "ct": chat["ct"],
                    "ct_merged": m.get("ct"), "bad": m.get("bad"),
                    "calibration_tokens": m.get("calibration_tokens", 0),
                    "calibration_share": (m.get("calibration_tokens", 0) / m["ct"]) if m.get("ct") else 0.0})
    return out[skip:]


def correlate_multiset(item, log, requests, ready_offset, consumed=0, skip=READINESS_REQUESTS):
    """Concurrent requests (multiset, not order) of ONE record: its requests take the next len(requests)
    post-readiness done lines after `consumed` (records of a cell run one after another), matched by
    (prompt count, token count) when the item has those groups; log order breaks ties. Fail closed on a
    shortage of lines, an unmatched request, or a value not above min. Returns (matches, None) or
    ([], reason). The caller checks the cell's total (concurrent_check)."""
    group, minimum = item.get("group"), float(item.get("min", 0))
    cands = _done_candidates(item, log, ready_offset, skip)[consumed:consumed + len(requests)]
    for c in cands:
        if c.get("bad"):
            return [], f"{c['bad']}: {c['line'][:160]}"
    # merged token sums are not the request's completion count when calibration segments are present (the
    # wire-1106d rc1 mixed log: 8 requests of 64 tokens merge to 29..64): such items match by order and report
    # the gap per request (ct_gap); a tolerance check applies only where item match_tokens is true
    use_ct = bool(item.get("match_tokens", True))
    tol = int(item.get("token_tolerance", 0))  # oMLX's merged token sums miss the first tokens of a request
    if len(cands) != len(requests):
        return [], f"count mismatch: {len(cands)} done lines for {len(requests)} requests"
    used, out = set(), []
    for rid, req in enumerate(requests):
        found = None
        for j, c in enumerate(cands):
            if j in used:
                continue
            if c["prompt"] is not None and c["prompt"] != req.get("pt"):
                continue
            if use_ct and c["ct"] is not None and req.get("ct") is not None and not (
                req["ct"] - tol <= c["ct"] <= req["ct"]
            ):
                continue
            found = j
            break
        if found is None:
            return [], f"request {rid} (prompt {req.get('pt')}, tokens {req.get('ct')}): no done line"
        used.add(found)
        c = cands[found]
        calibrated = c.get("calibration_share", 0.0) >= 1.0
        if not calibrated and not c["value"] > minimum:
            return [], f"request {rid}: value {c['value']} not above {minimum}"
        out.append({"rid": rid, "prompt": c["prompt"], "line": c["line"],
                    "value": 0.0 if calibrated else c["value"], "group": group,
                    "ct_expected": req.get("ct"), "ct_merged": c.get("ct_merged", c.get("ct")),
                    "ct_gap": (req.get("ct") - c.get("ct_merged", c.get("ct")))
                    if (req.get("ct") is not None and c.get("ct_merged", c.get("ct")) is not None) else None,
                    "calibration_share": round(c.get("calibration_share", 0.0), 4),
                    "calibrated": calibrated,
                    # partial log coverage: the merged MTP token sum misses more than the documented tolerance
                    "partial_coverage": (req.get("ct") is not None and c.get("ct_merged", c.get("ct")) is not None and
                                         (req["ct"] - c.get("ct_merged", c.get("ct"))) > int(item.get("token_tolerance", 0)))})
    return out, None


QUEUED = re.compile(r"queued=([0-9.]+)")


def queued_s(line: str):
    """TensorFold's queue wait for a request (serialized concurrency), or None when the line has none."""
    m = QUEUED.search(line or "")
    return float(m.group(1)) if m else None


def read_concurrent_records(path) -> list:
    """The conc / mixed rows of a cell file (malformed lines are skipped here; the parse guards report them)."""
    out = []
    p = Path(path)
    for line in (p.read_text(errors="replace").splitlines() if p.is_file() else []):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("part") in ("conc", "mixed"):
            out.append(rec)
    return out


# Yunshu dev0 concurrent requests run on the batch path: no DFlash per request (wire-1106c). A cell whose
# requests carry no speculative counters is derived from its DFlash round lines in the log slice.
DFLASH_BLOCK_MAX = 8  # the DFlash2 drafter block (wire logs: block=8): a round emits at most this many tokens
DFLASH_ROUND = re.compile(r"DFlash round profile: (\d+) rounds")
CONC_AR_LABEL = "ar (dev0: concurrent requests do not draft)"


# lines a speculative engine prints per request; an engine whose mode is AR must print none of them
AR_DRAFT_MARKERS = re.compile(
    r"accepted=\d+|accept=\d+/|draft_n|accepted/drafted|MTP\[\S+\] finish=|DFlash round profile|"
    r"draft acceptance =|\[spec-stats\]|vlm_mtp stats|spec: \S+ accepted/drafted"
)


def ar_drafting_line(log: str, ready_offset=None):
    """The first per-request drafting line after readiness in an AR engine's log, or None."""
    m = AR_DRAFT_MARKERS.search(log or "", ready_offset or 0)
    return _line_at(log, m.start(), m.end()) if m else None


def _ar_concurrent(records, log, ready_offset):
    """AR mode (mlx-lm and other engines with no speculative path): nothing to accept. The cell's requests
    must complete (a request list per record) and no drafting line may appear after readiness."""
    problems = []
    total = 0
    for rec in records:
        reqs = rec.get("requests")
        if not isinstance(reqs, list) or not reqs:
            problems.append(f"{rec.get('part')} n={rec.get('n')}: no requests list")
            continue
        total += len(reqs)
    drafting = ar_drafting_line(log, ready_offset)
    if drafting is not None:
        problems.append(f"ar engine shows drafting: {drafting[:120]}")
    stats = {"requests": total, "per_row": []}
    if problems:
        return {"engaged_mode": UNVERIFIED, "matched_line": drafting, "evidence_kind": "ar_drafting",
                "source": "slice", "stats": stats, "label": None}, problems
    return {"engaged_mode": "ar", "matched_line": None, "evidence_kind": "ar_no_drafting",
            "source": "slice", "stats": stats, "label": None}, []


def _yunshu_concurrent(records, mode, log="", ready_offset=None):
    """Yunshu conc/mixed cells. With per-request counters (x_yunshu.speculative): each request must show
    accepted > 0. Without them: the cell's DFlash rounds in the log slice decide. rounds * DFLASH_BLOCK_MAX
    < the cell's completion tokens -> the concurrent path does not draft: engaged 'ar' (a label, not a
    violation). Rounds that could cover the tokens -> the mode, from the rounds (no per-request proof)."""
    reqs_all = [q for rec in records for q in (rec.get("requests") or [])]
    if reqs_all and not all(isinstance(q.get("spec"), dict) for q in reqs_all):
        tokens = sum(int(q.get("ct") or 0) for q in reqs_all)
        window = (log or "")[ready_offset or 0:]
        rounds = sum(int(m.group(1)) for m in DFLASH_ROUND.finditer(window))
        stats = {"rounds": rounds, "tokens": tokens, "requests": len(reqs_all)}
        if rounds * DFLASH_BLOCK_MAX < tokens:
            return {"engaged_mode": "ar", "matched_line": f"DFlash rounds {rounds} x block {DFLASH_BLOCK_MAX} < "
                    f"{tokens} tokens", "evidence_kind": "concurrent_rounds", "source": "slice",
                    "stats": stats, "label": CONC_AR_LABEL}, []
        return {"engaged_mode": mode, "matched_line": f"DFlash rounds {rounds} cover {tokens} tokens",
                "evidence_kind": "concurrent_rounds", "source": "slice", "stats": stats, "label": None}, []
    problems, matched = [], []
    for rec in records:
        reqs = rec.get("requests")
        if not isinstance(reqs, list) or not reqs:
            problems.append(f"{rec.get('part')} n={rec.get('n')}: no requests list")
            continue
        for rid, req in enumerate(reqs):
            spec = req.get("spec")
            if not isinstance(spec, dict) or not _is_positive(spec.get("accepted")):
                problems.append(f"{rec.get('part')} n={rec.get('n')} request {rid}: no accepted count > 0 ({spec!r})")
                continue
            matched.append({"rid": rid, "part": rec.get("part"), "prompt": req.get("pt"),
                            "line": f"x_yunshu.speculative accepted={spec['accepted']} drafted={spec.get('drafted')}",
                            "value": float(spec["accepted"]), "group": "accepted", "queued_s": None})
    if problems or not matched:
        return {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": "concurrent",
                "source": "record", "stats": {"per_row_problem": "; ".join(problems) or "no requests",
                                              "per_row": matched}, "label": None}, problems
    per_row = [{"row": i, "prompt": x["prompt"], "line": x["line"], "value": x["value"], "group": "accepted",
                "queued_s": None, "rid": x["rid"], "part": x["part"]} for i, x in enumerate(matched)]
    return {"engaged_mode": mode, "matched_line": matched[-1]["line"], "evidence_kind": "record",
            "source": "record", "stats": {"per_row": per_row, "concurrent": len(matched)},
            "label": None}, []


def concurrent_attr_rows(table, engine, mode, log, records, ready_offset, probe_responses=None) -> list:
    """part=attr rows for the concurrent requests (rid, record part), from the same verdict as the audit."""
    ev, _problems = concurrent_check(table, engine, mode, log, records, ready_offset, probe_responses)
    out = []
    for pr in (ev or {}).get("stats", {}).get("per_row", []):
        out.append({
            "rec_part": pr.get("part"),
            "rid": pr.get("rid"),
            "accepted": int(pr["value"]) if pr.get("group") in COUNT_GROUPS else
            "unavailable (line states a tokens-per-cycle value, not an accepted count)",
            "queued_s": pr.get("queued_s"),
            "ct_expected": pr.get("ct_expected"),
            "ct_merged": pr.get("ct_merged"),
            "ct_gap": pr.get("ct_gap"),
            "partial_coverage": pr.get("partial_coverage", False),
            "calibration_share": pr.get("calibration_share"),
            "matched_line": pr.get("line"),
        })
    return out


CONC_PROBE_AR_LABEL = "ar (concurrent path does not draft)"
PROBE_ONLY_ENGINES = {"mlx-vlm-server"}  # their only per-request draft stats are the non-streaming probe


def concurrent_probe_verdict(mode, probe_responses, records):
    """Probe-only engines (mlx-vlm): the same N concurrent non-streaming requests, read per response.
    All drafted and accepted > 0 -> the mode (dflash); all zero or absent -> 'ar' with the label (the
    concurrent path does not draft; a label, not a violation); anything mixed -> unverified (fail closed)."""
    n = sum(len(r.get("requests") or []) for r in records)
    stats = {"responses": len(probe_responses or []), "requests": n}
    if not probe_responses or len(probe_responses) != n:
        return {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": "concurrent_probe",
                "source": "probe", "stats": stats, "label": None}, [
            f"concurrent probe: {len(probe_responses or [])} responses for {n} requests"]
    drafted = []
    for resp in probe_responses:
        t = (resp or {}).get("timings") or {}
        drafted.append((t.get("draft_n"), t.get("draft_n_accepted")))
    stats["probe"] = drafted
    if all(_is_positive(d) and _is_positive(a) for d, a in drafted):
        return {"engaged_mode": mode, "matched_line": f"concurrent probe draft_n/accepted {drafted}",
                "evidence_kind": "concurrent_probe", "source": "probe", "stats": stats, "label": None}, []
    if all(not _is_positive(d) and not _is_positive(a) for d, a in drafted):
        return {"engaged_mode": "ar", "matched_line": f"concurrent probe zero draft stats {drafted}",
                "evidence_kind": "concurrent_probe", "source": "probe", "stats": stats,
                "label": CONC_PROBE_AR_LABEL}, []
    return {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": "concurrent_probe",
            "source": "probe", "stats": stats, "label": None}, [
        f"concurrent probe mixed (some responses draft, some do not): {drafted}"]


def _concurrent_item(m, log):
    """The per-request done-line item of a mode: the first acceptance item whose regex matches the log
    (an engine can list several lines; the first listed may be a different line, e.g. kapitsa's spec:)."""
    items = [e for e in ((m or {}).get("evidence", []))
             if e.get("kind") == "acceptance_regex" and e.get("group")]
    for e in items:
        if re.search(str(e["pattern_or_path"]), log or ""):
            return e
    return items[0] if items else None


def concurrent_check(table, engine, mode, log, records, ready_offset, probe_responses=None):
    """THE concurrent verdict (in-job check and audit both call this). records: the cell's conc / mixed
    rows. Returns (ev, problems): ev has the evaluate() shape (engaged_mode, matched_line, evidence_kind,
    source, stats with per_row) when every record's requests are matched; problems lists every failure
    (fail closed). ev is None when there are no records."""
    if not records:
        return None, []
    if engine == "yunshu":
        return _yunshu_concurrent(records, mode, log, ready_offset)
    if engine in PROBE_ONLY_ENGINES and probe_responses is not None:
        return concurrent_probe_verdict(mode, probe_responses, records)
    if mode in AR_MODES:
        return _ar_concurrent(records, log, ready_offset)
    m = mode_entry(table, engine, mode)
    item = _concurrent_item(m, log)
    problems, matched = [], []
    consumed = 0
    total = sum(len(r.get("requests") or []) for r in records)
    skip = READINESS_REQUESTS + WARM_EXTRA.get(engine, 0)
    if item is not None and total != len(_done_candidates(item, log, ready_offset, skip)):
        n_lines = len(_done_candidates(item, log, ready_offset, skip))
        problems.append(f"cell: {n_lines} done lines for {total} requests")
    for rec in records:
        reqs = rec.get("requests")
        if not isinstance(reqs, list) or not reqs:
            problems.append(f"{rec.get('part')} n={rec.get('n')}: no requests list")
            continue
        if item is None:
            problems.append(f"no per-request acceptance item for {engine}/{mode}")
            continue
        found, why = correlate_multiset(item, log, reqs, ready_offset, consumed=consumed, skip=skip)
        consumed += len(reqs)
        if why is not None:
            problems.append(f"concurrent {rec.get('part')} n={rec.get('n')}: {why}")
        for mt in found:
            mt["queued_s"] = queued_s(mt["line"])
            mt["part"] = rec.get("part")
            matched.append(mt)
    if problems or not matched:
        return {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": "concurrent",
                "source": "slice", "stats": {"per_row_problem": "; ".join(problems) or "no requests",
                                             "per_row": matched}, "label": None}, problems
    per_row = [{"row": i, "prompt": x.get("prompt"), "line": x["line"], "value": x["value"],
                "group": x.get("group"), "queued_s": x.get("queued_s"),
                "rid": x.get("rid"), "part": x.get("part"),
                "calibration_share": x.get("calibration_share"), "calibrated": x.get("calibrated", False),
                "ct_expected": x.get("ct_expected"), "ct_merged": x.get("ct_merged"), "ct_gap": x.get("ct_gap"),
                "partial_coverage": x.get("partial_coverage", False)}
               for i, x in enumerate(matched)]
    partial = [x for x in per_row if x.get("partial_coverage")]
    calibrated = [x for x in matched if x.get("calibrated")]
    if calibrated and len(calibrated) == len(matched):
        return {"engaged_mode": "ar", "matched_line": matched[-1]["line"], "evidence_kind": "acceptance_regex",
                "source": "slice", "stats": {"per_row": per_row, "concurrent": len(matched)},
                "label": "ar (calibration only)"}, []
    label = None
    if calibrated:
        pct = round(100.0 * sum(x.get("calibration_share", 0.0) for x in matched) / len(matched))
        label = f"{mode} (partial: {pct}% calibration)"
    if partial:
        # the merged MTP sums miss tokens for these requests: the acceptance is a lower bound on the log, not a
        # verdict on the request; surfaced in the table note, not a violation (the uid count still fails closed)
        label = (label or mode) + f" [partial log coverage: {len(partial)} request(s) with ct gap > tolerance]"
    return {"engaged_mode": mode, "matched_line": matched[-1]["line"], "evidence_kind": "acceptance_regex",
            "source": "slice", "stats": {"per_row": per_row, "concurrent": len(matched),
                                         "partial_coverage": len(partial)},
            "label": label}, []


def acceptance_proven(ev: dict) -> bool:
    """True when the evidence evaluation itself proved drafting per measured request: every row has its
    own positive matched value (per_row), or the probe's draft stats are positive. This is the single
    source of the acceptance verdict; the record fields (attr.accepted) are not read for it."""
    stats = ev.get("stats") or {}
    if stats.get("per_row") and not stats.get("per_row_problem"):
        return all(float(r.get("value", 0)) > 0 for r in stats["per_row"])
    probe = stats.get("probe")
    if probe and all(_is_positive(v) for v in probe):
        return True
    return bool(stats.get("acceptance_positive_rows"))


def acceptance_problem(engaged_mode, m, rows) -> str | None:
    """For a speculative mode: why its per-request acceptance does not prove drafting, or None."""
    if engaged_mode in AR_MODES or engaged_mode == UNVERIFIED:
        return None
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return "speculative mode but no per-request records to check acceptance"
    paths = acceptance_paths(m)
    if any(_is_positive(_dotted(row, p)) for row in rows for p in paths):
        return None
    return f"speculative mode {engaged_mode!r} but every request's acceptance stat ({', '.join(paths)}) is zero or absent"


# mlx-vlm 0.7.11 streamed rows carry no draft stats; their decode speed is the server's own line
# (generation.py:397 "Request completed: ... decode=<x> tok/s"), one per request in order.
COMPLETED_DECODE = re.compile(r"Request completed: [^\n]*?decode=([0-9.]+) tok/s")
# the probe timings a row keeps (serving_path_vlm_omlx.json, streaming_vs_nonstreaming)
PROBE_FIELDS = (
    "draft_kind",
    "draft_rounds",
    "draft_n",
    "draft_n_accepted",
    "predicted_per_second",
    "predicted_n",
    "predicted_ms",
)


def completed_decode_tok_s(log: str) -> list[float]:
    """Decode speed of each completed request in the log, in order."""
    return [float(m.group(1)) for m in COMPLETED_DECODE.finditer(log or "")]


def probe_fields(response) -> dict:
    """The probe's timings subset kept in the record (None for a field the response lacks)."""
    timings = (response or {}).get("timings") or {}
    return {k: timings.get(k) for k in PROBE_FIELDS}


YUNSHU_RUNNER = re.compile(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b")


def yunshu_runner_evidence(log: str, ready_offset=None) -> dict:
    """Yunshu's own evidence: the initialized VLM batch runner line (the last one wins)."""
    log = log or ""
    found = list(YUNSHU_RUNNER.finditer(log))
    if not found:
        return {"engaged_mode": UNVERIFIED, "matched_line": None, "evidence_kind": None,
                "source": None, "stats": {}}
    m = found[-1]
    src = "startup_log" if ready_offset is not None and m.start() < ready_offset else "slice"
    return {
        "engaged_mode": m.group(1),
        "matched_line": _line_at(log, m.start(), m.end()),
        "evidence_kind": "log_regex",
        "source": src,
        "stats": {"runner_lines": len(found)},
    }
