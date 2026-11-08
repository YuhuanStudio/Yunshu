"""probe: deterministic engine counters from one short run per arm (no timing reps).

Parses the server log of a tfbench decode run into per-request counters (tokens/round, nodes/round,
rounds, acceptance histogram) plus an aggregate ms/round, joins them with the tfbench rows (decode
tok/s, TTFT, ct) and compares base with candidate. Greedy counters are deterministic, so any move
is flagged with its numbers; ms/round and tok/s move within a tolerance. Pure functions only (the
stage in stages.py runs the servers); no counter line = no evidence = error, never a pass."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROW_DRIVER = re.compile(
    r"Round driver (\w+) row: (\d+) rounds, ([\d.]+) tokens/round, ([\d.]+) nodes/round"
    r"(?:, (\d+) copy rounds)?(?:.*?accepted-length histogram (\[[\d, ]*\]))?"
)
WINDOW_DRIVER = re.compile(
    r"Round driver (\w+) window of (\d+) rounds \(total \d+\): .*?([\d.]+) ms/round"
)
PROFILE_OLD = re.compile(
    r"(\w+) round profile: (\d+) rounds, ([\d.]+) nodes/round, draft ([\d.]+) ms, "
    r"verify ([\d.]+) ms, commit\+host ([\d.]+) ms"
)

# defaults; every one overridable through the suite config
TOL = {
    "tpr_pct": 0.1,  # same log format: tokens/round (greedy-deterministic)
    "tpr_pct_cross": 3.0,  # base and cand print different formats (old lane vs driver)
    "nodes_abs": 0.1,  # nodes/round is printed with one decimal
    "ms_pct": 3.0,
    "tps_pct": 5.0,  # one rep: a loose timing flag only
    "ttft_pct": 3.0,  # client TTFT (cold / warm / turn-2): flagged beyond BOTH the % and the ms floor
    "ttft_abs_ms": 20.0,
    "stage_pct": 25.0,  # one per-request stage of the engine's own stage line
    "stage_abs_ms": 10.0,
}

TURN_STAGES = re.compile(r"turn stages: ids=(\d+) hit=(\d+) \| (.*?) total=([\d.]+) ms")
FN_TTFT = re.compile(
    r"fn_ttft path=(\w+) ids=(\d+) hit=(\d+) segs=\d+ \| (.*?) total=([\d.]+) ms"
)
_KV = re.compile(r"(\w+)=(-?[\d.]+)")


def parse_stage_log(text: str) -> dict:
    """Per-request stage records: ``turn`` (hit_stages ``turn stages:`` lines, the driver) and ``fn``
    (``fn_ttft`` lines, Flash-Next on the driver and the old lane), in log order. Never raises: an arm
    without stage lines simply has empty lists."""
    turn, fn = [], []
    for line in text.splitlines():
        m = TURN_STAGES.search(line)
        if m:
            st = {k: float(v) for k, v in _KV.findall(m[3])}
            turn.append(
                {"ids": int(m[1]), "hit": int(m[2]), "total": float(m[4]), "stages": st}
            )
            continue
        m = FN_TTFT.search(line)
        if m:
            st = {k: float(v) for k, v in _KV.findall(m[4])}
            fn.append(
                {
                    "path": m[1],
                    "ids": int(m[2]),
                    "hit": int(m[3]),
                    "total": float(m[5]),
                    "stages": st,
                }
            )
    return {"turn": turn, "fn": fn}


def join_stages(stage_log: dict, rows: list) -> list:
    """For each tfbench decode row (request order) the stage records of its request, matched by prompt
    length in order (rehearsals and warmups between rows are skipped; ``hit`` is not used because a
    replayed prompt tail reports a longer hit than the cache the client saw). None when absent."""
    dec = [r for r in rows if r.get("part") == "decode" and r.get("phase")]
    out = [{"turn": None, "fn": None} for _ in dec]
    for kind in ("turn", "fn"):
        recs, at = stage_log[kind], 0
        for i, row in enumerate(dec):
            ids = row.get("prompt_tokens") or row.get("pt")
            for j in range(at, len(recs)):
                if recs[j]["ids"] == ids:
                    out[i][kind] = recs[j]
                    at = j + 1
                    break
    return out


def parse_log(text: str) -> dict:
    """{"fmt", "requests": [{rounds, tpr|None, nodes, hist|None, ms|None}], "windows": [(rounds, ms)]}.
    Raises ValueError when no counter line exists."""
    reqs, windows, fmts = [], [], set()
    for line in text.splitlines():
        m = ROW_DRIVER.search(line)
        if m:
            fmts.add("driver")
            reqs.append(
                {
                    "rounds": int(m[2]),
                    "tpr": float(m[3]),
                    "nodes": float(m[4]),
                    "hist": json.loads(m[6]) if m[6] else None,
                    "ms": None,
                }
            )
            continue
        m = WINDOW_DRIVER.search(line)
        if m:
            windows.append((int(m[2]), float(m[3])))
            continue
        m = PROFILE_OLD.search(line)
        if m:
            fmts.add("lane")
            reqs.append(
                {
                    "rounds": int(m[2]),
                    "tpr": None,  # filled from the tfbench row's token count
                    "nodes": float(m[3]),
                    "hist": None,
                    "ms": float(m[4]) + float(m[5]) + float(m[6]),
                    "verify_ms": float(m[5]),
                }
            )
    if not reqs:
        raise ValueError(
            "no round counter line in the server log (driver row / round profile)"
        )
    if len(fmts) > 1:
        raise ValueError(f"mixed counter formats in one log: {sorted(fmts)}")
    return {"fmt": fmts.pop(), "requests": reqs, "windows": windows}


def join_rows(parsed: dict, rows: list) -> list:
    """Per decode row (request order), the counters of its request: the last N counter records
    (earlier ones are warmup). Fails closed when there are fewer records than rows."""
    dec = [r for r in rows if r.get("part") == "decode" and r.get("phase")]
    reqs = parsed["requests"]
    if not dec:
        raise ValueError("no tfbench decode rows")
    if len(reqs) < len(dec):
        raise ValueError(f"{len(reqs)} counter records for {len(dec)} decode requests")
    out = []
    for row, q in zip(dec, reqs[len(reqs) - len(dec) :]):  # noqa: B905
        tpr = q["tpr"]
        if tpr is None:
            tpr = (row.get("ct") or 0) / q["rounds"] if q["rounds"] else None
            # a lane profile line has no token count: the join assumes one line per decode request.
            # An older build that skips some requests (and prints warmup lines) shifts the pairing, and
            # a reply of 256 tokens "in 1 round" is the symptom (2026-11-08 push-gate). A round commits
            # at most nodes+1 tokens, so anything above that is a mispaired record: fail closed.
            if tpr is not None and tpr > q["nodes"] + 1:
                raise ValueError(
                    f"counter record ({q['rounds']} rounds, {q['nodes']} nodes/round) cannot belong to "
                    f"a {row.get('ct')}-token reply: lane profile lines do not pair with the requests"
                )
        out.append(
            {
                "cell": f"{row.get('kind')}@{row.get('ctx')}:{row.get('phase')}",
                "rounds": q["rounds"],
                "tpr": tpr,
                "nodes": q["nodes"],
                "hist": q["hist"],
                "dec_tps": row.get("dec_tps"),
                "ttft_s": row.get("ttft_s"),
            }
        )
    return out


def arm_ms(parsed: dict) -> float | None:
    """Mean ms/round over the run, weighted by rounds."""
    if parsed["fmt"] == "driver":
        ws = parsed["windows"]
        tot = sum(r for r, _ in ws)
        return sum(r * m for r, m in ws) / tot if tot else None
    qs = [q for q in parsed["requests"] if q["ms"] is not None]
    tot = sum(q["rounds"] for q in qs)
    return sum(q["rounds"] * q["ms"] for q in qs) / tot if tot else None


def summarize(text: str, rows: list) -> dict:
    stage_log = parse_stage_log(text)
    try:
        parsed = parse_log(text)
        cells = join_rows(parsed, rows)
    except ValueError:
        # no decode round counters (e.g. main's Flash-Next MTP lane): the TTFT stage lines can still be
        # compared; without them there is no evidence at all and the caller fails closed / hands over
        if not (stage_log["turn"] or stage_log["fn"]):
            raise
        parsed = {"fmt": "none", "requests": [], "windows": []}
        cells = [
            {
                "cell": f"{r.get('kind')}@{r.get('ctx')}:{r.get('phase')}",
                "rounds": None,
                "tpr": None,
                "nodes": None,
                "hist": None,
                "dec_tps": r.get("dec_tps"),
                "ttft_s": r.get("ttft_s"),
            }
            for r in rows
            if r.get("part") == "decode" and r.get("phase")
        ]
    dec = [r for r in rows if r.get("part") == "decode" and r.get("phase")]
    for cell, stages in zip(cells, join_stages(stage_log, rows)):  # noqa: B905
        cell["stages"] = stages
    return {
        "fmt": parsed["fmt"],
        "ms_per_round": arm_ms(parsed) if parsed["fmt"] != "none" else None,
        "cells": cells,
        "stage_lines": bool(stage_log["turn"] or stage_log["fn"]),
        "stage_schema": 1,
        "n_dec": len(dec),
    }


def _pct(a: float, b: float) -> float:
    return 100.0 * (b - a) / a if a else float("inf")


def compare(base: dict, cand: dict, tol: dict | None = None) -> dict:
    """{"moved": [str], "ok": bool, "numbers": {...}}. `moved` lists every counter outside its
    tolerance with both numbers; empty = nothing moved (PASS without timing reps)."""
    t = {**TOL, **(tol or {})}
    cross = base["fmt"] != cand["fmt"]
    moved, cells = [], []
    bc, cc = base["cells"], cand["cells"]
    if [c["cell"] for c in bc] != [c["cell"] for c in cc]:
        return {
            "ok": False,
            "moved": [
                f"cell lists differ: {[c['cell'] for c in bc]} vs {[c['cell'] for c in cc]}"
            ],
            "numbers": {},
        }
    tpr_tol = t["tpr_pct_cross"] if cross else t["tpr_pct"]
    nocount = "none" in (
        base["fmt"],
        cand["fmt"],
    )  # an arm without round counters: TTFT stages only
    for b, c in zip(bc, cc):  # noqa: B905
        d = {"cell": b["cell"]}
        for name in ("tpr", "nodes", "rounds"):
            d[name] = [b[name], c[name]]
        if nocount:
            pass
        elif b["tpr"] is None or c["tpr"] is None:
            moved.append(f"{b['cell']}: tokens/round missing")
        elif abs(_pct(b["tpr"], c["tpr"])) > tpr_tol:
            moved.append(f"{b['cell']}: tokens/round {b['tpr']:.2f} -> {c['tpr']:.2f}")
        if not nocount and abs(c["nodes"] - b["nodes"]) > t["nodes_abs"]:
            moved.append(
                f"{b['cell']}: nodes/round {b['nodes']:.1f} -> {c['nodes']:.1f}"
            )
        if not nocount and not cross and b["rounds"] != c["rounds"]:
            moved.append(f"{b['cell']}: rounds {b['rounds']} -> {c['rounds']}")
        if b["hist"] and c["hist"] and b["hist"] != c["hist"]:
            moved.append(
                f"{b['cell']}: acceptance histogram {b['hist']} -> {c['hist']}"
            )
        if b["dec_tps"] and c["dec_tps"]:
            dp = _pct(b["dec_tps"], c["dec_tps"])
            d["dec_tps"] = [b["dec_tps"], c["dec_tps"], round(dp, 1)]
            if b["cell"].endswith(":cold") and dp < -t["tps_pct"]:
                moved.append(
                    f"{b['cell']}: decode tok/s {b['dec_tps']:.1f} -> {c['dec_tps']:.1f} ({dp:+.1f}%)"
                )
        if b["ttft_s"] and c["ttft_s"]:
            d["ttft_s"] = [
                b["ttft_s"],
                c["ttft_s"],
                round(_pct(b["ttft_s"], c["ttft_s"]), 1),
            ]
        cells.append(d)
    tt = compare_ttft(base, cand, t)
    moved += tt["moved"]
    for d, e in zip(cells, tt["cells"]):  # noqa: B905
        d.update({k: v for k, v in e.items() if k != "cell"})
    bm, cm = base["ms_per_round"], cand["ms_per_round"]
    if bm and cm:
        if _pct(bm, cm) > t["ms_pct"]:
            moved.append(
                f"ms/round {bm:.1f} -> {cm:.1f} ({_pct(bm, cm):+.1f}%, limit {t['ms_pct']}%)"
            )
    elif not nocount:
        moved.append("ms/round missing in an arm (no window/profile line)")
    return {
        "ok": not moved,
        "moved": moved,
        "numbers": {
            "formats": [base["fmt"], cand["fmt"]],
            "ms_per_round": [bm, cm],
            "cells": cells,
            "tol": t,
        },
    }


def _stage_moves(cell: str, label: str, b: dict, c: dict, t: dict) -> list:
    """Every stage (and the total) present in both records that moved beyond its ms floor AND %."""
    out = []
    pairs = {"total": (b["total"], c["total"])}
    for k in b["stages"]:
        if k in c["stages"]:
            pairs[k] = (b["stages"][k], c["stages"][k])
    for k, (x, y) in pairs.items():
        d = y - x
        if d > t["stage_abs_ms"] and _pct(max(x, 1e-9), y) > t["stage_pct"]:
            out.append(f"{cell}: {label} stage {k} {x:.1f} -> {y:.1f} ms ({d:+.1f})")
    return out


def compare_ttft(base: dict, cand: dict, tol: dict | None = None) -> dict:
    """TTFT part of the probe: per cell the client TTFT, then the engine's stage lines. Flags the cell
    whose client TTFT grew beyond the ms floor AND the %, and every stage that did the same where both
    arms print comparable stage records (``turn stages`` anywhere; ``fn_ttft`` stages only when both are
    the same path, otherwise just its total). When the client TTFT moved the cell's stage numbers of both
    arms are attached so the culprit stage is visible even across formats."""
    t = {**TOL, **(tol or {})}
    moved, cells = [], []
    for b, c in zip(base["cells"], cand["cells"]):  # noqa: B905
        d = {"cell": b["cell"]}
        bt, ct = b.get("ttft_s"), c.get("ttft_s")
        flagged = False
        if bt and ct:
            dms = (ct - bt) * 1e3
            pct = _pct(bt, ct)
            d["ttft_ms"] = [round(bt * 1e3, 1), round(ct * 1e3, 1), round(pct, 1)]
            if dms > t["ttft_abs_ms"] and pct > t["ttft_pct"]:
                flagged = True
                moved.append(
                    f"{b['cell']}: client TTFT {bt * 1e3:.0f} -> {ct * 1e3:.0f} ms ({dms:+.0f} ms, {pct:+.1f}%)"
                )
        bs, cs = b.get("stages") or {}, c.get("stages") or {}
        stage_moves = []
        if bs.get("turn") and cs.get("turn"):
            stage_moves += _stage_moves(b["cell"], "turn", bs["turn"], cs["turn"], t)
        if bs.get("fn") and cs.get("fn"):
            if bs["fn"]["path"] == cs["fn"]["path"]:
                stage_moves += _stage_moves(b["cell"], "fn_ttft", bs["fn"], cs["fn"], t)
            else:
                stage_moves += _stage_moves(
                    b["cell"],
                    f"fn_ttft {bs['fn']['path']}->{cs['fn']['path']}",
                    {"total": bs["fn"]["total"], "stages": {}},
                    {"total": cs["fn"]["total"], "stages": {}},
                    t,
                )
        moved += stage_moves
        if flagged or stage_moves:
            d["stages"] = {
                "base": {k: v and v["stages"] for k, v in bs.items()},
                "cand": {k: v and v["stages"] for k, v in cs.items()},
            }
            if flagged and not stage_moves:
                top = {
                    k: dict(sorted(v["stages"].items(), key=lambda kv: -kv[1])[:3])
                    for k, v in cs.items()
                    if v
                }
                moved.append(
                    f"{b['cell']}: TTFT moved outside every comparable stage; cand's largest stages {top}"
                )
        cells.append(d)
    return {"moved": moved, "cells": cells}


def verdict_text(res: dict) -> str:
    if res["ok"]:
        return "probe PASS: counters and ms/round unchanged; no timing reps needed"
    return "probe flagged, run the speed stage: " + "; ".join(res["moved"])


def cache_path(root: Path, key: str) -> Path:
    return root / "probe" / f"{key}.json"


def cache_load(root: Path, key: str) -> dict | None:
    p = cache_path(root, key)
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("cells") else None


def cache_store(root: Path, key: str, summary: dict) -> None:
    p = cache_path(root, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary))
    tmp.replace(p)
