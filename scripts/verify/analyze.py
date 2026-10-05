"""Pure analysis of stage evidence (no I/O beyond the rows handed in; CPU-unit-tested)."""

from __future__ import annotations

import statistics
from collections.abc import Iterable

PHASES = ("cold", "warm", "turn2")


def decode_index(rows: Iterable[dict]) -> dict:
    """tfbench decode records keyed by (ctx, kind, phase); the last row of a key wins."""
    out = {}
    for r in rows:
        if r.get("part") == "decode" and r.get("phase") in PHASES:
            out[(r.get("ctx"), r.get("kind"), r["phase"])] = r
    return out


def session_info(rows: Iterable[dict]) -> dict:
    for r in rows:
        if r.get("part") == "session":
            return r
    return {}


def compare_identity(a_rows, b_rows, a_name="base", b_name="cand", ctxs=None) -> dict:
    """Greedy digests equal on every cell both arms produced; a missing cell fails closed."""
    a, b = decode_index(a_rows), decode_index(b_rows)
    keys = sorted(k for k in set(a) | set(b) if ctxs is None or k[0] in ctxs)
    bad, n = [], 0
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        if ra is None or rb is None:
            bad.append(
                {
                    "cell": list(k),
                    "why": f"missing in {a_name if ra is None else b_name}",
                }
            )
            continue
        n += 1
        if ra.get("sha") != rb.get("sha") or ra.get("ct") != rb.get("ct"):
            bad.append(
                {
                    "cell": list(k),
                    "why": "digest differs",
                    a_name: {
                        "sha": ra.get("sha"),
                        "ct": ra.get("ct"),
                        "finish": ra.get("finish"),
                    },
                    b_name: {
                        "sha": rb.get("sha"),
                        "ct": rb.get("ct"),
                        "finish": rb.get("finish"),
                    },
                }
            )
    return {"compared": n, "mismatches": bad, "ok": n > 0 and not bad}


def check_apc(rows, require_hit: bool = True) -> dict:
    """Prefix-cache hit == miss: per (ctx, kind) the warm request (served from the cache) must
    produce the cold request's digest, and report cached tokens when a hit is required."""
    idx = decode_index(rows)
    cells = sorted({(k[0], k[1]) for k in idx})
    bad, n, hits = [], 0, []
    for ctx, kind in cells:
        cold, warm = idx.get((ctx, kind, "cold")), idx.get((ctx, kind, "warm"))
        if cold is None or warm is None:
            bad.append({"cell": [ctx, kind], "why": "cold or warm record missing"})
            continue
        n += 1
        hits.append({"cell": [ctx, kind], "cached": warm.get("cached")})
        if cold.get("sha") != warm.get("sha") or cold.get("ct") != warm.get("ct"):
            bad.append(
                {
                    "cell": [ctx, kind],
                    "why": "warm (cache hit) differs from cold (miss)",
                    "cold": cold.get("sha"),
                    "warm": warm.get("sha"),
                }
            )
        if require_hit and not (warm.get("cached") or 0) > 0:
            bad.append(
                {
                    "cell": [ctx, kind],
                    "why": f"no cache hit (cached={warm.get('cached')}): APC not engaged",
                }
            )
    return {"compared": n, "mismatches": bad, "hits": hits, "ok": n > 0 and not bad}


# ── speed ────────────────────────────────────────────────────────────────
METRICS = {
    # name: (phase, field, higher_is_better)
    "decode_tps": ("cold", "dec_tps", True),
    "cold_ttft_s": ("cold", "ttft_s", False),
    "followup_ttft_s": ("turn2", "ttft_s", False),
    "warm_ttft_s": ("warm", "ttft_s", False),
}


def _pct(a: float, b: float) -> float:
    return (b / a - 1.0) * 100.0


def speed_compare(base_reps: list, cand_reps: list, tol_pct: float = 2.0) -> dict:
    """Per (ctx, kind, metric): medians, per-rep paired deltas, noise = half range of the paired
    deltas; a regression is a median worsening beyond max(tol, noise). Reps pair by index."""
    out, regress, missing = [], [], []
    n = min(len(base_reps), len(cand_reps))
    if n == 0:
        return {
            "ok": False,
            "cells": [],
            "regressions": [],
            "missing": ["no reps"],
            "reps": 0,
        }
    bidx = [decode_index(r) for r in base_reps[:n]]
    cidx = [decode_index(r) for r in cand_reps[:n]]
    cells = sorted({(k[0], k[1]) for i in bidx + cidx for k in i})
    for ctx, kind in cells:
        for name, (phase, fld, higher) in METRICS.items():
            bv, cv = [], []
            for i in range(n):
                rb, rc = (
                    bidx[i].get((ctx, kind, phase)),
                    cidx[i].get((ctx, kind, phase)),
                )
                vb = rb.get(fld) if rb else None
                vc = rc.get(fld) if rc else None
                if not vb or not vc:
                    continue
                bv.append(vb)
                cv.append(vc)
            if len(bv) < n:
                missing.append(f"{kind}@{ctx} {name}: {len(bv)}/{n} paired reps")
                continue
            deltas = [_pct(a, b) for a, b in zip(bv, cv)]  # noqa: B905 - Python 3.9
            med = _pct(statistics.median(bv), statistics.median(cv))
            noise = (max(deltas) - min(deltas)) / 2.0 if n > 1 else 0.0
            worse = -med if higher else med  # positive = candidate worse
            limit = max(tol_pct, noise)
            verdict = (
                "regression"
                if worse > limit
                else ("improvement" if -worse > limit else "neutral")
            )
            row = {
                "ctx": ctx,
                "kind": kind,
                "metric": name,
                "base_median": round(statistics.median(bv), 3),
                "cand_median": round(statistics.median(cv), 3),
                "delta_pct": round(med, 2),
                "rep_deltas_pct": [round(d, 2) for d in deltas],
                "noise_pct": round(noise, 2),
                "limit_pct": round(limit, 2),
                "verdict": verdict,
            }
            out.append(row)
            if verdict == "regression":
                regress.append(row)
    return {
        "ok": not regress and not missing and bool(out),
        "cells": out,
        "regressions": regress,
        "missing": missing,
        "reps": n,
        "tol_pct": tol_pct,
    }


# ── memory ───────────────────────────────────────────────────────────────
def memory_arm_metrics(rows: Iterable[dict], arm: str) -> dict:
    """Per rep: peak sampled footprint, footprint after idle, footprint held after the short
    follow-up; medians across reps (GiB)."""
    peak, idle, held, apc = {}, {}, {}, {}
    for r in rows:
        if r.get("arm") != arm or "footprint_gib" not in r:
            continue
        rep = r.get("rep", 0)
        peak[rep] = max(
            peak.get(rep, 0.0), r.get("peak_footprint_gib") or 0.0, r["footprint_gib"]
        )
        if r.get("step") == "idle20s":
            idle[rep] = r["footprint_gib"]
        if r.get("step") == "idle-after":
            held[rep] = r["footprint_gib"]
            apc[rep] = sum(
                v
                for k, v in r.items()
                if "apc_resident_bytes" in k and isinstance(v, (int, float))
            )
    med = lambda d: round(statistics.median(d.values()), 3) if d else None  # noqa: E731
    return {"peak": med(peak), "idle": med(idle), "held": med(held), "reps": len(peak)}


def memory_compare(
    rows, base="base", cand="cand", tol_pct: float = 3.0, abs_gib: float = 0.25
) -> dict:
    b, c = memory_arm_metrics(rows, base), memory_arm_metrics(rows, cand)
    cells, regress, missing = [], [], []
    for m in ("peak", "idle", "held"):
        if b[m] is None or c[m] is None:
            missing.append(m)
            continue
        limit = b[m] * tol_pct / 100.0 + abs_gib
        d = c[m] - b[m]
        row = {
            "metric": m,
            "base_gib": b[m],
            "cand_gib": c[m],
            "delta_gib": round(d, 3),
            "limit_gib": round(limit, 3),
        }
        cells.append(row)
        if d > limit:
            regress.append(row)
    return {
        "ok": not regress and not missing and b["reps"] > 0 and c["reps"] > 0,
        "cells": cells,
        "regressions": regress,
        "missing": missing,
        "reps": [b["reps"], c["reps"]],
    }


# ── long-context retrieval (needle) ───────────────────────────────────────
def needle_scores(rows) -> dict:
    """{ctx: {item: bool}} from needle records (the last record of an item wins)."""
    out: dict = {}
    for r in rows:
        if r.get("part") == "needle":
            out.setdefault(r["ctx"], {})[r["item"]] = bool(r.get("correct"))
    return out


def needle_compare(base_rows, cand_rows, allowed: int = 1) -> dict:
    """Retrieval answers: net difference in correct items <= `allowed`; an item missing in
    either arm is a failure. `per_ctx` lists [base, cand, items] per context."""
    b, c = needle_scores(base_rows), needle_scores(cand_rows)
    per_ctx, missing, bt, ct, items = {}, [], 0, 0, 0
    for ctx in sorted(set(b) | set(c)):
        bi, ci = b.get(ctx, {}), c.get(ctx, {})
        ids = sorted(set(bi) | set(ci))
        miss = [i for i in ids if i not in bi or i not in ci]
        if miss or not ids:
            missing.append(f"ctx {ctx}: items {miss or 'none'}")
        both = [i for i in ids if i in bi and i in ci]
        per_ctx[ctx] = [sum(bi[i] for i in both), sum(ci[i] for i in both), len(both)]
        bt += per_ctx[ctx][0]
        ct += per_ctx[ctx][1]
        items += len(both)
    net = ct - bt
    return {
        "items": items,
        "base_correct": bt,
        "cand_correct": ct,
        "net": net,
        "per_ctx": per_ctx,
        "missing": missing,
        "ok": items > 0 and not missing and abs(net) <= allowed,
        "allowed": allowed,
    }


# ── quality ──────────────────────────────────────────────────────────────
def quality_compare(base_rows, cand_rows, target_n: int, allowed: int = 1) -> dict:
    """Paired accuracy: net difference in correct answers <= `allowed`; every item scored in
    both arms (an unscored item is a failure, not a skip)."""

    def last(rows):
        d = {}
        for r in rows:
            if (
                r.get("kind") == "q"
                and not r.get("error")
                and r.get("correct") is not None
            ):
                d[r["id"]] = bool(r["correct"])
        return d

    b, c = last(base_rows), last(cand_rows)
    ids = sorted(set(b) & set(c))
    bc, cc = sum(b[i] for i in ids), sum(c[i] for i in ids)
    only_b = sum(1 for i in ids if b[i] and not c[i])
    only_c = sum(1 for i in ids if c[i] and not b[i])
    complete = len(ids) >= target_n
    net = cc - bc
    return {
        "n": len(ids),
        "target_n": target_n,
        "base_correct": bc,
        "cand_correct": cc,
        "net": net,
        "base_only": only_b,
        "cand_only": only_c,
        "complete": complete,
        "ok": complete and abs(net) <= allowed,
        "allowed": allowed,
    }
