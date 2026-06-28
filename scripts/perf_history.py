"""Time-named performance history + evolution analysis.

Problem this solves: REGRESSION_REPORT.md is OVERWRITTEN each run and leans on
"×N" ratios computed against a baseline that shifts between runs (each run's own
cold-prefill, or that run's mlx-lm), so you can't tell real improvement from
baseline drift. This module keeps an APPEND-ONLY, time-named history of the
ABSOLUTE numbers and renders a trend so performance evolution is visible.

Two modes:
  snapshot   extract absolute KPIs from docs/reports/regression_report.json and append a
             new docs/reports/perf_history/perf_<UTC-timestamp>.json (never overwrites)
  trend      load ALL snapshots and render docs/reports/PERF_TREND.md — per-KPI absolute
             value over time + Δ vs previous + Δ vs first, with ↑/↓ direction

Run:
  PYTHONPATH=. uv run python scripts/perf_history.py snapshot   # after a regression
  PYTHONPATH=. uv run python scripts/perf_history.py trend
  PYTHONPATH=. uv run python scripts/perf_history.py both       # snapshot then trend
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Wave 688 docs reorg: all machine-generated reports live under docs/reports/.
HIST_DIR = os.path.join(REPO, "docs", "reports", "perf_history")
REPORT_JSON = os.path.join(REPO, "docs", "reports", "regression_report.json")
TREND_MD = os.path.join(REPO, "docs", "reports", "PERF_TREND.md")

# KPI direction: True = higher is better, False = lower is better.
# Matched by substring on the KPI key.
_HIGHER_BETTER = ("decode_tps", "prefill_tps", "/batch", "agg_tps", "_pct", "gates_pass",
                  "/sys", "tflops")
_LOWER_BETTER = ("ttft_ms", "cold_ms", "_mb", "_seconds", "/lat")


def _short(model: str) -> str:
    return os.path.basename(str(model)).replace("-MLX", "").replace("-Instruct", "")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _pct(s: str):
    import re
    m = re.search(r"([\d.]+)%", str(s))
    return float(m.group(1)) if m else None


def _gpu_tflops(seconds: float = 2.0) -> float | None:
    """Measure the GPU's CURRENT sustained fp16 matmul throughput (TFLOP/s) — a
    self-calibrating THERMAL-STATE tag (no temp sensor / sudo needed).

    Why every snapshot needs it: PERF_TREND showed an entire full-run column down
    25-40% across ALL frameworks (incl. external mlx-lm/vLLM/oMLX) — that is a hot,
    memory-pressured machine, NOT a code regression. Without a per-snapshot thermal
    reading there is no way to tell the two apart, so an honest trend MUST tag each
    run. This M3 Max's COOL ceiling for 8192³ fp16 is ~9.5 TFLOP/s (Wave 657); a
    snapshot reading well below that means its perf numbers were thermally
    suppressed and should be read in that light, not as regressions.
    """
    try:
        import time as _t

        import mlx.core as mx
        N = 8192
        a = mx.random.normal((N, N), dtype=mx.float16)
        b = mx.random.normal((N, N), dtype=mx.float16)
        mx.eval(a @ b)  # warm
        t0 = _t.perf_counter()
        iters = 0
        while _t.perf_counter() - t0 < seconds:
            mx.eval(a @ b)
            iters += 1
        dt = _t.perf_counter() - t0
        del a, b
        mx.clear_cache()
        return round((2 * N ** 3 * iters) / dt / 1e12, 1)
    except Exception:
        return None


# ── KPI extraction from a regression_report.json ──────────────────────────────

def extract_kpis(results: list) -> dict:
    """Flatten the ABSOLUTE numbers out of a regression result list."""
    kpis: dict[str, float] = {}
    for it in results:
        name = it.get("name", "")
        metrics = it.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        art = metrics.get("artifact")

        # framework speed: per model, per framework — ttft / decode / batch agg
        if "framework speed" in name and isinstance(art, list):
            for entry in art:
                model = _short(entry.get("model", "?"))
                for row in entry.get("rows", []):
                    fw = row.get("fw", "?")
                    if isinstance(row.get("ttft_ms"), (int, float)):
                        kpis[f"fw/{model}/{fw}/ttft_ms"] = round(row["ttft_ms"], 1)
                    if isinstance(row.get("decode_tps"), (int, float)):
                        kpis[f"fw/{model}/{fw}/decode_tps"] = round(row["decode_tps"], 1)
                    for b, v in (row.get("batch") or {}).items():
                        if isinstance(v, (int, float)):
                            kpis[f"fw/{model}/{fw}/batch{b}"] = round(v, 1)

        # cache-tier matrix: per model fast-path absolute + per-tier TTFT
        if "cache-tier matrix" in name and isinstance(art, list):
            for entry in art:
                model = _short(entry.get("name", "?"))
                ft = entry.get("ft") or {}
                for k in ("prefill_tps", "decode_tps", "cold_ms",
                          "hot_entry_mb", "warm_entry_mb", "ssd_disk_mb"):
                    if isinstance(ft.get(k), (int, float)):
                        kpis[f"cache/{model}/{k}"] = round(ft[k], 2)
                # per-model thermal tag (measured right after THIS model's cache
                # bench) — lets the report thermal-discount cache KPIs accurately
                # instead of trusting the coarse end-of-run snapshot tag.
                if isinstance(ft.get("_gpu_tflops"), (int, float)):
                    kpis[f"cache/{model}/_gpu_tflops"] = round(ft["_gpu_tflops"], 1)
                for tier, td in (ft.get("tiers") or {}).items():
                    if isinstance((td or {}).get("ttft_ms"), (int, float)):
                        kpis[f"cache/{model}/tier_{tier}_ttft_ms"] = round(td["ttft_ms"], 1)

        # quality KPIs
        if name.startswith("quality: MMLU"):
            v = _pct(it.get("summary", ""))
            if v is not None:
                kpis["quality/mmlu_pct"] = v
        if name.startswith("quality vs mlx-lm"):
            ea = metrics.get("engine_acc")
            if isinstance(ea, (int, float)):
                kpis["quality/vs_mlx_engine_pct"] = ea

    # overall gate pass count
    gated = [r for r in results if str(r.get("gate")) in ("True", "true", True)]
    if gated:
        kpis["gates_pass"] = sum(1 for r in gated if r.get("status") == "PASS")
        kpis["gates_total"] = len(gated)
    return kpis


# ── snapshot ──────────────────────────────────────────────────────────────────

def snapshot(report_json: str = REPORT_JSON) -> str | None:
    if not os.path.exists(report_json):
        print(f"SKIP snapshot: {report_json} not found")
        return None
    results = json.load(open(report_json))
    kpis = extract_kpis(results)
    # Only snapshot runs that produced real metric data (full/standard) — a smoke
    # run has no framework/cache numbers and would pollute the trend with gaps.
    has_perf = any(k.startswith(("fw/", "cache/")) for k in kpis)
    if not has_perf:
        print("snapshot skipped: no framework/cache metrics (smoke run)")
        return None
    # Always tag the run's thermal state so the trend can tell a hot/throttled
    # machine apart from a real code regression (see _gpu_tflops).
    _tf = _gpu_tflops()
    if _tf is not None:
        kpis["gpu_tflops"] = _tf
        print(f"  thermal tag: {_tf} TFLOP/s (cool ceiling ≈9.5 on this M3 Max)")
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tier = "unknown"
    go = all(r.get("status") == "PASS" for r in results if str(r.get("gate")) in ("True", True))
    snap = {"timestamp": ts, "git_sha": _git_sha(), "tier": tier, "go": go,
            "n_kpis": len(kpis), "has_perf": has_perf, "kpis": kpis}
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"perf_{ts}.json")
    json.dump(snap, open(path, "w"), indent=2)
    print(f"snapshot -> {os.path.relpath(path, REPO)}  ({len(kpis)} KPIs, has_perf={has_perf})")
    return path


def snapshot_from_kpis(kpis: dict, source: str = "bench_serve", extra: dict | None = None) -> str | None:
    """Append a time-named perf-history snapshot from an arbitrary absolute-KPI
    dict (e.g. bench_serve's server-based numbers). Same append-only file format
    as snapshot(), so these feed the same PERF_TREND.md evolution view.
    """
    if not kpis:
        return None
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = {"timestamp": ts, "git_sha": _git_sha(), "tier": source, "go": True,
            "n_kpis": len(kpis), "has_perf": True, "kpis": kpis}
    if extra:
        snap.update(extra)
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"perf_{ts}.json")
    json.dump(snap, open(path, "w"), indent=2)
    print(f"snapshot -> {os.path.relpath(path, REPO)}  ({len(kpis)} KPIs, source={source})")
    return path


# ── trend ───────────────────────────────────────────────────────────────────

def _load_snaps() -> list[dict]:
    snaps = []
    for p in sorted(glob.glob(os.path.join(HIST_DIR, "perf_*.json"))):
        try:
            snaps.append(json.load(open(p)))
        except Exception:
            pass
    return [s for s in snaps if s.get("has_perf")] or snaps


def _better(key: str) -> int:
    if any(t in key for t in _HIGHER_BETTER):
        return 1
    if any(t in key for t in _LOWER_BETTER):
        return -1
    return 0


def _arrow(delta: float, key: str) -> str:
    if delta == 0:
        return "→"
    d = _better(key)
    if d == 0:
        return "↑" if delta > 0 else "↓"
    improved = (delta > 0) if d == 1 else (delta < 0)
    return "🟢" + ("↑" if delta > 0 else "↓") if improved else "🔴" + ("↑" if delta > 0 else "↓")


def trend() -> str | None:
    snaps = _load_snaps()
    if not snaps:
        print("no perf-history snapshots yet")
        return None
    # all KPI keys across history
    keys = sorted({k for s in snaps for k in s.get("kpis", {})})
    cols = [s["timestamp"][:13] for s in snaps]  # YYYYMMDDTHH

    lines = ["# Performance trend (absolute, append-only)", "",
             f"_{len(snaps)} snapshot(s); newest = rightmost. Absolute values — NOT ratios "
             "(baselines shift between runs). 🟢 = improved vs first run, 🔴 = regressed._", ""]
    lines.append(f"Snapshots: " + ", ".join(
        f"`{s['timestamp']}`({s.get('git_sha','?')})" for s in snaps))
    lines.append("")

    # group keys by area prefix
    def area(k):
        return k.split("/")[0]
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(area(k), []).append(k)

    for g in sorted(groups):
        lines += [f"## {g}", "", "| metric | " + " | ".join(cols)
                  + " | Δ first→last |", "|---|" + "---|" * (len(cols) + 1)]
        for k in groups[g]:
            series = [s.get("kpis", {}).get(k) for s in snaps]
            cells = []
            prev = None
            for v in series:
                if v is None:
                    cells.append("·")
                else:
                    cells.append(f"{v:g}")
                prev = v
            # delta first→last over the non-null endpoints
            nn = [(i, v) for i, v in enumerate(series) if v is not None]
            if len(nn) >= 2:
                first_v, last_v = nn[0][1], nn[-1][1]
                d = last_v - first_v
                pct = (d / first_v * 100) if first_v else 0.0
                delta = f"{_arrow(d, k)} {d:+g} ({pct:+.0f}%)"
            else:
                delta = "—"
            lines.append("| " + k.split("/", 1)[-1] + " | " + " | ".join(cells) + " | " + delta + " |")
        lines.append("")

    os.makedirs(os.path.dirname(TREND_MD), exist_ok=True)
    open(TREND_MD, "w").write("\n".join(lines) + "\n")
    print(f"trend -> {os.path.relpath(TREND_MD, REPO)}  ({len(snaps)} snapshots, {len(keys)} KPIs)")
    return TREND_MD


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("snapshot", "both"):
        snapshot()
    if mode in ("trend", "both"):
        trend()
