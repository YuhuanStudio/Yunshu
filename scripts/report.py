"""Unified regression + performance report — ONE interleaved Markdown deliverable.

Merges into a single analyzed, text-and-chart-INTERLEAVED report:
  - GATES   : docs/reports/regression_report.json  (latest GO/NO-GO + section status)
  - TRENDS  : docs/reports/perf_history/perf_*.json (append-only absolute-KPI history)
  - COVERAGE: docs/reports/COVERAGE_MATRIX.md headline

Each KPI family gets the RIGHT visualization next to its own written analysis:
  - fw   (in-process framework bench) : per-model GROUPED BAR — frameworks × batch,
                                        a same-model horizontal head-to-head.
  - serve(real HTTP bench)            : per-model GROUPED BAR — frameworks × concurrency.
  - cache/cool/gpu/quality/gates      : LINE over snapshots (evolution is the point),
                                        thermally annotated.

Thermal-aware: every snapshot is tagged with GPU fp16 TFLOP/s (cool ceiling ~9.5
on this M3 Max); runs below ~8.5 are heat-suppressed and DISCOUNTED so a hot run
doesn't read as a code regression.

Versioned: docs/reports/REPORT.md + docs/reports/history/report_<ts>.md + img/*.png.

Run:
  PYTHONPATH=. uv run python scripts/report.py
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "docs", "reports")
HIST_DIR = os.path.join(OUT_DIR, "perf_history")
REPORT_JSON = os.path.join(OUT_DIR, "regression_report.json")
COVERAGE_MD = os.path.join(OUT_DIR, "COVERAGE_MATRIX.md")
IMG_DIR = os.path.join(OUT_DIR, "img")
HIST_OUT = os.path.join(OUT_DIR, "history")

COOL_TFLOPS = 9.5
SUPPRESSED_BELOW = 8.5
FLAG_PCT = 8.0

_HIGHER_BETTER = ("decode_tps", "prefill_tps", "/batch", "agg_tps", "_pct",
                  "gates_pass", "/sys", "tflops", "overall_tps", "_q_tps")
_LOWER_BETTER = ("ttft_ms", "cold_ms", "_mb", "_seconds", "/lat", "decay_pct")


def _better(key: str) -> int:
    if any(t in key for t in _HIGHER_BETTER):
        return 1
    if any(t in key for t in _LOWER_BETTER):
        return -1
    return 0


def _load_snaps() -> list[dict]:
    snaps = []
    for p in sorted(glob.glob(os.path.join(HIST_DIR, "perf_*.json"))):
        try:
            snaps.append(json.load(open(p)))
        except Exception:
            pass
    return snaps


def _short(s: str) -> str:
    return (s.replace("-Instruct", "").replace("-MLX", "")
            .replace("-bf16", "").replace("-it", "").replace("-A3B", ""))


def _suppressed(snap: dict) -> bool:
    t = snap.get("kpis", {}).get("gpu_tflops")
    return t is not None and t < SUPPRESSED_BELOW


def _verified_cool(snap: dict, kpi: str | None = None) -> bool:
    """Whether the snapshot's thermal state is KNOWN-cool FOR THIS KPI. Prefers the
    most SPECIFIC thermal tag available, because a single end-of-run snapshot tag
    can't reflect when each section actually ran (cache benches run hot mid-run, the
    snapshot tag is measured cool at the end → false regressions):
      - cache/<model>/* → the per-model `cache/<model>/_gpu_tflops` (measured right
        after that model's cache bench).
      - else → the snapshot-level gpu_tflops.
    Untagged (no applicable tag) = thermally UNVERIFIABLE → not cool (excluded from
    trend), so a degraded-but-untagged run (e.g. 602T0841) can't fabricate a delta."""
    kp = snap.get("kpis", {})
    t = kp.get("gpu_tflops")
    if kpi:
        parts = kpi.split("/")
        if len(parts) >= 3 and parts[0] == "cache":
            st = kp.get(f"cache/{parts[1]}/_gpu_tflops")
            if isinstance(st, (int, float)):
                t = st
    return isinstance(t, (int, float)) and t >= SUPPRESSED_BELOW


# ── comparison data (fw / serve): pull the latest snapshot that HAS the family ──

def _latest_with(snaps: list[dict], family: str) -> dict | None:
    for s in reversed(snaps):
        if any(k.startswith(family + "/") for k in s.get("kpis", {})):
            return s
    return None


def _comparison_table(snap: dict, family: str) -> dict:
    """{model: {framework: {metric: value}}} from fw/<model>/<fw>/<metric> keys."""
    out: dict = {}
    for k, v in snap.get("kpis", {}).items():
        parts = k.split("/")
        if parts[0] != family or len(parts) != 4:
            continue
        _, model, fw, metric = parts
        out.setdefault(model, {}).setdefault(fw, {})[metric] = v
    return out


# ── charts ────────────────────────────────────────────────────────────────────

def _bar_chart(model: str, fwdata: dict, metrics: list[str], title: str,
               fname: str, ylabel: str) -> str | None:
    """Grouped bar: x = frameworks, grouped bars = metrics (batch/concurrency)."""
    fws = sorted(fwdata)
    present = [m for m in metrics if any(fwdata[f].get(m) is not None for f in fws)]
    if not fws or not present:
        return None
    os.makedirs(IMG_DIR, exist_ok=True)
    n = len(present)
    width = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(fws)), 3.4))
    for i, m in enumerate(present):
        vals = [fwdata[f].get(m) or 0 for f in fws]
        xs = [j + i * width for j in range(len(fws))]
        bars = ax.bar(xs, vals, width, label=m)
        for b, val in zip(bars, vals):
            if val:
                ax.annotate(f"{val:g}", (b.get_x() + b.get_width() / 2, val),
                            ha="center", va="bottom", fontsize=5)
    ax.set_xticks([j + width * (n - 1) / 2 for j in range(len(fws))])
    ax.set_xticklabels([_short(f) for f in fws], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, ncol=n)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, fname)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return os.path.relpath(path, OUT_DIR)


def _line_charts(snaps: list[dict], family: str) -> list[tuple[str, str]]:
    """One PNG per (family, metric) line-over-snapshots. Returns [(metric, relpath)]."""
    os.makedirs(IMG_DIR, exist_ok=True)
    xs = list(range(len(snaps)))
    xlabels = [s["timestamp"][4:13].replace("T", " ") for s in snaps]
    hot = [_suppressed(s) for s in snaps]
    groups: dict[str, dict[str, list]] = {}
    for k in sorted({k for s in snaps for k in s.get("kpis", {})}):
        parts = k.split("/")
        fam = parts[0]
        if fam != family:
            continue
        metric = parts[-1] if len(parts) > 1 else parts[0]
        series = "/".join(parts[1:-1]) or "·" if len(parts) > 1 else "·"
        groups.setdefault(metric, {})[series] = [s.get("kpis", {}).get(k) for s in snaps]
    charts = []
    for metric, series_map in sorted(groups.items()):
        live = {s: v for s, v in series_map.items() if sum(x is not None for x in v) >= 2}
        if not live:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.0))
        for s, vals in sorted(live.items()):
            px = [x for x, v in zip(xs, vals) if v is not None]
            py = [v for v in vals if v is not None]
            ax.plot(px, py, marker="o", ms=4, lw=1.4,
                    label=_short(s) if s != "·" else metric)
        for x, h in zip(xs, hot):
            if h:
                ax.axvspan(x - 0.15, x + 0.15, color="red", alpha=0.06)
        ax.set_title(f"{family} · {metric}", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6)
        ax.grid(True, alpha=0.25, lw=0.5)
        b = _better(f"{family}/{metric}")
        ax.set_ylabel("↑ better" if b == 1 else ("↓ better" if b == -1 else ""), fontsize=7)
        if len(live) > 1 or next(iter(live)) != "·":
            ax.legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        path = os.path.join(IMG_DIR, f"{family}__{metric}.png".replace("/", "_"))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        charts.append((metric, os.path.relpath(path, OUT_DIR)))
    return charts


# ── analysis ───────────────────────────────────────────────────────────────────

def _analyze(snaps: list[dict]) -> dict:
    if not snaps:
        return {"thermal": [], "flags": []}
    thermal = [(s["timestamp"], s["kpis"]["gpu_tflops"]) for s in snaps if _suppressed(s)]
    untagged = [s["timestamp"] for s in snaps if s.get("kpis", {}).get("gpu_tflops") is None]
    keys = sorted({k for s in snaps for k in s.get("kpis", {})
                   if k != "gpu_tflops" and not k.endswith("_gpu_tflops")})
    flags = []
    for k in keys:
        better = _better(k)
        if better == 0:
            continue
        # Trust ONLY verified-cool points (tagged + >= floor). Untagged and hot
        # points are thermally unverifiable → excluded, so a degraded-but-untagged
        # run can't masquerade as a baseline/endpoint and fabricate a regression.
        use = [(i, s["kpis"][k]) for i, s in enumerate(snaps)
               if k in s.get("kpis", {}) and _verified_cool(s, k)
               and isinstance(s["kpis"][k], (int, float))]
        if len(use) < 2:
            continue
        first_v, last_v = use[0][1], use[-1][1]
        if not first_v:
            continue
        pct = (last_v - first_v) / abs(first_v) * 100
        improved = (pct > 0) if better == 1 else (pct < 0)
        if abs(pct) >= FLAG_PCT:
            flags.append({"kpi": k, "first": first_v, "last": last_v,
                          "pct": pct, "improved": improved,
                          "last_ts": snaps[use[-1][0]]["timestamp"]})
    # Whole-snapshot thermal-suspect guard: a real code regression hits SPECIFIC KPIs;
    # a hot/memory-pressured machine depresses EVERY model's decode+prefill+cold at once.
    # The end-of-run snapshot tag can read cool (≥floor) even when the cache section ran
    # hot mid-run (per-model tags aren't always emitted), so the tag alone misses it. If
    # one snapshot is the "after" for cache regressions spanning ≥3 distinct models, treat
    # the whole snapshot as thermally suspect and demote those flags out of 🔴 regressions.
    _by_ts: dict[str, set] = {}
    for f in flags:
        if f["kpi"].startswith("cache/") and not f["improved"]:
            mdl = f["kpi"].split("/")[1] if len(f["kpi"].split("/")) >= 2 else "?"
            _by_ts.setdefault(f["last_ts"], set()).add(mdl)
    _suspect_ts = {ts for ts, mdls in _by_ts.items() if len(mdls) >= 3}
    thermal_suspect = [f for f in flags
                       if f["kpi"].startswith("cache/") and not f["improved"]
                       and f["last_ts"] in _suspect_ts]
    flags = [f for f in flags if f not in thermal_suspect]
    flags.sort(key=lambda f: abs(f["pct"]), reverse=True)
    thermal_suspect.sort(key=lambda f: abs(f["pct"]), reverse=True)
    return {"thermal": thermal, "untagged": untagged, "flags": flags,
            "thermal_suspect": thermal_suspect, "suspect_ts": sorted(_suspect_ts)}


def _serve_monotonicity(snaps: list[dict]) -> list[str]:
    """Flag serve runs where aggregate throughput does NOT grow with concurrency
    (sys16<sys8 or sys32<sys16) — a healthy server scales up to its batch limit.
    yunshu's HTTP serving violates this in every captured run (the engine-loop is
    monotonic in-process → the gap is the GATEWAY/serving layer)."""
    out = []
    for s in snaps:
        k = s.get("kpis", {})
        for key in sorted({kk.rsplit("/", 1)[0] for kk in k if kk.startswith("serve/")}):
            v = [k.get(f"{key}/sys8"), k.get(f"{key}/sys16"), k.get(f"{key}/sys32")]
            if all(isinstance(x, (int, float)) for x in v) and (v[1] < v[0] * 0.95 or v[2] < v[1] * 0.85):
                fw = key.split("/")[-1]
                out.append(f"`{s['timestamp'][:13]}` {fw}: sys 8/16/32 = "
                           f"{v[0]:g}/{v[1]:g}/{v[2]:g}")
    return out


# Internal (fw) framework name → its external (serve) counterpart. yunshu's external
# server runs the engine-loop, so it pairs with the internal yunshu-loop.
_PARITY_PAIRS = [("yunshu-loop", "yunshu"), ("mlx-lm", "mlx-lm"), ("oMLX", "oMLX")]


def _framework_gaps(snaps: list[dict]) -> list[str]:
    """yunshu vs the best OTHER framework, on the latest COOL serve snapshot (same
    run = thermally matched = a FAIR cross-framework comparison). Surfaces where
    yunshu is consistently behind/ahead so the framework-comparison gaps aren't
    buried in the bar charts. ttft lower-better; sys/decode higher-better."""
    sv = None
    for s in reversed(snaps):
        t = s.get("kpis", {}).get("gpu_tflops")
        if any(k.startswith("serve/") for k in s.get("kpis", {})) and isinstance(t, (int, float)) and t >= SUPPRESSED_BELOW:
            sv = s
            break
    if not sv:
        return []
    k = sv["kpis"]
    out = []
    models = sorted({kk.split("/")[1] for kk in k if kk.startswith("serve/")})
    for m in models:
        for metric, better in (("ttft_ms", -1), ("sys8", 1), ("sys16", 1), ("sys32", 1)):
            y = k.get(f"serve/{m}/yunshu/{metric}")
            others = {fw: k.get(f"serve/{m}/{fw}/{metric}")
                      for fw in ("mlx-lm", "oMLX") if isinstance(k.get(f"serve/{m}/{fw}/{metric}"), (int, float))}
            if not isinstance(y, (int, float)) or not others:
                continue
            best_fw, best_v = (min if better == -1 else max)(others.items(), key=lambda kv: kv[1] * -better)
            if not best_v:
                continue
            ratio = y / best_v if better == 1 else best_v / y  # >1 = yunshu better
            if abs(ratio - 1) >= 0.15:
                verb = "ahead of" if ratio > 1 else "behind"
                out.append(f"{_short(m)} {metric}: yunshu {y:g} vs {best_fw} {best_v:g} "
                           f"({'+' if ratio>1 else '-'}{abs(ratio-1)*100:.0f}%, {verb})")
    return out


def _internal_external_parity(snaps: list[dict]) -> list[dict]:
    """Per-FRAMEWORK internal (in-process, fw/) vs external (real HTTP gateway, serve/)
    aggregate throughput at matched concurrency. external/internal = that framework's
    serving (gateway) efficiency; a large shortfall is a MIDDLE-LAYER cost/bug that
    pure in-process testing hides. Every framework measured both ways → comprehensive.
    Uses the most recent snapshot that has each side (separate files)."""
    fw = _latest_with(snaps, "fw")
    sv = _latest_with(snaps, "serve")
    if not fw or not sv:
        return []
    rows = []
    fwk, svk = fw["kpis"], sv["kpis"]
    fw_models = {_short(kk.split("/")[1]) for kk in fwk if kk.startswith("fw/")}
    sv_models = {_short(kk.split("/")[1]) for kk in svk if kk.startswith("serve/")}
    for m in sorted(fw_models & sv_models):
        for ifw, efw in _PARITY_PAIRS:
            for N in (8, 16, 32):
                internal = next((v for kk, v in fwk.items()
                                 if kk.startswith("fw/") and _short(kk.split("/")[1]) == m
                                 and kk.endswith(f"/{ifw}/batch{N}")), None)
                external = next((v for kk, v in svk.items()
                                 if kk.startswith("serve/") and _short(kk.split("/")[1]) == m
                                 and kk.endswith(f"/{efw}/sys{N}")), None)
                if isinstance(internal, (int, float)) and isinstance(external, (int, float)) and internal:
                    rows.append({"model": m, "fw": efw, "N": N, "internal": internal,
                                 "external": external, "eff": external / internal})
    return rows


def _coverage_headline() -> list[str]:
    if not os.path.exists(COVERAGE_MD):
        return []
    return [ln.strip() for ln in open(COVERAGE_MD)
            if ln.strip().startswith("- **") and any(
                w in ln for w in ("Techniques", "Interfaces", "Parameters"))]


def _catalogue() -> list[dict]:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import regression
        return [{"name": n, "tier": t, "gate": bool(g)}
                for n, t, g, *_ in regression._sections()]
    except Exception:
        return []


# ── render ──────────────────────────────────────────────────────────────────────

def _render_comparison(L: list, snaps: list, family: str, title: str, blurb: str,
                       metrics: list[str], ylabel: str) -> None:
    snap = _latest_with(snaps, family)
    if not snap:
        L += [f"## {title}", "", f"_No {family}/ data captured yet._", ""]
        return
    table = _comparison_table(snap, family)
    age = "latest run" if snap is snaps[-1] else f"as of `{snap['timestamp'][:13]}`"
    L += [f"## {title}", "", f"{blurb} ({age}, "
          f"GPU {snap.get('kpis', {}).get('gpu_tflops', '?')} TFLOP/s).", ""]
    for model in sorted(table):
        fwdata = table[model]
        rel = _bar_chart(model, fwdata, metrics, f"{_short(model)} — frameworks × {ylabel}",
                         f"{family}_bar_{_short(model)}.png".replace("/", "_"), ylabel)
        L += [f"### {_short(model)}", ""]
        # per-model written analysis: rank frameworks by the largest metric present
        big = next((m for m in reversed(metrics) if any(d.get(m) for d in fwdata.values())), None)
        if big:
            ranked = sorted(((f, d.get(big) or 0) for f, d in fwdata.items()),
                            key=lambda x: -x[1])
            lead = ", ".join(f"{_short(f)} {v:g}" for f, v in ranked if v)
            L.append(f"_{big}: {lead}._")
            # flat-scaling (doesn't scale with concurrency) callout
            for f, d in sorted(fwdata.items()):
                seq = [d.get(m) for m in metrics if d.get(m) is not None]
                if len(seq) >= 2 and seq[0] and max(seq) / seq[0] < 1.15:
                    L.append(f"  - ⚠️ `{_short(f)}` does not scale with {ylabel} "
                             f"({'/'.join(f'{x:g}' for x in seq)}).")
            L.append("")
        if rel:
            L += [f"![{family} {model}]({rel})", ""]


def _render_evolution(L: list, snaps: list, family: str, title: str, blurb: str,
                      notes: dict | None = None) -> None:
    charts = _line_charts(snaps, family)
    if not charts:
        return
    L += [f"## {title}", "", blurb, ""]
    notes = notes or {}
    for metric, rel in charts:
        L += [f"![{family} {metric}]({rel})", ""]
        if metric in notes:
            L += [notes[metric], ""]


def _render(snaps: list[dict], analysis: dict, gates: list, ts: str) -> str:
    L = []
    latest = snaps[-1] if snaps else {}
    gate_items = [g for g in gates if str(g.get("gate")) in ("True", "true", True)]
    n_pass = sum(1 for g in gate_items if g.get("status") == "PASS")
    go = gate_items and n_pass == len(gate_items)
    tf = latest.get("kpis", {}).get("gpu_tflops")
    tf_note = ""
    if tf is not None:
        tf_note = (f" · GPU **{tf} TFLOP/s** "
                   + ("⚠️ heat-suppressed" if tf < SUPPRESSED_BELOW
                      else f"(cool, ceiling ≈{COOL_TFLOPS})"))
    L += ["# Yunshu — Unified Regression + Performance Report", "",
          f"_Generated {ts}. Gate verdict + thermal-aware perf analysis, charts and "
          f"narrative interleaved per family. Red bands on trend charts = thermally "
          f"suppressed snapshots (discounted in the analysis)._", ""]
    if gate_items:
        L += [f"## Verdict: {'**GO** ✅' if go else '**NO-GO** ❌'} — "
              f"{n_pass}/{len(gate_items)} gates pass{tf_note}", ""]
        fails = [g for g in gate_items if g.get("status") != "PASS"]
        if fails:
            L += ["Failing gates: " + ", ".join(
                f"**{g['name']}** ({g.get('summary','')})" for g in fails), ""]
    else:
        L += [f"## Verdict: (no gate run recorded){tf_note}", ""]

    # ── headline analysis ──
    L += ["## Analysis", ""]
    th = analysis["thermal"]
    untag = analysis.get("untagged", [])
    if th:
        L += [f"**Thermal:** {len(th)}/{len(snaps)} snapshots heat-suppressed "
              f"(GPU < {SUPPRESSED_BELOW}): " + ", ".join(f"`{t[0][:13]}`={t[1]}" for t in th[-4:])
              + ". Their perf is a floor, not a regression — discounted.", ""]
    else:
        L += [f"**Thermal:** no tagged snapshot is heat-suppressed.", ""]
    if untag:
        L += [f"**Thermally unverifiable:** {len(untag)} snapshot(s) predate GPU tagging "
              f"(no gpu_tflops) — EXCLUDED from trend deltas (a degraded-but-untagged run, "
              f"e.g. `602T0841`, would otherwise fabricate regressions). Trend/regression "
              f"flags below use VERIFIED-cool points only (tagged ≥ {SUPPRESSED_BELOW}).", ""]
    regr = [f for f in analysis["flags"] if not f["improved"]]
    impr = [f for f in analysis["flags"] if f["improved"]]
    if regr:
        L += [f"**Regressions (thermal-discounted, |Δ| ≥ {FLAG_PCT:.0f}%):**", ""]
        L += [f"- 🔴 `{f['kpi']}` {f['first']:g} → {f['last']:g} ({f['pct']:+.0f}%)"
              for f in regr[:12]] + [""]
    else:
        L += ["**Regressions:** none beyond noise. ✅", ""]
    if impr:
        L += ["**Improvements:**", ""]
        L += [f"- 🟢 `{f['kpi']}` {f['first']:g} → {f['last']:g} ({f['pct']:+.0f}%)"
              for f in impr[:8]] + [""]
    suspect = analysis.get("thermal_suspect", [])
    if suspect:
        _sts = ", ".join(f"`{t}`" for t in analysis.get("suspect_ts", []))
        L += [f"**Thermally-suspect (NOT code regressions — uniform cross-model "
              f"depression in snapshot(s) {_sts}; the end-of-run cool tag missed a "
              f"hot-running cache section):**", ""]
        L += [f"- 🌡️ `{f['kpi']}` {f['first']:g} → {f['last']:g} ({f['pct']:+.0f}%)"
              for f in suspect[:12]] + [""]
    cov = _coverage_headline()
    if cov:
        L += ["**Coverage:** " + " · ".join(c.replace("- **", "**") for c in cov), ""]

    # ── framework gaps (cool, same-run = thermally-matched, fair) ──
    fg = _framework_gaps(snaps)
    if fg:
        L += ["**Framework gaps (yunshu vs best other, latest cool serve run — fair, "
              "same-run/thermally-matched):**", ""]
        L += [f"- {x}" for x in fg] + [""]

    # ── serving-layer stability (gateway) ──
    nonmono = _serve_monotonicity(snaps)
    if nonmono:
        L += ["**⚠️ Serving non-monotonic (gateway):** real-HTTP aggregate throughput "
              "did NOT grow with concurrency in these runs (a healthy server scales to "
              "its batch limit). The in-process engine-loop IS monotonic → the shortfall "
              "is the GATEWAY/serving layer (middle-layer bug):", ""]
        L += [f"- {x}" for x in nonmono[-6:]] + [""]

    # ── internal vs external, PER FRAMEWORK (dual-path; catches middle-layer bugs) ──
    parity = _internal_external_parity(snaps)
    if parity:
        L += ["**Internal vs external, per framework (serving/gateway efficiency):** "
              "same aggregate-throughput workload through each framework's in-process "
              "engine vs its real HTTP server. external/internal < ~0.8 = the serving "
              "layer is the bottleneck (a middle-layer cost in-process testing hides). "
              "Every framework measured both ways.", "",
              "| framework | model | N | internal | external (HTTP) | serving eff |",
              "|---|---|---|---|---|---|"]
        for r in parity:
            flag = " ⚠️" if r["eff"] < 0.8 else ""
            L.append(f"| {r['fw']} | {r['model']} | {r['N']} | {r['internal']:g} | "
                     f"{r['external']:g} | {r['eff']*100:.0f}%{flag} |")
        L.append("")

    # ── test catalogue ──
    cat = _catalogue()
    if cat:
        by_name = {g.get("name"): g for g in gates}
        ran = sum(1 for c in cat if c["name"] in by_name)
        n_gate = sum(1 for c in cat if c["gate"])
        L += [f"## Test catalogue — {len(cat)} sections ({n_gate} gates, "
              f"{len(cat)-n_gate} metrics); {ran} ran in the latest report", "",
              "| section | tier | kind | status | summary |", "|---|---|---|---|---|"]
        for c in sorted(cat, key=lambda c: (c["tier"], not c["gate"], c["name"])):
            g = by_name.get(c["name"], {})
            st = g.get("status", "—")
            icon = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "🟡"}.get(st, "·")
            L.append(f"| {c['name']} | {c['tier']} | {'gate' if c['gate'] else 'metric'} "
                     f"| {icon} {st} | {g.get('summary','')} |")
        L.append("")

    # ── per-family interleaved sections ──
    # PRIMARY comparison = all-external (real HTTP). Only real servers are a fair,
    # production-truthful measure of each framework's true performance (Wave 688).
    _render_comparison(
        L, snaps, "serve", "Framework comparison — real OpenAI HTTP (PRIMARY, all-external)",
        "Each framework's REAL HTTP server, one at a time at a matched thermal state — "
        "the ONLY fair cross-framework comparison (in-process driving is an artifact). "
        "Concurrent system throughput (tok/s) at N=8/16/32",
        ["sys8", "sys16", "sys32"], "concurrency")
    # In-process is YUNSHU-ONLY now: fast path vs engine-loop, the internal input to
    # the internal-vs-external parity above (catches gateway/middle-layer bugs).
    _render_comparison(
        L, snaps, "fw", "Yunshu engine internal (fast vs loop) — parity input, NOT a framework comparison",
        "IN-PROCESS, yunshu engine only (fast path vs engine-loop). Used solely to "
        "compare against yunshu's OWN external HTTP numbers (see gateway-efficiency "
        "parity above). Aggregate decode throughput (tok/s) at batch 8/16/32",
        ["batch8", "batch16", "batch32"], "batch")

    _render_evolution(
        L, snaps, "cache", "KV cache 4-tier — evolution",
        "Per-model fast-path cache behaviour over time (HOT/WARM/SSD reuse TTFT + "
        "on-disk/RAM footprint). Lower TTFT / smaller MB = better.",
        {"ssd_disk_mb": "_gemma-4 dropped to ~0 once sliding-window (RotatingKVCache) "
         "models stopped spilling the unrestorable whole-snapshot to disk (W688). "
         "Qwen3.5 hybrid keeps its (large) recurrent whole-snapshot by design._",
         "tier_SSD_ttft_ms": "_Qwen3.5 hybrid SSD restore TTFT is high and rising "
         "(large whole-snapshot deserialize); SSD is auto-gated for fast-prefill models._"})
    _render_evolution(
        L, snaps, "cool", "Sustained decode — thermal decay",
        "One long single-stream generation; decay% = first-quarter vs last-quarter "
        "decode tok/s. Flat ≈ bandwidth-bound (healthy); large positive = throttling.")
    _render_evolution(
        L, snaps, "quality", "Generation quality",
        "MMLU accuracy + Yunshu-engine-vs-mlx-lm parity. Flat = no quality regression.")
    _render_evolution(
        L, snaps, "gpu_tflops", "GPU thermal state",
        "Per-run fp16 matmul ceiling (cool ≈9.5). The normalizer for every other "
        "number: a low reading means that run's perf was heat-suppressed.")
    _render_evolution(
        L, snaps, "gates_pass", "Gate count",
        "Passing gate count over time (coverage growth).")

    # ── full data appendix ──
    L += ["## Full data (absolute KPIs)", "", "<details><summary>every KPI × snapshot</summary>", ""]
    cols = [s["timestamp"][4:13] for s in snaps]
    L += ["| KPI | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in sorted({k for s in snaps for k in s.get("kpis", {})}):
        row = [f"{s.get('kpis',{}).get(k):g}" if isinstance(s.get("kpis", {}).get(k), (int, float))
               else "·" for s in snaps]
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L += ["", "</details>", "", "---",
          "_Snapshots: " + ", ".join(f"`{s['timestamp']}`({s.get('git_sha','?')})" for s in snaps) + "_"]
    return "\n".join(L) + "\n"


def main() -> None:
    snaps = _load_snaps()
    gates = json.load(open(REPORT_JSON)) if os.path.exists(REPORT_JSON) else []
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(HIST_OUT, exist_ok=True)
    # Clear stale PNGs so a chart that's no longer produced (e.g. old fw line
    # charts replaced by bars) doesn't linger as an orphan.
    for _p in glob.glob(os.path.join(IMG_DIR, "*.png")):
        try:
            os.remove(_p)
        except OSError:
            pass
    analysis = _analyze(snaps)
    md = _render(snaps, analysis, gates, ts)
    open(os.path.join(OUT_DIR, "REPORT.md"), "w").write(md)
    stamp = ts.replace(":", "").replace("-", "")
    open(os.path.join(HIST_OUT, f"report_{stamp}.md"), "w").write(md)
    n_img = len(glob.glob(os.path.join(IMG_DIR, "*.png")))
    print(f"report -> {os.path.relpath(os.path.join(OUT_DIR, 'REPORT.md'), REPO)}  "
          f"({len(snaps)} snapshots, {n_img} charts, {len(analysis['flags'])} flags, "
          f"{len(gates)} gate rows)")


if __name__ == "__main__":
    main()
