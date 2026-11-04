#!/usr/bin/env python3
"""Stream a fullcmp run into Weights & Biases as CHARTS, unattended (CPU only; reads finished jsonl files).

Each round (only when a result file changed) it renders the comparison charts with matplotlib and logs them as
images, so the W&B workspace shows real plots that update by themselves (the media panel always shows the
newest round; its slider goes back in time):
  charts/cold_ttft_<kind>        cold TTFT vs context, one line per engine (log scale)
  charts/decode_<phase>_<kind>   decode tok/s vs context (cold, warm, turn2)
  charts/turn2_ttft_<kind>       follow-up TTFT vs context (log scale)
  charts/concurrency             aggregate tok/s at 2/4/8 concurrent requests (1K, distinct prompts)
  charts/mixed                   mixed load (one long + short requests): aggregate tok/s and long-request TTFT
  progress/*                     scalars over time: cell-reps done, % done
It also logs one `cells` table (every cell x phase: medians, rep spread, cached tokens, labels) for lookups.

    venv-wandb/bin/python scripts/dev/fullcmp_wandb.py --raw <raw root> --manifest <run>/manifest.jsonl \
        --project yunshu-fullcmp --name <run name> --until-pid <driver pid>

Run under `taskpolicy -b nice -n 19` so it never shows up in a measured job's quiet gate. Needs `wandb login`
once. Uploads aggregated numbers, labels and the rendered charts only (no prompts, outputs or logs).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullcmp_report as fr  # noqa: E402

CTX_ORDER = ["1K", "4K", "8K", "16K", "32K", "64K", "128K"]
COLORS = ["#0f6e63", "#d0742c", "#4a6fd1", "#a0438f", "#5d8a2f", "#b8902a", "#2f8fa8", "#c4453d",
          "#6b5bb5", "#7d7d2b", "#3c6e8f", "#9a5a3a"]
COLUMNS = ["batch", "family", "engine", "ctx", "conc", "kind", "out", "variant", "phase", "n",
           "ttft_s", "dec_tps", "agg_tps", "dec_min", "dec_max", "ttft_min", "ttft_max", "cached",
           "ttft_s_spread_pct", "dec_tps_spread_pct", "agg_tps_spread_pct", "label"]


def color(engine: str) -> str:
    try:
        return COLORS[fr.ENGINE_ORDER.index(engine) % len(COLORS)]
    except ValueError:
        return "#777777"


def _fig():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    ax.grid(True, alpha=0.3)
    return plt, fig, ax


def _img(wandb, plt, fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return wandb.Image(Image.open(buf))


def ctx_chart(wandb, rows, kind, phase, key, title, ylabel, log=False):
    plt, fig, ax = _fig()
    drew = False
    for eng in fr.ENGINE_ORDER:
        pts = {r["ctx"]: r.get(key) for r in rows if r["engine"] == eng and r["phase"] == phase and r["conc"] == 1
               and r["kind"] == kind and r["out"] == 512 and r["family"] in ("core", "ctx") and r.get(key)}
        xs = [i for i, c in enumerate(CTX_ORDER) if c in pts]
        if not xs:
            continue
        ax.plot(xs, [pts[CTX_ORDER[i]] for i in xs], marker="o", color=color(eng), label=eng, linewidth=2)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xticks(range(len(CTX_ORDER)), CTX_ORDER)
    ax.set_xlabel("context length")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return _img(wandb, plt, fig)


def conc_chart(wandb, rows):
    plt, fig, ax = _fig()
    levels = [2, 4, 8]
    engines = [e for e in fr.ENGINE_ORDER if any(r["engine"] == e and r["phase"] == "conc" and r["family"] == "conc"
                                                  and r["ctx"] == "1K" for r in rows)]
    if not engines:
        plt.close(fig)
        return None
    width = 0.8 / len(engines)
    for i, eng in enumerate(engines):
        vals = []
        for n in levels:
            v = [r.get("agg_tps") for r in rows if r["engine"] == eng and r["phase"] == "conc"
                 and r["family"] == "conc" and r["ctx"] == "1K" and r["conc"] == n]
            vals.append(v[0] if v and v[0] else 0)
        ax.bar([x + i * width for x in range(len(levels))], vals, width, color=color(eng), label=eng)
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(levels))], [f"{n} concurrent" for n in levels])
    ax.set_ylabel("aggregate tok/s")
    ax.set_title("Concurrent throughput (1K prompts, distinct)")
    ax.legend(fontsize=8, ncol=2)
    return _img(wandb, plt, fig)


def mixed_chart(wandb, rows):
    mix = [r for r in rows if r["phase"] == "mixed"]
    if not mix:
        return None
    plt, fig, ax = _fig()
    engines = [e for e in fr.ENGINE_ORDER if any(r["engine"] == e for r in mix)]
    xs = list(range(len(engines)))
    agg = [next((r.get("agg_tps") or 0 for r in mix if r["engine"] == e), 0) for e in engines]
    ttl = [next((r.get("ttft_max") or 0 for r in mix if r["engine"] == e), 0) for e in engines]
    ax.bar(xs, agg, color=[color(e) for e in engines])
    ax.set_xticks(xs, engines, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("aggregate tok/s")
    ax2 = ax.twinx()
    ax2.plot(xs, ttl, "k^", label="long request TTFT (s)")
    ax2.set_ylabel("long-request TTFT s")
    ax2.legend(fontsize=8, loc="upper right")
    ax.set_title("Mixed load (one long + short requests)")
    return _img(wandb, plt, fig)


def snapshot(wandb, run, raw: str, manifest: str) -> None:
    rows, meta = fr.collect(raw)
    prog = fr.progress(manifest, raw)
    done = sum(p["done"] for p in prog)
    planned = sum(p["planned"] for p in prog) or 1
    log = {"progress/done": done, "progress/planned": planned, "progress/pct": 100.0 * done / planned,
           "progress/unsupported_cells": len(meta["unsupported"])}
    for kind in ("code", "prose"):
        for name, phase, key, title, ylabel, lg in (
            (f"charts/cold_ttft_{kind}", "cold", "ttft_s", f"Cold TTFT vs context ({kind})", "seconds (log)", True),
            (f"charts/decode_cold_{kind}", "cold", "dec_tps", f"Cold decode speed ({kind})", "tok/s", False),
            (f"charts/decode_warm_{kind}", "warm", "dec_tps", f"Warm decode speed ({kind})", "tok/s", False),
            (f"charts/decode_turn2_{kind}", "turn2", "dec_tps", f"Follow-up (turn 2) decode speed ({kind})", "tok/s", False),
            (f"charts/turn2_ttft_{kind}", "turn2", "ttft_s", f"Follow-up (turn 2) TTFT ({kind})", "seconds (log)", True),
        ):
            img = ctx_chart(wandb, rows, kind, phase, key, title, ylabel, lg)
            if img is not None:
                log[name] = img
    for name, fn in (("charts/concurrency", conc_chart), ("charts/mixed", mixed_chart)):
        img = fn(wandb, rows)
        if img is not None:
            log[name] = img
    log["cells"] = wandb.Table(columns=COLUMNS, data=[[r.get(c) for c in COLUMNS] for r in rows])
    run.log(log)
    run.summary.update({"updated": time.strftime("%Y-%m-%d %H:%M:%S")})


def alive(pid: int) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--project", default="yunshu-fullcmp")
    ap.add_argument("--name", required=True)
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--until-pid", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    import wandb

    run = wandb.init(project=a.project, name=a.name, config={"raw": a.raw, "manifest": a.manifest},
                     dir=os.environ.get("WANDB_DIR", "/Volumes/P5Plus/yunshu-build/wandb"))
    last_sig = None
    while True:
        try:
            # skip when no result file changed (a snapshot parses every file)
            sig = max((e.stat().st_mtime for d in os.scandir(a.raw) if d.is_dir() for e in os.scandir(d.path)),
                      default=0)
            if sig != last_sig:
                snapshot(wandb, run, a.raw, a.manifest)
                last_sig = sig
        except Exception as exc:  # noqa: BLE001 - keep streaming; a half-written cell file is retried next round
            print(f"snapshot failed: {exc}", file=sys.stderr, flush=True)
        if a.once or not alive(a.until_pid):
            if not a.once:
                snapshot(wandb, run, a.raw, a.manifest)
            break
        time.sleep(a.interval)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
