#!/usr/bin/env python3
"""Stream a fullcmp run into Weights & Biases, unattended (CPU only; reads finished jsonl files).

Every --interval seconds it re-reads the raw root through fullcmp_report.collect and logs to one W&B run:
  - progress (cell-reps done / planned, overall and per engine),
  - a table of every cell x phase (medians over reps, rep spread, cached tokens, labels),
  - per-engine summary series (cold TTFT, cold/warm/turn2 decode, concurrent throughput) keyed by context,
so the W&B charts update by themselves. It exits after --until-pid has exited and one final upload.

    venv-wandb/bin/python scripts/dev/fullcmp_wandb.py --raw /Volumes/Micron/yunshu-fullcmp/real-1104 \
        --manifest <run>/manifest.jsonl --project yunshu-fullcmp --name dev0-real-1104 --until-pid <driver pid>

Needs `wandb login` once on this machine (the key lives in ~/.netrc). Uploads only the aggregated numbers
and labels (no prompts, no model outputs, no logs).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullcmp_report as fr  # noqa: E402

CTX_ORDER = ["1K", "4K", "8K", "16K", "32K", "64K", "128K"]
COLUMNS = ["batch", "family", "engine", "ctx", "conc", "kind", "out", "variant", "phase", "n",
           "ttft_s", "dec_tps", "agg_tps", "dec_min", "dec_max", "ttft_min", "ttft_max", "cached",
           "ttft_s_spread_pct", "dec_tps_spread_pct", "agg_tps_spread_pct", "label"]


def alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def snapshot(wandb, run, raw: str, manifest: str, step: int) -> None:
    rows, meta = fr.collect(raw)
    prog = fr.progress(manifest, raw)
    done = sum(p["done"] for p in prog)
    planned = sum(p["planned"] for p in prog) or 1
    log = {"progress/done": done, "progress/planned": planned, "progress/pct": 100.0 * done / planned,
           "progress/unsupported_cells": len(meta["unsupported"]), "progress/rows": len(rows)}
    for p in prog:
        log[f"progress_engine/{p['batch']}/{p['engine']}"] = 100.0 * p["done"] / max(p["planned"], 1)
    table = wandb.Table(columns=COLUMNS, data=[[r.get(c) for c in COLUMNS] for r in rows])
    log["cells"] = table
    # one line chart per metric: x = context index, one series per engine
    for kind in ("code", "prose"):
        for metric, phase, key in (("cold_ttft_s", "cold", "ttft_s"), ("cold_decode_tps", "cold", "dec_tps"),
                                   ("warm_decode_tps", "warm", "dec_tps"), ("turn2_ttft_s", "turn2", "ttft_s"),
                                   ("turn2_decode_tps", "turn2", "dec_tps")):
            ys, keys = [], []
            for eng in fr.ENGINE_ORDER:
                pts = {r["ctx"]: r.get(key) for r in rows if r["engine"] == eng and r["phase"] == phase
                       and r["conc"] == 1 and r["kind"] == kind and r["out"] == 512
                       and r["family"] in ("core", "ctx") and r.get(key) is not None}
                if pts:
                    keys.append(eng)
                    ys.append([pts.get(c) for c in CTX_ORDER])
            if keys:
                xs = list(range(len(CTX_ORDER)))
                ys = [[(v if v is not None else float("nan")) for v in y] for y in ys]
                log[f"curves/{kind}/{metric}"] = wandb.plot.line_series(
                    xs=xs, ys=ys, keys=keys, title=f"{metric} vs context ({kind}); x = " + ",".join(CTX_ORDER),
                    xname="context (index)")
    for r in rows:
        if r["phase"] in ("conc", "mixed") and r.get("agg_tps") is not None:
            log[f"conc/{r['family']}/{r['ctx']}/c{r['conc']}/{r['engine']}"] = r["agg_tps"]
    run.log(log)  # wandb advances its own step; an explicit step below a resumed run step is dropped
    run.summary.update({"engines": meta["engines"], "updated": time.strftime("%Y-%m-%d %H:%M:%S")})


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

    run = wandb.init(project=a.project, name=a.name, id=a.name, resume="allow",
                     config={"raw": a.raw, "manifest": a.manifest}, dir=os.environ.get("WANDB_DIR", "/Volumes/P5Plus/yunshu-build/wandb"))
    step = int(run.step or 0)
    last_sig = None
    while True:
        try:
            # skip when no result file changed: a snapshot parses every file, and a CPU burst during a
            # job's quiet window would make the quiet gate refuse it
            sig = max((e.stat().st_mtime for d in os.scandir(a.raw) if d.is_dir() for e in os.scandir(d.path)), default=0)
            if sig != last_sig:
                snapshot(wandb, run, a.raw, a.manifest, step)
                step += 1
                last_sig = sig
        except Exception as exc:  # noqa: BLE001 - keep streaming; a half-written cell file is retried next round
            print(f"snapshot failed: {exc}", file=sys.stderr, flush=True)
        if a.once or not alive(a.until_pid):
            if not a.once:
                snapshot(wandb, run, a.raw, a.manifest, step)  # final upload after the run ended
            break
        time.sleep(a.interval)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
