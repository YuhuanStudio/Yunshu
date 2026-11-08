#!/usr/bin/env python3
"""Stream a fullcmp run into Weights & Biases as native W&B curves, unattended (CPU only; reads jsonl files).

Layout (one W&B group per raw root, so a project can hold several comparisons):
  - one run per ENGINE (run name = engine id). Its history holds one row per point, with W&B custom x-axes
    (define_metric step_metric), so the project workspace overlays every engine as its own colored line:
        x = context_k   (1, 4, 8, 16, 32, 64, 128):  cold_ttft_s/<kind>, warm_ttft_s/<kind>, turn2_ttft_s/<kind>,
                         decode_cold/<kind>, decode_warm/<kind>, decode_turn2/<kind>
        x = concurrency (2, 4, 8):  conc_agg_tps/distinct, conc_agg_tps/same, conc_slowest_ttft_s/distinct,
                         mixed_agg_tps, mixed_long_ttft_s
    Its summary holds the engine's version, spec mode, cells done and labels.
    Each round that sees new data rebuilds the engine runs (delete + re-log), so a curve always shows the current
    medians over reps instead of appending duplicate points.
  - one run named "progress" with time on the x axis: cells done, % done per batch, unsupported cells, and a
    `cells` table (every cell x phase: medians, rep spread, cached tokens, labels).

    venv-wandb/bin/python scripts/dev/fullcmp_wandb.py --raw <raw root> --manifest <run>/manifest.jsonl \
        --project yunshu-fullcmp --group <name> --until-pid <driver pid>

Run under `taskpolicy -b nice -n 19` so it never shows up in a measured job's quiet gate. Needs `wandb login`
once. Uploads aggregated numbers and labels only (no prompts, outputs or logs).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullcmp_report as fr  # noqa: E402

CTX_K = {"1K": 1, "4K": 4, "8K": 8, "16K": 16, "32K": 32, "64K": 64, "128K": 128}
COLUMNS = ["batch", "family", "engine", "ctx", "conc", "kind", "out", "variant", "phase", "n",
           "ttft_s", "dec_tps", "agg_tps", "dec_min", "dec_max", "ttft_min", "ttft_max", "cached",
           "ttft_s_spread_pct", "dec_tps_spread_pct", "agg_tps_spread_pct", "label"]
CTX_METRICS = [  # (metric prefix, phase, field)
    ("cold_ttft_s", "cold", "ttft_s"), ("warm_ttft_s", "warm", "ttft_s"), ("turn2_ttft_s", "turn2", "ttft_s"),
    ("decode_cold", "cold", "dec_tps"), ("decode_warm", "warm", "dec_tps"), ("decode_turn2", "turn2", "dec_tps"),
]


def engine_points(rows, engine):
    """History rows for one engine: one row per context point, one per concurrency point."""
    by_ctx = {}
    for r in rows:
        if (r["engine"] != engine or r["conc"] != 1 or r["out"] != 512 or r["family"] not in ("core", "ctx")
                or r["ctx"] not in CTX_K):
            continue
        for prefix, phase, field in CTX_METRICS:
            if r["phase"] == phase and r.get(field) is not None:
                by_ctx.setdefault(CTX_K[r["ctx"]], {})[f"{prefix}/{r['kind']}"] = r[field]
    by_conc = {}
    for r in rows:
        if r["engine"] != engine or r["ctx"] != "1K" and r["phase"] != "mixed":
            continue
        if r["phase"] == "conc" and r.get("agg_tps") is not None:
            tag = "same" if r["family"] == "conc-same" else "distinct"
            by_conc.setdefault(r["conc"], {})[f"conc_agg_tps/{tag}"] = r["agg_tps"]
            if r.get("ttft_max") is not None:
                by_conc[r["conc"]][f"conc_slowest_ttft_s/{tag}"] = r["ttft_max"]
        if r["phase"] == "mixed" and r.get("agg_tps") is not None:
            by_conc.setdefault(r["conc"], {})["mixed_agg_tps"] = r["agg_tps"]
            if r.get("ttft_max") is not None:
                by_conc[r["conc"]]["mixed_long_ttft_s"] = r["ttft_max"]
    out = [{"context_k": k, **v} for k, v in sorted(by_ctx.items())]
    out += [{"concurrency": n, **v} for n, v in sorted(by_conc.items())]
    return out


def rebuild_engine_runs(wandb, api, project, entity, group, rows, meta, prog):
    path = f"{entity}/{project}" if entity else project
    for old in api.runs(path, filters={"group": group, "jobType": "engine"}):
        old.delete()
    for eng in fr.ENGINE_ORDER:
        pts = engine_points(rows, eng)
        if not pts:
            continue
        run = wandb.init(project=project, entity=entity, group=group, job_type="engine", name=eng,
                         reinit="create_new", config={"engine": eng, **(meta["engines"].get(eng) or {})})
        run.define_metric("context_k")
        run.define_metric("concurrency")
        for prefix, _, _ in CTX_METRICS:
            run.define_metric(f"{prefix}/*", step_metric="context_k")
        for name in ("conc_agg_tps/*", "conc_slowest_ttft_s/*", "mixed_agg_tps", "mixed_long_ttft_s"):
            run.define_metric(name, step_metric="concurrency")
        for p in pts:
            run.log(p)
        done = sum(p["done"] for p in prog if p["engine"] == eng)
        planned = sum(p["planned"] for p in prog if p["engine"] == eng)
        labels = sorted({r["label"] for r in rows if r["engine"] == eng and r.get("label")})
        run.summary.update({"cells_done": done, "cells_planned": planned, "labels": "; ".join(labels)[:2000]})
        run.finish()


def log_progress(wandb, prun, rows, meta, prog):
    done = sum(p["done"] for p in prog)
    planned = sum(p["planned"] for p in prog) or 1
    log = {"progress/cells_done": done, "progress/cells_planned": planned, "progress/pct": 100.0 * done / planned,
           "progress/unsupported_cells": len(meta["unsupported"])}
    for b in ("B1", "B2"):
        bd = sum(p["done"] for p in prog if p["batch"] == b)
        bp = sum(p["planned"] for p in prog if p["batch"] == b) or 1
        log[f"progress/{b}_pct"] = 100.0 * bd / bp
    log["cells"] = wandb.Table(columns=COLUMNS, data=[[r.get(c) for c in COLUMNS] for r in rows])
    prun.log(log)


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
    ap.add_argument("--entity", default=None)
    ap.add_argument("--group", required=True)
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--until-pid", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    api = wandb.Api()
    entity = a.entity or api.default_entity
    last_sig = None
    final = False
    while True:
        try:
            sig = max((e.stat().st_mtime for d in os.scandir(a.raw) if d.is_dir() for e in os.scandir(d.path)),
                      default=0)
            if sig != last_sig or final:
                rows, meta = fr.collect(a.raw)
                prog = fr.progress(a.manifest, a.raw)
                rebuild_engine_runs(wandb, api, a.project, entity, a.group, rows, meta, prog)
                prun = wandb.init(project=a.project, entity=entity, group=a.group, job_type="progress",
                                  name="progress", id=f"progress-{a.group}", resume="allow", reinit="create_new")
                log_progress(wandb, prun, rows, meta, prog)
                prun.finish()
                last_sig = sig
                print(f"{time.strftime('%H:%M:%S')} uploaded {len(rows)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep streaming; a half-written cell file is retried next round
            print(f"round failed: {exc!r}", file=sys.stderr, flush=True)
        if a.once or final:
            break
        if not alive(a.until_pid):
            final = True  # one last upload after the run ended
            continue
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
