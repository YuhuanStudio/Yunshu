"""Re-estimate run-to-run variability of decode tok/s and TTFT from the logs we already have.

Reads (read-only): raw bench / run ``*.jsonl`` files under a runs directory (default
``docs/research/runs`` of the main checkout; private data, never written to) and the numbers
quoted in ``docs/reports/PERF_TREND.md``. Prints sigma estimates and the per-arm sample size
needed to detect a 3% and a 10% change.

Method (nothing is imputed; a cell or metric without enough data is reported as missing):

- A *cell* is the rows of one file that share every identifying field (kind, case, task, pp,
  context, label, layer, note, type, concurrency, ...) except the repeat counters.
- Same-prompt variability: cells with n >= 2 repeats. Per-cell CV = sd / mean; pooled
  CV = sqrt(sum(df * cv^2) / sum(df)) with df = n - 1.
- Different-prompt variability: within one file and one configuration group, the CV of the
  cell means across prompts (cells of different case / task / context).
- The pooled CV is dominated by cells that mix regimes the harness did not label (cold and
  warm, interleaved configurations), so the median cell CV is reported next to it: it is the
  controlled-repeat noise floor, the pooled value the realistic ceiling.
- Sample size per arm for a two-sided alpha = 0.05, power 0.8 two-sample test of a relative
  change d: n = 15.7 (sigma / d)^2; paired on the same prompt: n = 7.85 (sigma_d / d)^2 with
  sigma_d = sqrt(2) sigma.
- PERF_TREND ranges ("92.0-93.7 tok/s") have no recorded n: they are listed separately with
  the range / mean ratio and are NOT merged into the pooled estimate.

Usage: python scripts/research/serving/profile_history.py [--runs DIR] [--perf-trend FILE] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MAIN = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")

# metric name in the rows -> (reported metric, scale to the reported unit)
METRICS: dict[str, tuple[str, float]] = {
    "decode_tps": ("decode_tps", 1.0),
    "decode_tok_s": ("decode_tps", 1.0),
    "x_decode_tps": ("decode_tps", 1.0),
    "tps": ("decode_tps", 1.0),
    "ttft_s": ("ttft_ms", 1000.0),
    "ttft_ms": ("ttft_ms", 1.0),
}
# fields that name the prompt (vary between cells of one configuration)
PROMPT_KEYS = (
    "case",
    "task",
    "pp",
    "context",
    "i",
    "type",
    "prompt_tokens",
    "title",
    "tg",
)
# repeat counters / outcomes: never part of a cell identity
SKIP_KEYS = {
    "rep",
    "n",
    "t",
    "t_start",
    "t_end",
    "started",
    "ok",
    "footprint_gib",
    "rss_gib",
}
MIN_DF = 5


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


CONFIG_INTS = (
    "block",
    "depth",
    "rows",
    "concurrency",
    "k",
    "tokens",
    "max_tokens",
    "bs",
    "draft_block_size",
)
OUTCOME = {
    "same_as_first",
    "ok",
    "correct",
    "finish",
    "finish_reason",
    "status",
    "done",
    "http_status",
    "expected",
    "got",
    "pred",
    "answer",
    "text_head",
    "content",
    "cold",
}


def _scalars(row: dict, metric_keys: set[str]) -> tuple[list, list]:
    """(configuration items, prompt items) of one row: short strings, bools and the named
    integer knobs; metrics, counters and outcomes are not part of an identity."""
    config, prompt = [], []
    for k in sorted(row):
        v = row[k]
        if k in metric_keys or k in SKIP_KEYS or k in OUTCOME:
            continue
        ok = (isinstance(v, str) and len(v) <= 64) or isinstance(v, bool)
        ok = ok or (_is_num(v) and (k in PROMPT_KEYS or k in CONFIG_INTS))
        if not ok:
            continue
        (prompt if k in PROMPT_KEYS else config).append((k, v))
    return config, prompt


def collect(runs: Path) -> dict[str, dict[tuple, dict[tuple, list[float]]]]:
    """metric -> (file, configuration) -> prompt -> values. Rows without a metric (the
    ``meta`` lines a harness writes first) set configuration for the rows after them."""
    out: dict[str, dict[tuple, dict[tuple, list[float]]]] = {
        "decode_tps": defaultdict(lambda: defaultdict(list)),
        "ttft_ms": defaultdict(lambda: defaultdict(list)),
    }
    for f in sorted(runs.rglob("*.jsonl")):
        try:
            lines = f.read_text().splitlines()
        except OSError:
            continue
        ctx: dict[str, Any] = {}
        for ln in lines:
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            mkeys = {k for k in row if k in METRICS}
            if not mkeys:
                cfg, _ = _scalars(row, set())
                ctx.update(dict(cfg))
                continue
            seen: set[str] = set()
            config, prompt = _scalars(row, mkeys)
            group = tuple(sorted({**ctx, **dict(config)}.items()))
            for k in sorted(mkeys):
                name, scale = METRICS[k]
                v = row[k]
                if name in seen or not _is_num(v) or v <= 0:
                    continue
                seen.add(name)
                out[name][(str(f.relative_to(runs)), group)][tuple(prompt)].append(
                    float(v) * scale
                )
    return out


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = statistics.fmean(values)
    return statistics.stdev(values) / m if m > 0 else None


def n_per_arm(sigma: float, delta: float, paired: bool = False) -> int | None:
    """Per-arm n to detect a relative change ``delta`` (two-sided alpha 0.05, power 0.8)."""
    if not (sigma > 0 and delta > 0):
        return None
    if paired:
        return math.ceil(7.85 * (math.sqrt(2) * sigma / delta) ** 2)
    return math.ceil(15.7 * (sigma / delta) ** 2)


def summarize_metric(cells: dict[tuple, dict[tuple, list[float]]]) -> dict:
    same: list[tuple[float, int]] = []  # (cv, df)
    for prompts in cells.values():
        for vals in prompts.values():
            c = cv(vals)
            if c is not None:
                same.append((c, len(vals) - 1))
    diff: list[float] = []
    for prompts in cells.values():
        means = [statistics.fmean(v) for v in prompts.values() if v]
        c = cv(means) if len(means) >= 3 else None
        if c is not None:
            diff.append(c)
    res: dict[str, Any] = {}
    df = sum(d for _, d in same)
    if df >= MIN_DF:
        pooled = math.sqrt(sum(d * c * c for c, d in same) / df)
        res["same_prompt"] = {
            "cells": len(same),
            "df": df,
            "pooled_cv": round(pooled, 4),
            "median_cell_cv": round(statistics.median(c for c, _ in same), 4),
            "p90_cell_cv": round(
                sorted(c for c, _ in same)[int(0.9 * (len(same) - 1))], 4
            ),
        }
        for d in (0.03, 0.10):
            res["same_prompt"][f"n_per_arm_{int(d * 100)}pct"] = n_per_arm(pooled, d)
            res["same_prompt"][f"n_paired_{int(d * 100)}pct"] = n_per_arm(
                pooled, d, True
            )
            res["same_prompt"][f"n_per_arm_{int(d * 100)}pct_at_median_cv"] = n_per_arm(
                statistics.median(c for c, _ in same), d
            )
    else:
        res["same_prompt"] = {"missing": f"df={df} < {MIN_DF}: not enough repeats"}
    if len(diff) >= 3:
        med = statistics.median(diff)
        res["different_prompts"] = {
            "groups": len(diff),
            "median_cv": round(med, 4),
            "mean_cv": round(statistics.fmean(diff), 4),
        }
        for d in (0.03, 0.10):
            res["different_prompts"][f"n_per_arm_{int(d * 100)}pct"] = n_per_arm(med, d)
    else:
        res["different_prompts"] = {"missing": f"{len(diff)} groups < 3"}
    return res


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)\s*tok/s")


def perf_trend_ranges(path: Path) -> list[dict]:
    """Quoted ``a-b tok/s`` ranges; each is a range / midpoint ratio, n unknown."""
    try:
        text = path.read_text()
    except OSError:
        return []
    rows = []
    for ln in text.splitlines():
        for a, b in _RANGE.findall(ln):
            lo, hi = float(a), float(b)
            if 0 < lo < hi and hi / lo < 3:
                rows.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "range_over_mid": round((hi - lo) / ((hi + lo) / 2), 4),
                    }
                )
    return rows


def build_report(runs: Path, perf_trend: Path) -> dict:
    data = collect(runs)
    report: dict[str, Any] = {"runs_dir": str(runs), "metrics": {}}
    for name, cells in data.items():
        report["metrics"][name] = summarize_metric(cells)
        report["metrics"][name]["files"] = len({k[0] for k in cells})
    ranges = perf_trend_ranges(perf_trend)
    if ranges:
        r = [x["range_over_mid"] for x in ranges]
        report["perf_trend_ranges"] = {
            "count": len(r),
            "median_range_over_mid": round(statistics.median(r), 4),
            "note": "n per range is not recorded; sigma is not derived from these",
        }
    else:
        report["perf_trend_ranges"] = {"missing": "no ranges parsed"}
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=MAIN / "docs/research/runs")
    ap.add_argument(
        "--perf-trend", type=Path, default=MAIN / "docs/reports/PERF_TREND.md"
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not args.runs.is_dir():
        print(f"runs directory not found: {args.runs}", file=sys.stderr)
        return 1
    report = build_report(args.runs, args.perf_trend)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
