#!/usr/bin/env python3
"""Render the public docs/BENCHMARKS.md from the parity board (board.json). CPU only.

The board already applies the fail-closed rules (clean quiet M5 jobs, terminal-complete rows, engaged
mode proven, identical provenance across reps), so this file only formats it. A cell is "measured" with
>= 3 reps (median +- MAD); fewer reps are printed in parentheses as provisional and are never ranked.
No private paths are written.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ENGINE_LABELS = {
    "yunshu-new": "Yunshu",
    "tf-new": "TensorFold",
    "splash": "Splash",
    "omlx": "oMLX",
    "mtplx": "MTPLX",
    "mlxlm": "mlx-lm",
    "llamacpp": "llama.cpp",
}
ENGINE_ORDER = tuple(ENGINE_LABELS)
DIFFERENT_WEIGHTS = {"splash", "llamacpp"}
CTX_LABEL = {1024: "1K", 8192: "8K", 32768: "32K", 65536: "64K", 131072: "128K"}

TABLES = (
    ("ttft_cold_s", "Cold TTFT (seconds, lower is better)", "{:.3g}"),
    ("ttft_warm_s", "Repeated-request TTFT (seconds, lower is better)", "{:.3g}"),
    ("followup_ttft_s", "Follow-up TTFT (seconds, lower is better)", "{:.3g}"),
    ("decode_cold_tps", "Decode tok/s, cold request, 2048-token reply", "{:.3g}"),
    ("decode_warm_tps", "Decode tok/s, repeated request", "{:.3g}"),
    ("decode_turn2_tps", "Decode tok/s, follow-up turn (256-token reply)", "{:.3g}"),
    ("prefill_cold_tps", "Cold prefill tok/s (engine accounting only)", "{:.4g}"),
    ("memory_peak_gib", "Peak memory, GiB over pre-launch host baseline", "{:.3g}"),
    (
        "memory_idle_gib",
        "Idle memory 30 s after the group, GiB over baseline",
        "{:.3g}",
    ),
    ("conc8_agg_tps", "Aggregate tok/s, concurrent requests", "{:.3g}"),
    ("accuracy_needle", "Long-context needle accuracy (10 questions)", "{:.3g}"),
    ("agentic_session_s", "Agentic session seconds (lower is better)", "{:.3g}"),
)


def fmt_cell(stat, pattern):
    if not stat:
        return "unknown"
    if stat.get("status") == "measured":
        return f"{pattern.format(stat['median'])} ±{pattern.format(stat['mad'])} (n={len(stat['samples'])})"
    samples = [s["value"] for s in stat.get("samples", [])]
    if not samples:
        return "unknown"
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    return f"({pattern.format(median)}, n={len(samples)})"


def best_engines(row_engines, higher):
    measured = {
        e: s["median"] for e, s in row_engines.items() if s.get("status") == "measured"
    }
    if not measured:
        return set()
    target = (max if higher else min)(measured.values())
    return {e for e, v in measured.items() if v == target}


def render_table(board, metric, title, pattern, engines):
    rows = [i for i in board["items"] if i["metric"] == metric]
    lines, any_data = [], False
    for item in sorted(rows, key=lambda i: (i["ctx"], i["kind"])):
        if not any(item["engines"].get(e, {}).get("samples") for e in engines):
            continue
        any_data = True
        best = best_engines(item["engines"], item["higher_is_better"])
        cells = []
        for e in engines:
            text = fmt_cell(item["engines"].get(e), pattern)
            cells.append(f"**{text}**" if e in best else text)
        ctx = CTX_LABEL.get(item["ctx"], str(item["ctx"]))
        kind = (
            ""
            if item["kind"] in ("session", "needle", "conc8", "agentic")
            else f" {item['kind']}"
        )
        lines.append(f"| {ctx}{kind} | " + " | ".join(cells) + " |")
    if not any_data:
        return f"### {title}\n\nNo engine has a valid measurement yet: all cells are unknown.\n"
    head = (
        "| context | "
        + " | ".join(
            ENGINE_LABELS[e] + ("†" if e in DIFFERENT_WEIGHTS else "") for e in engines
        )
        + " |"
    )
    sep = "|---|" + "---|" * len(engines)
    return "\n".join([f"### {title}", "", head, sep, *lines, ""])


def render_head_to_head(board):
    h2h = board.get("head_to_head") or {}
    if not h2h:
        return "No head-to-head comparison is available yet.\n"
    lines = [
        "| vs | metric family | Yunshu wins | ties | losses | cells | evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for engine in ENGINE_ORDER:
        for family, c in sorted((h2h.get(engine) or {}).items()):
            lines.append(
                f"| {ENGINE_LABELS[engine]} | {family} | {c['win']} | {c['tie']} | {c['loss']} | {c['n']} | "
                + (
                    "some cells have fewer than 3 reps"
                    if c.get("provisional")
                    else "all cells have 3 or more reps"
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def versions(board):
    seen = {}
    for item in board["items"]:
        for engine, stat in item["engines"].items():
            for sample in stat.get("samples", []):
                seen.setdefault(
                    engine,
                    (
                        sample.get("version"),
                        sample.get("mode"),
                        sample.get("weights_vs_oQ4e"),
                    ),
                )
    return seen


def render(board, intro):
    engines = list(ENGINE_ORDER)
    out = [intro.rstrip(), "", "## Engine versions and configurations", ""]
    out += [
        "| engine | version | decode mode engaged | weights vs Yunshu's checkpoint |",
        "|---|---|---|---|",
    ]
    for engine, (version, mode, weights) in sorted(
        versions(board).items(), key=lambda kv: ENGINE_ORDER.index(kv[0])
    ):
        version = re.sub(
            r"\s*\((?:[^)]*snapshot014[^)]*|[^)]*own venv)\)", "", str(version)
        )
        out.append(f"| {ENGINE_LABELS[engine]} | {version} | {mode} | {weights} |")
    unmeasured = [ENGINE_LABELS[e] for e in engines if e not in versions(board)]
    if unmeasured:
        out.append("")
        out.append("No valid cell yet for: " + ", ".join(unmeasured) + ".")
    out += [
        "",
        "## Results",
        "",
        "Cells are median ±MAD (n = independent server sessions). A value in parentheses has fewer than "
        "3 reps: it is shown for information and is never ranked. **Bold** marks the best of the engines with "
        "3 or more reps. † = different weights (not the same checkpoint).",
        "",
    ]
    for metric, title, pattern in TABLES:
        out.append(render_table(board, metric, title, pattern, engines))
    out += [
        "## Head to head (Yunshu against each engine)",
        "",
        render_head_to_head(board),
    ]
    out += [
        "",
        "## Board verdict",
        "",
        board.get("summary", ""),
        "",
        f"`{board.get('verdict', '')}`",
        "",
    ]
    return "\n".join(out)


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--board", type=Path, default=root / "docs/research/parityboard/board.json"
    )
    ap.add_argument(
        "--intro", type=Path, default=root / "scripts/dev/benchmarks_intro.md"
    )
    ap.add_argument("--out", type=Path, default=root / "docs/BENCHMARKS.md")
    args = ap.parse_args()
    board = json.loads(args.board.read_text())
    args.out.write_text(render(board, args.intro.read_text()) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
