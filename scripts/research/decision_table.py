"""Turn copy-width and DFlash-precision sweep outputs into a decision table.

    decision_table.py copy  /path/to/copy-32k-r0-0134 [more output stems ...]
    decision_table.py draft /path/to/draft-ab-1k-0134.jsonl [...]

Fail-closed: a file without a final ``complete`` record, a contended session,
or an arm whose engaged mode is not the expected one is listed under REJECTED
and contributes no number. Copy arms are named ``<stem>-w<rows>-r<rep>*.jsonl``
(bench_copy_width.py); rows 0 is copy-off and every other arm's token digests
must equal it.
"""

from __future__ import annotations

import glob
import json
import re
import statistics
import sys
from collections import defaultdict


def records(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append({"corrupt": True})
    return out


def verdict(recs: list[dict], mode: str | None) -> str | None:
    """None when the file is trustworthy, else the reason it is not."""
    done = [r for r in recs if r.get("complete")]
    if not done:
        return "no complete record"
    if any(r.get("corrupt") for r in recs):
        return "corrupt line"
    if any(r.get("contended") or r.get("success") is False for r in recs):
        return "contended / failed"
    if mode and done[-1].get("mode") not in (mode, None):
        return f"engaged mode {done[-1].get('mode')} != {mode}"
    return None


def copy_table(stems: list[str]) -> tuple[list[str], list[str]]:
    cells: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    shas: dict[tuple, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    rejected = []
    for stem in stems:
        for path in sorted(glob.glob(f"{stem}-w*-r*.jsonl")):
            m = re.search(r"-w(\d+)-r(\d+)", path)
            if not m:
                continue
            rows = int(m.group(1))
            recs = records(path)
            why = verdict(recs, "mtp")
            if why:
                rejected.append(f"{path}: {why}")
                continue
            for r in recs:
                if r.get("part") in ("decode", "agent") and "dec_tps" in r:
                    k = (r.get("ctx"), r.get("kind"), r.get("phase"), r.get("part"))
                    cells[k][rows].append(r["dec_tps"])
                    shas[k][rows].add(r.get("sha"))
    lines = ["ctx kind phase | rows: median tok/s (n) | digest == copy-off"]
    for k in sorted(cells, key=str):
        parts = []
        base = shas[k].get(0)
        for rows in sorted(cells[k]):
            xs = cells[k][rows]
            same = (
                ""
                if rows == 0 or not base
                else (" same" if shas[k][rows] == base else " DIGEST DIFFERS")
            )
            parts.append(f"{rows}: {statistics.median(xs):.1f} ({len(xs)}){same}")
        lines.append(" ".join(str(x) for x in k) + " | " + " | ".join(parts))
    return lines, rejected


def draft_table(paths: list[str]) -> tuple[list[str], list[str]]:
    rows: dict[tuple, list[float]] = defaultdict(list)
    parity, rejected = {}, []
    for path in paths:
        recs = records(path)
        why = verdict(recs, None)
        if why:
            rejected.append(f"{path}: {why}")
            continue
        for r in recs:
            if r.get("mode") in ("plain", "dflash") and "tps" in r:
                k = (
                    r["context"],
                    r["task"],
                    r["mode"],
                    r.get("bits"),
                    r.get("context_fused"),
                    r.get("selector"),
                )
                rows[k].append(r["tps"])
                if "parity" in r:
                    parity[k] = parity.get(k, True) and r["parity"]
    lines = ["context task mode bits fused selector | median tok/s (n) | parity"]
    for k in sorted(rows, key=str):
        lines.append(
            " ".join(map(str, k))
            + f" | {statistics.median(rows[k]):.1f} ({len(rows[k])}) | "
            + str(parity.get(k, "-"))
        )
    return lines, rejected


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("copy", "draft"):
        print(__doc__)
        return 2
    lines, rejected = (copy_table if argv[1] == "copy" else draft_table)(argv[2:])
    print("\n".join(lines))
    if rejected:
        print("REJECTED:\n  " + "\n  ".join(rejected))
    return int(bool(rejected) or len(lines) < 2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
