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

import argparse
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def job_evidence(directory: Path) -> dict[str, dict]:
    """Map exact declared outputs to queue evidence; ambiguous paths fail closed."""
    evidence: dict[str, dict] = {}
    for path in directory.glob("*.json"):
        job = json.loads(path.read_text())
        for output in job.get("outputs", []):
            evidence[output] = job if output not in evidence else {}
    return evidence


def verdict(recs: list[dict], mode: str | None, job: dict | None = None) -> str | None:
    """None when the file is trustworthy, else the reason it is not."""
    from device_evidence import require_same_device

    try:
        require_same_device(recs + ([job] if job else []), performance=True)
    except ValueError as exc:
        return str(exc)
    done = [r for r in recs if r.get("complete")]
    if not done or not recs[-1].get("complete"):
        return "no final complete record"
    if any(r.get("corrupt") for r in recs):
        return "corrupt line"
    if job is not None and (job.get("state") != "done" or job.get("rc") != 0):
        return "queue job did not succeed"
    reclassified = bool(
        job and job.get("reclassified") and job.get("contended") is False
    )
    if job and job.get("contended"):
        return "queue job contended"
    for row in recs:
        if row.get("error") or ("rc" in row and row["rc"] != 0):
            return "failed arm"
        if row.get("contended") and not reclassified:
            return "contended / failed"
        if row.get("success") is False:
            # Older harnesses latched the old CPU threshold into completion.
            # Queue reclassification cannot excuse a skipped/failed arm.
            if not (reclassified and row.get("contended") and not row.get("reason")):
                return "contended / failed"
    if any(r.get("parity") is False for r in recs):
        return "token parity failed"
    engaged = done[-1].get("mode")
    if mode:
        if isinstance(engaged, list):
            matches = bool(engaged) and all(
                re.search(rf"Speculative decoding:\s*{re.escape(mode)}\b", line, re.I)
                for line in engaged
            )
        else:
            matches = engaged == mode
        if not matches:
            return f"engaged mode {engaged} != {mode}"
    return None


def copy_table(
    stems: list[str], evidence: dict[str, dict] | None = None
) -> tuple[list[str], list[str]]:
    cells: dict[tuple, dict[tuple[int, bool], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    shas: dict[tuple, dict[tuple[int, bool], set]] = defaultdict(
        lambda: defaultdict(set)
    )
    rejected = []
    evidence = evidence or {}
    for stem in stems:
        summary = Path(stem + ".jsonl")
        if summary.exists():
            why = verdict(records(str(summary)), None, evidence.get(str(summary)))
            if why:
                rejected.append(f"{summary}: {why}")
                continue
        for path in sorted(glob.glob(f"{stem}-w*-r*.jsonl")):
            m = re.search(r"-w(\d+)-r(\d+)", path)
            if not m:
                continue
            rows = int(m.group(1))
            arm = (rows, Path(path).stem.endswith("-cost"))
            recs = records(path)
            why = verdict(recs, "mtp", evidence.get(path, evidence.get(str(summary))))
            if why:
                rejected.append(f"{path}: {why}")
                continue
            source_sha = next(
                (r.get("python_sha256") for r in recs if r.get("part") == "session"),
                None,
            )
            measured = [
                r
                for r in recs
                if r.get("part") in ("decode", "agent") and "dec_tps" in r
            ]
            if any(not r.get("sha") for r in measured):
                rejected.append(f"{path}: missing token digest")
                continue
            for r in recs:
                if r.get("part") in ("decode", "agent") and "dec_tps" in r:
                    k = (
                        r.get("ctx"),
                        r.get("kind"),
                        r.get("phase"),
                        r.get("part"),
                        source_sha,
                    )
                    cells[k][arm].append(r["dec_tps"])
                    shas[k][arm].add(r.get("sha"))
    lines = [
        "ctx kind phase part source | rows: median tok/s (n) | digest == baseline (copy-off when present)"
    ]
    for k in sorted(cells, key=str):
        parts = []
        baseline = (0, False) if (0, False) in shas[k] else min(shas[k])
        base = shas[k][baseline]
        for arm in sorted(cells[k]):
            rows, cost = arm
            xs = cells[k][arm]
            same = (
                ""
                if arm == baseline
                else (" same" if shas[k][arm] == base else " DIGEST DIFFERS")
            )
            name = f"{rows}-cost" if cost else str(rows)
            parts.append(f"{name}: {statistics.median(xs):.1f} ({len(xs)}){same}")
        lines.append(" ".join(str(x) for x in k) + " | " + " | ".join(parts))
    return lines, rejected


def draft_table(
    paths: list[str], evidence: dict[str, dict] | None = None
) -> tuple[list[str], list[str]]:
    rows: dict[tuple, list[float]] = defaultdict(list)
    parity, rejected = {}, []
    evidence = evidence or {}
    for path in paths:
        recs = records(path)
        why = verdict(recs, None, evidence.get(path))
        if why:
            rejected.append(f"{path}: {why}")
            continue
        source_sha = next(
            (r.get("python_sha256") for r in recs if r.get("part") == "snapshot"),
            None,
        )
        for r in recs:
            if r.get("mode") in ("plain", "dflash") and "tps" in r:
                k = (
                    r["context"],
                    r["task"],
                    r["mode"],
                    r.get("bits"),
                    r.get("context_fused"),
                    r.get("selector"),
                    r.get("compiled_conv", False),
                    source_sha,
                )
                rows[k].append(r["tps"])
                if "parity" in r:
                    parity[k] = parity.get(k, True) and r["parity"]
    lines = [
        "context task mode bits fused selector compiled_conv source | median tok/s (n) | parity"
    ]
    for k in sorted(rows, key=str):
        lines.append(
            " ".join(map(str, k))
            + f" | {statistics.median(rows[k]):.1f} ({len(rows[k])}) | "
            + str(parity.get(k, "-"))
        )
    return lines, rejected


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["copy", "draft"])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--gpuq-jobs", type=Path)
    args = parser.parse_args(argv[1:])
    evidence = job_evidence(args.gpuq_jobs) if args.gpuq_jobs else None
    lines, rejected = (copy_table if args.kind == "copy" else draft_table)(
        args.paths, evidence
    )
    print("\n".join(lines))
    if rejected:
        print("REJECTED:\n  " + "\n  ".join(rejected))
    return int(bool(rejected) or len(lines) < 2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
